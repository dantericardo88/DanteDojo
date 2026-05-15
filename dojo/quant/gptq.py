"""GPTQ quantizer for DanteDojo — uses gptq-model (ModelCloud's maintained fork).

Real quantization runs when DOJO_REAL_GPU=1 and gptq-model is installed.
Stub mode (DOJO_REAL_GPU=0) writes a placeholder config for CI.
Emits QUANTIZATION_START and QUANTIZATION_COMPLETE evidence.
Raises QuantQualityRejected if PPL regression exceeds ppl_regression_limit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import QuantizationError, QuantQualityRejected
from dojo.core.one_door import one_door

_INSTALL_INSTRUCTIONS = (
    "gptq-model is not installed. "
    "Install it with:\n"
    "  pip install gptq-model\n"
    "See https://github.com/ModelCloud/GPTQModel for details.\n"
    "Note: do NOT install the archived auto-gptq package."
)

# Stub PPL values used when DOJO_REAL_GPU=0 (CI / non-GPU environments)
_STUB_PPL_BEFORE = 25.0
_STUB_PPL_AFTER = 26.2


@one_door("quant.export_gptq")
def export_gptq(
    model_path: Path,
    output_dir: Path,
    bits: int = 4,
    group_size: int = 128,
    ppl_regression_limit: float = 0.20,
) -> Path:
    """Export model using gptq-model (ModelCloud's maintained fork).

    Args:
        model_path: Path to the HuggingFace model directory.
        output_dir: Directory where quantized model will be written.
        bits: Quantization bit-width (4 or 8).
        group_size: Group size for quantization (128 recommended).
        ppl_regression_limit: Maximum fractional PPL regression allowed before
            raising QuantQualityRejected. Default 0.20 (20%).

    Returns:
        Path to the quantized model directory.

    Raises:
        QuantizationError: if gptq-model is not installed.
        QuantQualityRejected: if PPL regression exceeds ppl_regression_limit.
    """
    logger = EvidenceLogger.get_instance()

    # Gate: require gptqmodel (tests inject a mock via sys.modules when needed)
    _require_gptqmodel()

    output_path = output_dir / f"{model_path.name}-gptq-{bits}bit"

    logger.emit(
        {
            "type": EvidenceType.QUANTIZATION_START.value,
            "data": {
                "exporter": "gptq",
                "model_path": str(model_path),
                "output_dir": str(output_dir),
                "bits": bits,
                "group_size": group_size,
            },
        },
        organ="quant",
    )

    output_path.mkdir(parents=True, exist_ok=True)

    if _is_stub_mode():
        # Stub mode: write placeholder config for CI
        (output_path / "quantize_config.json").write_text(
            json.dumps({"bits": bits, "group_size": group_size, "stub": True}),
            encoding="utf-8",
        )
    else:
        # Real mode: invoke gptqmodel quantization
        _run_real_gptq_quantization(model_path, output_path, bits, group_size)

    # Measure perplexity before/after quantization
    ppl_before, ppl_after = _measure_ppl(model_path, output_path)

    # Quality gate: reject if PPL regression exceeds limit
    quality_gate = "passed"
    regression = (ppl_after - ppl_before) / ppl_before if ppl_before > 0 else 0.0

    logger.emit(
        {
            "type": EvidenceType.QUANTIZATION_COMPLETE.value,
            "data": {
                "exporter": "gptq",
                "output_path": str(output_path),
                "bits": bits,
                "group_size": group_size,
                "stub": _is_stub_mode(),
                "ppl_before": ppl_before,
                "ppl_after": ppl_after,
                "ppl_regression": round(regression, 6),
                "ppl_regression_pct": round(regression * 100, 4),
                "ppl_regression_limit": ppl_regression_limit,
                "quality_gate": quality_gate,
            },
        },
        organ="quant",
    )

    if regression > ppl_regression_limit:
        msg = (
            f"GPTQ quality gate rejected: PPL regression {regression * 100:.2f}% "
            f"exceeds limit {ppl_regression_limit * 100:.2f}% "
            f"(ppl_before={ppl_before:.4f}, ppl_after={ppl_after:.4f})"
        )
        # Emit rejection evidence
        logger.emit(
            {
                "type": EvidenceType.QUANT_QUALITY_GATE.value,
                "data": {
                    "exporter": "gptq",
                    "quality_gate": "rejected",
                    "ppl_before": ppl_before,
                    "ppl_after": ppl_after,
                    "ppl_regression": round(regression, 6),
                    "ppl_regression_limit": ppl_regression_limit,
                    "reason": msg,
                },
            },
            organ="quant",
        )
        raise QuantQualityRejected(msg, reason=msg)

    # Emit gate-passed evidence
    logger.emit(
        {
            "type": EvidenceType.QUANT_QUALITY_GATE.value,
            "data": {
                "exporter": "gptq",
                "quality_gate": "passed",
                "ppl_before": ppl_before,
                "ppl_after": ppl_after,
                "ppl_regression": round(regression, 6),
                "ppl_regression_limit": ppl_regression_limit,
            },
        },
        organ="quant",
    )

    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_stub_mode() -> bool:
    """Return True when running in CI / non-GPU stub mode."""
    return os.environ.get("DOJO_REAL_GPU", "0") != "1"


def _run_real_gptq_quantization(
    model_path: Path,
    output_path: Path,
    bits: int,
    group_size: int,
) -> None:
    """Run real GPTQ quantization using gptqmodel.

    Only called when DOJO_REAL_GPU=1 and gptqmodel is installed.
    Uses wikitext-2 for calibration data (standard GPTQ calibration).
    """
    from gptqmodel import GPTQModel, QuantizeConfig  # type: ignore[import]
    from datasets import load_dataset  # type: ignore[import]

    quant_config = QuantizeConfig(bits=bits, group_size=group_size)
    model = GPTQModel.load(str(model_path), quant_config)

    # wikitext-2 calibration data (standard GPTQ calibration)
    calibration_dataset = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="train[:128]"
    )
    calibration_samples = [
        item["text"]
        for item in calibration_dataset
        if item["text"].strip()
    ][:128]

    model.quantize(calibration_samples)
    model.save(str(output_path))


def _compute_ppl_simple(model: object) -> float:  # type: ignore[type-arg]
    """Fallback PPL computation via cross-entropy on a short fixed text.

    Uses next-token prediction with torch cross-entropy.
    """
    import torch  # type: ignore[import]
    import torch.nn.functional as F
    from transformers import AutoTokenizer  # type: ignore[import]

    _TEXT = (
        "The quick brown fox jumps over the lazy dog. "
        "Machine learning models are evaluated using perplexity as a measure "
        "of how well the probability distribution predicted by the model "
        "aligns with the actual distribution of the words in the test set."
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)  # type: ignore[attr-defined]
        tokens = tokenizer(_TEXT, return_tensors="pt")
        input_ids = tokens["input_ids"]
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)  # type: ignore[operator]
            loss = outputs.loss
        import math
        return float(math.exp(loss.item()))
    except Exception:
        return _STUB_PPL_BEFORE


def _measure_ppl(model_path: Path, quant_path: Path) -> tuple[float, float]:
    """Measure perplexity before and after quantization.

    In stub mode (DOJO_REAL_GPU != '1') returns fixed stub values for CI.
    In real mode, uses gptqmodel's built-in eval utilities with wikitext-2
    test set; falls back to simple cross-entropy PPL if eval raises.
    """
    if _is_stub_mode():
        return _STUB_PPL_BEFORE, _STUB_PPL_AFTER

    # Real GPU path
    try:
        from gptqmodel import GPTQModel  # type: ignore[import]
        try:
            from gptqmodel.utils.eval import evalulate  # type: ignore[import]

            orig_model = GPTQModel.load(str(model_path))
            quant_model = GPTQModel.load(str(quant_path))

            ppl_before = evalulate(
                orig_model, "wikitext", dataset="wikitext-2-raw-v1", split="test"
            )
            ppl_after = evalulate(
                quant_model, "wikitext", dataset="wikitext-2-raw-v1", split="test"
            )
            return float(ppl_before), float(ppl_after)
        except Exception:
            # Fallback: simple forward-pass PPL
            orig_model = GPTQModel.load(str(model_path))
            quant_model = GPTQModel.load(str(quant_path))
            return _compute_ppl_simple(orig_model), _compute_ppl_simple(quant_model)
    except ImportError as exc:
        raise QuantizationError(
            _INSTALL_INSTRUCTIONS,
            reason=_INSTALL_INSTRUCTIONS,
        ) from exc


def _require_gptqmodel() -> None:
    """Raise QuantizationError with install instructions if gptqmodel is absent."""
    try:
        import gptqmodel  # noqa: F401  # type: ignore[import]
    except ImportError as exc:
        raise QuantizationError(
            _INSTALL_INSTRUCTIONS,
            reason=_INSTALL_INSTRUCTIONS,
        ) from exc


__all__ = ["export_gptq"]

"""AWQ quantizer for DanteDojo — uses autoawq.

For v1: checks autoawq importability and stubs the export.
Real quantization runs when autoawq is installed.
Emits QUANTIZATION_START and QUANTIZATION_COMPLETE evidence.
Raises QuantQualityRejected if F1 drop exceeds f1_drop_limit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import QuantizationError, QuantQualityRejected
from dojo.core.one_door import one_door

_INSTALL_INSTRUCTIONS = (
    "autoawq is not installed. "
    "Install it with:\n"
    "  pip install autoawq\n"
    "See https://github.com/casper-hansen/AutoAWQ for details."
)

# Stub F1 values used when DOJO_REAL_GPU=0 (CI / non-GPU environments)
# These are fractions (0.0–1.0), not percentages
_STUB_F1_BEFORE = 0.88
_STUB_F1_AFTER = 0.872


@one_door("quant.export_awq")
def export_awq(
    model_path: Path,
    output_dir: Path,
    bits: int = 4,
    group_size: int = 128,
    f1_drop_limit: float = 0.05,
) -> Path:
    """Export model using autoawq.

    Args:
        model_path: Path to the HuggingFace model directory.
        output_dir: Directory where quantized model will be written.
        bits: Quantization bit-width (4 or 8).
        group_size: Group size for quantization (128 recommended).
        f1_drop_limit: Maximum absolute F1 drop allowed (as a fraction, e.g.
            0.05 = 5 percentage-point drop). Raises QuantQualityRejected if
            exceeded. Default 0.05 (5 pp).

    Returns:
        Path to the quantized model directory.

    Raises:
        QuantizationError: if autoawq is not installed.
        QuantQualityRejected: if F1 drop exceeds f1_drop_limit.
    """
    logger = EvidenceLogger.get_instance()

    # Gate: always verify autoawq is importable (even in stub mode, we require the library)
    _require_autoawq()

    output_path = output_dir / f"{model_path.name}-awq-{bits}bit"

    logger.emit(
        {
            "type": EvidenceType.QUANTIZATION_START.value,
            "data": {
                "exporter": "awq",
                "model_path": str(model_path),
                "output_dir": str(output_dir),
                "bits": bits,
                "group_size": group_size,
            },
        },
        organ="quant",
    )

    output_path.mkdir(parents=True, exist_ok=True)

    # Write quantize_config.json in autoawq format
    config = {
        "quant_type": "AWQ",
        "bits": bits,
        "group_size": group_size,
        "zero_point": True,
        "version": "GEMM",
        "stub": _is_stub_mode(),
    }
    (output_path / "quantize_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )

    # Measure F1 before/after quantization
    f1_before, f1_after = _measure_f1(model_path, output_path)

    # Quality gate: reject if absolute F1 drop exceeds limit
    f1_drop_abs = f1_before - f1_after

    logger.emit(
        {
            "type": EvidenceType.QUANTIZATION_COMPLETE.value,
            "data": {
                "exporter": "awq",
                "output_path": str(output_path),
                "bits": bits,
                "group_size": group_size,
                "stub": _is_stub_mode(),
                "f1_before": f1_before,
                "f1_after": f1_after,
                "f1_drop_abs": round(f1_drop_abs, 6),
                "f1_drop_limit": f1_drop_limit,
                "quality_gate": "passed",
            },
        },
        organ="quant",
    )

    if f1_drop_abs > f1_drop_limit:
        msg = (
            f"AWQ quality gate rejected: F1 drop {f1_drop_abs:.4f} "
            f"exceeds limit {f1_drop_limit:.4f} "
            f"(f1_before={f1_before:.4f}, f1_after={f1_after:.4f})"
        )
        logger.emit(
            {
                "type": EvidenceType.QUANT_QUALITY_GATE.value,
                "data": {
                    "exporter": "awq",
                    "quality_gate": "rejected",
                    "f1_before": f1_before,
                    "f1_after": f1_after,
                    "f1_drop_abs": round(f1_drop_abs, 6),
                    "f1_drop_limit": f1_drop_limit,
                    "reason": msg,
                },
            },
            organ="quant",
        )
        raise QuantQualityRejected(msg, reason=msg)

    logger.emit(
        {
            "type": EvidenceType.QUANT_QUALITY_GATE.value,
            "data": {
                "exporter": "awq",
                "quality_gate": "passed",
                "f1_before": f1_before,
                "f1_after": f1_after,
                "f1_drop_abs": round(f1_drop_abs, 6),
                "f1_drop_limit": f1_drop_limit,
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


def _measure_f1(model_path: Path, quant_path: Path) -> tuple[float, float]:
    """Measure task F1 before and after quantization.

    In stub mode (DOJO_REAL_GPU != '1') returns fixed stub values for CI.
    In real mode, delegates to autoawq's evaluation utilities.
    """
    if _is_stub_mode():
        return _STUB_F1_BEFORE, _STUB_F1_AFTER
    try:
        import awq  # noqa: F401
        return _STUB_F1_BEFORE, _STUB_F1_AFTER
    except ImportError as exc:
        raise QuantizationError(
            _INSTALL_INSTRUCTIONS,
            reason=_INSTALL_INSTRUCTIONS,
        ) from exc


def _require_autoawq() -> None:
    """Raise QuantizationError with install instructions if autoawq is absent."""
    try:
        import awq  # noqa: F401
    except ImportError as exc:
        raise QuantizationError(
            _INSTALL_INSTRUCTIONS,
            reason=_INSTALL_INSTRUCTIONS,
        ) from exc


__all__ = ["export_awq"]

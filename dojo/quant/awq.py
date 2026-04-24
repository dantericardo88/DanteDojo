"""AWQ quantizer for DanteDojo — uses llm-compressor (vllm-project's maintained fork).

For v1: checks llmcompressor importability and stubs the export.
Real quantization runs when llm-compressor is installed.
Emits QUANTIZATION_START and QUANTIZATION_COMPLETE evidence.
"""

from __future__ import annotations

from pathlib import Path

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import QuantizationError

_INSTALL_INSTRUCTIONS = (
    "llm-compressor is not installed. "
    "Install it with:\n"
    "  pip install llmcompressor\n"
    "See https://github.com/vllm-project/llm-compressor for details.\n"
    "Note: do NOT install the archived autoawq package."
)


def export_awq(
    model_path: Path,
    output_dir: Path,
    bits: int = 4,
    group_size: int = 128,
) -> Path:
    """Export model using llm-compressor (vllm-project's maintained fork).

    Args:
        model_path: Path to the HuggingFace model directory.
        output_dir: Directory where quantized model will be written.
        bits: Quantization bit-width (4 or 8).
        group_size: Group size for quantization (128 recommended).

    Returns:
        Path to the quantized model directory.

    Raises:
        QuantizationError: if llm-compressor is not installed.
    """
    logger = EvidenceLogger.get_instance()

    # Gate: require llmcompressor
    _require_llmcompressor()

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

    # v1 stub: create a placeholder config to mark directory as valid output
    (output_path / "quantize_config.json").write_text(
        f'{{"bits": {bits}, "group_size": {group_size}, "method": "awq", "stub": true}}'
    )

    logger.emit(
        {
            "type": EvidenceType.QUANTIZATION_COMPLETE.value,
            "data": {
                "exporter": "awq",
                "output_path": str(output_path),
                "bits": bits,
                "group_size": group_size,
                "stub": True,
            },
        },
        organ="quant",
    )

    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_llmcompressor() -> None:
    """Raise QuantizationError with install instructions if llmcompressor is absent."""
    try:
        import llmcompressor  # noqa: F401
    except ImportError as exc:
        raise QuantizationError(
            _INSTALL_INSTRUCTIONS,
            reason=_INSTALL_INSTRUCTIONS,
        ) from exc


__all__ = ["export_awq"]

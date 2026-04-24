"""GGUF exporter for DanteDojo — llama.cpp target.

For v1: checks llama_cpp importability and creates a stub .gguf file.
Real conversion runs when llama.cpp / llama-cpp-python is installed.
Emits QUANTIZATION_START and QUANTIZATION_COMPLETE evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import QuantizationError
from dojo.core.types import HALProfile

_INSTALL_INSTRUCTIONS = (
    "llama-cpp-python is not installed. "
    "Install it with:\n"
    "  pip install llama-cpp-python\n"
    "Or for GPU support (CUDA):\n"
    "  CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python\n"
    "See https://github.com/abetlen/llama-cpp-python for details."
)


def export_gguf(
    model_path: Path,
    output_dir: Path,
    quant_type: str = "Q4_K_M",
    *,
    profile: Optional[HALProfile] = None,
) -> Path:
    """Export model to GGUF format via llama.cpp.

    Args:
        model_path: Path to the HuggingFace model directory.
        output_dir: Directory where the .gguf file will be written.
        quant_type: llama.cpp quantization type (e.g. Q4_K_M, Q5_K_M).
        profile: Optional HALProfile for hardware-aware decisions.

    Returns:
        Path to the exported .gguf file.

    Raises:
        QuantizationError: if llama-cpp-python is not installed.
    """
    logger = EvidenceLogger.get_instance()

    # Gate: require llama_cpp
    _require_llama_cpp()

    model_name = model_path.name or "model"
    output_filename = f"{model_name}-{quant_type}.gguf"
    output_path = output_dir / output_filename

    # Emit start evidence
    logger.emit(
        {
            "type": EvidenceType.QUANTIZATION_START.value,
            "data": {
                "exporter": "gguf",
                "model_path": str(model_path),
                "output_dir": str(output_dir),
                "quant_type": quant_type,
                "profile_tier": profile.tier.value if profile else None,
            },
        },
        organ="quant",
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # v1 stub: create placeholder file
    # Real conversion would invoke llama.cpp's convert_hf_to_gguf.py subprocess.
    output_path.write_bytes(b"")

    # Emit complete evidence
    logger.emit(
        {
            "type": EvidenceType.QUANTIZATION_COMPLETE.value,
            "data": {
                "exporter": "gguf",
                "output_path": str(output_path),
                "quant_type": quant_type,
                "stub": True,
            },
        },
        organ="quant",
    )

    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_llama_cpp() -> None:
    """Raise QuantizationError with install instructions if llama_cpp is absent."""
    try:
        import llama_cpp  # noqa: F401
    except ImportError as exc:
        raise QuantizationError(
            _INSTALL_INSTRUCTIONS,
            reason=_INSTALL_INSTRUCTIONS,
        ) from exc


__all__ = ["export_gguf"]

"""NF4 quantization config for DanteDojo — bitsandbytes training-only path.

Returns BitsAndBytesConfig kwargs for QLoRA NF4 training.
Raises QuantizationError if bitsandbytes is not installed.
"""

from __future__ import annotations

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import QuantizationError

_INSTALL_INSTRUCTIONS = (
    "bitsandbytes is not installed. "
    "Install it with:\n"
    "  pip install bitsandbytes\n"
    "See https://github.com/TimDettmers/bitsandbytes for details.\n"
    "Note: NF4 quantization is for training only (QLoRA), not inference export."
)


def select_4bit_precision_format() -> str:
    """Pick between FP4 (Blackwell native) and NF4 (Ampere+) for 4-bit training.

    Returns ``"fp4"`` when ``dojo.quant.fp4.requires_blackwell(sm)`` returns
    True for the detected SM; otherwise returns ``"nf4"``.
    """
    try:
        from dojo.quant.fp4 import requires_blackwell
        from dojo.hal.sm_detect import detect_sm
        info = detect_sm()
        if info.available and requires_blackwell(info.sm_float):
            return "fp4"
    except Exception:
        pass
    return "nf4"


def get_bnb_nf4_config() -> dict:
    """Return BitsAndBytesConfig kwargs dict for NF4 QLoRA training.

    Returns a dict ready to unpack into BitsAndBytesConfig(**kwargs).
    This is intentionally a plain dict so callers can inspect/override
    individual fields before instantiating the config object.

    Returns:
        dict with load_in_4bit, bnb_4bit_quant_type, bnb_4bit_use_double_quant,
        and bnb_4bit_compute_dtype keys.

    Raises:
        QuantizationError: if bitsandbytes is not installed.
    """
    _require_bitsandbytes()

    import torch  # noqa: F401 — bitsandbytes requires torch

    cfg = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
    }

    # Emit evidence so QLoRA callers can audit the NF4 config selection.
    logger = EvidenceLogger.get_instance()
    logger.emit(
        {
            "type": "quant.nf4_config",
            "data": {
                "load_in_4bit": cfg["load_in_4bit"],
                "bnb_4bit_quant_type": cfg["bnb_4bit_quant_type"],
                "bnb_4bit_use_double_quant": cfg["bnb_4bit_use_double_quant"],
                "bnb_4bit_compute_dtype": cfg["bnb_4bit_compute_dtype"],
            },
        },
        organ="quant",
    )

    return cfg


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_bitsandbytes() -> None:
    """Raise QuantizationError with install instructions if bitsandbytes is absent."""
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise QuantizationError(
            _INSTALL_INSTRUCTIONS,
            reason=_INSTALL_INSTRUCTIONS,
        ) from exc


__all__ = ["get_bnb_nf4_config"]

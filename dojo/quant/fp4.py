"""FP4 (E2M1) quantization for DanteDojo.

Implements the 4-bit floating-point format used by NVIDIA Blackwell (SM >= 10.0):
  - 1 sign bit, 2 exponent bits, 1 mantissa bit
  - 16 representable values: ±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}

Works on CPU without Blackwell hardware — the math is real, native Blackwell
tensor cores merely accelerate the same arithmetic.

Evidence is emitted for every quantization/dequantization call so the
constitutional audit chain tracks precision conversions.
"""

from __future__ import annotations

from typing import Optional

import torch

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import PrecisionUnsupported


# ---------------------------------------------------------------------------
# FP4 E2M1 constants
# ---------------------------------------------------------------------------

# All positive FP4 values in ascending order (sign=0 case).
# Encoding: bit 3 = sign, bits 2:1 = exponent, bit 0 = mantissa
# exp=00, man=0 → 0.0   (zero)
# exp=00, man=1 → 0.5   (denormal: 0.5 * man)
# exp=01, man=0 → 1.0   (1.0 * 2^0)
# exp=01, man=1 → 1.5   (1.5 * 2^0)
# exp=10, man=0 → 2.0   (1.0 * 2^1)
# exp=10, man=1 → 3.0   (1.5 * 2^1)
# exp=11, man=0 → 4.0   (1.0 * 2^2)
# exp=11, man=1 → 6.0   (1.5 * 2^2)
_FP4_POSITIVE_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)

# Full grid including negatives — indices 0-7 are positive, 8-15 are negative
_FP4_GRID = torch.cat([_FP4_POSITIVE_VALUES, -_FP4_POSITIVE_VALUES])

_FP4_MIN_SM = 10.0  # Blackwell hardware requirement for native FP4 ops


# ---------------------------------------------------------------------------
# Core quantization functions
# ---------------------------------------------------------------------------

def quantize_tensor_fp4(
    tensor: torch.Tensor,
    scale: Optional[float] = None,
) -> tuple[torch.Tensor, float]:
    """Quantize a float tensor to the nearest FP4 E2M1 grid point.

    Finds the closest value in the 16-element FP4 grid for each element.
    An optional scale factor is applied before quantization to utilise the
    full dynamic range of the FP4 format.

    Args:
        tensor: Input float tensor of any shape.
        scale:  Scale factor applied before quantization.  If None, computed
                automatically as ``max(|tensor|) / 6.0`` (maps the tensor's
                maximum absolute value to the FP4 maximum of 6.0).
                A scale of 1.0 uses the raw values as-is.

    Returns:
        Tuple of:
        - ``quantized``: Tensor of same shape containing the nearest FP4 value
          (still in float32, not packed bit-representation).
        - ``scale``:     The scale factor that was used (pass to
                         ``dequantize_tensor_fp4`` to invert).

    Raises:
        PrecisionUnsupported: if tensor is not a floating-point dtype.
    """
    if not tensor.is_floating_point():
        raise PrecisionUnsupported(
            "FP4 quantization requires a floating-point input tensor.",
            reason=f"Got dtype={tensor.dtype}",
        )

    if scale is None:
        max_abs = float(tensor.abs().max())
        scale = max_abs / 6.0 if max_abs > 0.0 else 1.0

    scaled = tensor.float() / scale

    # Nearest-neighbour lookup against the 16-element FP4 grid
    grid = _FP4_GRID.to(tensor.device)
    flat = scaled.reshape(-1)
    dists = (flat.unsqueeze(1) - grid.unsqueeze(0)).abs()
    indices = dists.argmin(dim=1)
    quantized = grid[indices].reshape(tensor.shape)

    EvidenceLogger.get_instance().emit(
        {
            "type": EvidenceType.QUANTIZATION_COMPLETE.value,
            "data": {
                "operation": "fp4_quantize",
                "dtype_in": str(tensor.dtype),
                "shape": list(tensor.shape),
                "scale": scale,
                "unique_values": int(indices.unique().numel()),
            },
        },
        organ="quant",
    )

    return quantized, scale


def dequantize_tensor_fp4(
    quantized: torch.Tensor,
    scale: float,
    target_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize an FP4-quantized tensor by reverting the scale factor.

    Args:
        quantized:    Tensor produced by ``quantize_tensor_fp4`` (float32,
                      values constrained to the FP4 grid).
        scale:        The scale factor returned by ``quantize_tensor_fp4``.
        target_dtype: Output dtype (default float32).

    Returns:
        Dequantized tensor of the same shape in ``target_dtype``.
    """
    result = (quantized * scale).to(target_dtype)

    EvidenceLogger.get_instance().emit(
        {
            "type": EvidenceType.QUANTIZATION_COMPLETE.value,
            "data": {
                "operation": "fp4_dequantize",
                "scale": scale,
                "dtype_out": str(target_dtype),
                "shape": list(result.shape),
            },
        },
        organ="quant",
    )

    return result


def compute_fp4_quantization_error(tensor: torch.Tensor) -> dict:
    """Measure the quantization error introduced by FP4 E2M1 rounding.

    Quantizes and immediately dequantizes the tensor, then computes:
    - ``max_abs_error``:  max |original - reconstructed|
    - ``rmse``:           root mean squared error
    - ``snr_db``:         signal-to-noise ratio in dB (signal = tensor energy)
    - ``scale``:          the auto-computed scale factor used

    Args:
        tensor: Input float tensor.

    Returns:
        Dict with keys: max_abs_error, rmse, snr_db, scale.
    """
    q, scale = quantize_tensor_fp4(tensor)
    recon = dequantize_tensor_fp4(q, scale, target_dtype=tensor.dtype)
    error = tensor.float() - recon.float()
    max_err = float(error.abs().max())
    rmse = float(error.pow(2).mean().sqrt())
    signal_power = float(tensor.float().pow(2).mean())
    snr_db = 10.0 * (signal_power / (rmse ** 2 + 1e-12)) if rmse > 0 else float("inf")

    return {
        "max_abs_error": max_err,
        "rmse": rmse,
        "snr_db": snr_db,
        "scale": scale,
    }


def fp4_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Emulate FP4 matrix multiplication via quantize-matmul-dequantize.

    Quantizes both operands to FP4, performs the matrix multiplication in
    float32, then returns the result along with quantization metadata.

    On Blackwell hardware this would map to native FP4 tensor core operations.
    On CPU/older GPU it runs the same arithmetic in software.

    Args:
        a: Left operand (2D tensor: M x K).
        b: Right operand (2D tensor: K x N).

    Returns:
        Tuple of:
        - Result tensor (M x N, float32).
        - Metadata dict with keys: scale_a, scale_b, a_shape, b_shape.
    """
    qa, scale_a = quantize_tensor_fp4(a)
    qb, scale_b = quantize_tensor_fp4(b)
    result = torch.matmul(qa, qb)
    return result, {"scale_a": scale_a, "scale_b": scale_b,
                    "a_shape": list(a.shape), "b_shape": list(b.shape)}


def requires_blackwell(sm_version: float) -> bool:
    """Return True if the given SM version supports native FP4 tensor cores."""
    return sm_version >= _FP4_MIN_SM


__all__ = [
    "quantize_tensor_fp4",
    "dequantize_tensor_fp4",
    "compute_fp4_quantization_error",
    "fp4_matmul",
    "requires_blackwell",
    "_FP4_GRID",
    "_FP4_POSITIVE_VALUES",
]

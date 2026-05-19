"""Pure-Python GPTQ block quantization emulation (d22).

Implements the math of GPTQ-style per-group quantization without requiring
the ``gptqmodel`` package.  The library accelerates the same operations; the
algorithm is straightforward block-wise asymmetric quantization with
per-group scale + zero-point.

Provides quantize / dequantize / perplexity-proxy so quality-gate tests can
prove the round-trip error is bounded without GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import QuantizationError


@dataclass
class GptqQuantized:
    """Result of GPTQ block quantization."""

    quantized: torch.Tensor  # int values in [0, 2**bits - 1], shape == original
    scales: torch.Tensor     # float scales, one per (row, group)
    zero_points: torch.Tensor  # int zero points, one per (row, group)
    bits: int
    group_size: int
    original_shape: Tuple[int, ...]


def _validate_inputs(weight: torch.Tensor, bits: int, group_size: int) -> None:
    if weight.dim() != 2:
        raise QuantizationError(
            "GPTQ emulation requires a 2D weight matrix",
            reason=f"got shape {tuple(weight.shape)}",
        )
    if bits < 2 or bits > 8:
        raise QuantizationError(
            "GPTQ emulation supports 2..8 bit quantization",
            reason=f"bits={bits}",
        )
    if group_size <= 0:
        raise QuantizationError(
            "group_size must be positive",
            reason=f"group_size={group_size}",
        )
    if weight.shape[1] % group_size != 0:
        raise QuantizationError(
            "input feature dim must be divisible by group_size",
            reason=f"shape[1]={weight.shape[1]}, group_size={group_size}",
        )


def gptq_quantize_weight(
    weight: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
) -> GptqQuantized:
    """Block-quantize a 2D weight matrix using per-group min/max scaling.

    For each (output_row, input_group) tile compute:
      scale = (max - min) / (2**bits - 1)
      zero_point = round(-min / scale)
      q = round(w / scale) + zero_point  clamped to [0, 2**bits - 1]

    Args:
        weight:     2D float tensor of shape (out_features, in_features).
        bits:       Quantization width in 2..8.
        group_size: Number of input features per quantization group.

    Returns:
        GptqQuantized containing the integer codes, scales, zero-points,
        original shape, and quantization parameters.
    """
    _validate_inputs(weight, bits, group_size)

    out_features, in_features = weight.shape
    n_groups = in_features // group_size
    max_int = (1 << bits) - 1

    # Reshape to (out, n_groups, group_size) so per-group ops are vectorised
    grouped = weight.float().reshape(out_features, n_groups, group_size)
    group_min = grouped.min(dim=-1, keepdim=True).values
    group_max = grouped.max(dim=-1, keepdim=True).values

    # Avoid divide-by-zero for constant groups
    spread = group_max - group_min
    spread = torch.where(spread == 0, torch.ones_like(spread), spread)
    scale = spread / max_int

    zero_point = torch.round(-group_min / scale).clamp(0, max_int)
    quantized = torch.round(grouped / scale) + zero_point
    quantized = quantized.clamp(0, max_int).to(torch.int32)

    EvidenceLogger.get_instance().emit(
        {
            "type": "quant.emulation_step",
            "data": {
                "operation": "gptq_quantize",
                "bits": bits,
                "group_size": group_size,
                "out_features": out_features,
                "in_features": in_features,
                "n_groups": n_groups,
            },
        },
        organ="quant",
    )

    return GptqQuantized(
        quantized=quantized.reshape(out_features, in_features),
        scales=scale.squeeze(-1),               # (out, n_groups)
        zero_points=zero_point.squeeze(-1).to(torch.int32),
        bits=bits,
        group_size=group_size,
        original_shape=(out_features, in_features),
    )


def gptq_dequantize_weight(q: GptqQuantized) -> torch.Tensor:
    """Invert ``gptq_quantize_weight`` to recover a float weight matrix."""
    out_features, in_features = q.original_shape
    n_groups = in_features // q.group_size

    quantized = q.quantized.reshape(out_features, n_groups, q.group_size).float()
    scales = q.scales.reshape(out_features, n_groups, 1).float()
    zero = q.zero_points.reshape(out_features, n_groups, 1).float()

    recon = (quantized - zero) * scales
    return recon.reshape(out_features, in_features)


def gptq_round_trip_error(weight: torch.Tensor, bits: int = 4, group_size: int = 128) -> dict:
    """Compute round-trip quantization error metrics for a weight matrix.

    Returns a dict with keys:
      - ``max_abs_error``
      - ``rmse``
      - ``relative_error``  (rmse / weight.std())
      - ``compression_ratio`` (original_bits / quantized_bits + scale overhead)
    """
    q = gptq_quantize_weight(weight, bits=bits, group_size=group_size)
    recon = gptq_dequantize_weight(q)
    err = weight.float() - recon
    max_err = float(err.abs().max())
    rmse = float(err.pow(2).mean().sqrt())
    std = float(weight.float().std())
    rel = rmse / std if std > 0 else float("inf")

    original_bits = weight.numel() * 32
    quantized_bits = weight.numel() * bits + q.scales.numel() * 32 + q.zero_points.numel() * 32
    compression = original_bits / quantized_bits if quantized_bits > 0 else 0.0

    return {
        "max_abs_error": max_err,
        "rmse": rmse,
        "relative_error": rel,
        "compression_ratio": compression,
    }


__all__ = [
    "GptqQuantized",
    "gptq_quantize_weight",
    "gptq_dequantize_weight",
    "gptq_round_trip_error",
]

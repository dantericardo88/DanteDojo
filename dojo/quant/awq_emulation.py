"""Pure-Python AWQ activation-aware quantization emulation (d23).

AWQ (Activation-aware Weight Quantization) scales each weight column by a
per-channel factor derived from activation statistics, so high-activation
columns lose less precision after low-bit quantization.

This module implements the core AWQ math in pure PyTorch with no autoawq
dependency.  Quality-gate tests can prove the round-trip preserves the
activation-weighted product even under 4-bit quantization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import QuantizationError


@dataclass
class AwqQuantized:
    """Result of AWQ quantization."""

    quantized: torch.Tensor       # int values in [0, 2**bits - 1]
    scales: torch.Tensor          # per-(row, group) float scale
    zero_points: torch.Tensor     # per-(row, group) int zero point
    activation_scale: torch.Tensor  # per-input-channel float
    bits: int
    group_size: int
    original_shape: Tuple[int, ...]


def _validate(weight: torch.Tensor, bits: int, group_size: int) -> None:
    if weight.dim() != 2:
        raise QuantizationError(
            "AWQ emulation requires a 2D weight matrix",
            reason=f"got shape {tuple(weight.shape)}",
        )
    if bits < 2 or bits > 8:
        raise QuantizationError(
            "AWQ emulation supports 2..8 bit quantization",
            reason=f"bits={bits}",
        )
    if weight.shape[1] % group_size != 0:
        raise QuantizationError(
            "input feature dim must be divisible by group_size",
            reason=f"shape[1]={weight.shape[1]}, group_size={group_size}",
        )


def compute_activation_scale(
    weight: torch.Tensor,
    sample_activations: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Compute per-input-channel scale from activation magnitudes.

    AWQ uses ``s_i = mean(|x_i|) ** alpha`` to weight columns by activation
    importance.  ``alpha`` is typically 0.5 and trades off activation
    sensitivity vs weight equalisation.

    Args:
        weight:             (out_features, in_features) weight matrix.
        sample_activations: (n_samples, in_features) activation calibration tensor.
        alpha:              Exponent in (0, 1]. Default 0.5.

    Returns:
        (in_features,) tensor of positive scales.
    """
    if sample_activations.dim() != 2:
        raise QuantizationError(
            "sample_activations must be 2D (n_samples, in_features)",
            reason=f"got shape {tuple(sample_activations.shape)}",
        )
    if sample_activations.shape[1] != weight.shape[1]:
        raise QuantizationError(
            "activation in_features must match weight in_features",
            reason=f"{sample_activations.shape[1]} vs {weight.shape[1]}",
        )

    mean_abs = sample_activations.float().abs().mean(dim=0)
    # Avoid zeros so dividing by scale below is safe
    mean_abs = mean_abs.clamp_min(1e-6)
    return mean_abs.pow(alpha)


def awq_quantize_weight(
    weight: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    activation_scale: Optional[torch.Tensor] = None,
) -> AwqQuantized:
    """Block-quantize a weight matrix with optional activation-aware scaling.

    If ``activation_scale`` is provided, columns are pre-scaled by 1/s_i so
    high-activation columns get more precision; the scale is preserved in
    the result so dequantize can invert it.
    """
    _validate(weight, bits, group_size)

    out_features, in_features = weight.shape

    if activation_scale is None:
        activation_scale = torch.ones(in_features, dtype=weight.dtype, device=weight.device)

    if activation_scale.shape != (in_features,):
        raise QuantizationError(
            "activation_scale must have shape (in_features,)",
            reason=f"got {tuple(activation_scale.shape)}",
        )

    # Scale columns so high-activation channels have higher magnitudes
    scaled = weight.float() * activation_scale.unsqueeze(0).float()

    # Now do block quantization on scaled weights
    n_groups = in_features // group_size
    max_int = (1 << bits) - 1
    grouped = scaled.reshape(out_features, n_groups, group_size)
    g_min = grouped.min(dim=-1, keepdim=True).values
    g_max = grouped.max(dim=-1, keepdim=True).values
    spread = g_max - g_min
    spread = torch.where(spread == 0, torch.ones_like(spread), spread)
    scale = spread / max_int
    zero_point = torch.round(-g_min / scale).clamp(0, max_int)
    quantized = torch.round(grouped / scale) + zero_point
    quantized = quantized.clamp(0, max_int).to(torch.int32)

    EvidenceLogger.get_instance().emit(
        {
            "type": "quant.emulation_step",
            "data": {
                "operation": "awq_quantize",
                "bits": bits,
                "group_size": group_size,
                "out_features": out_features,
                "in_features": in_features,
                "activation_aware": activation_scale is not None,
            },
        },
        organ="quant",
    )

    return AwqQuantized(
        quantized=quantized.reshape(out_features, in_features),
        scales=scale.squeeze(-1),
        zero_points=zero_point.squeeze(-1).to(torch.int32),
        activation_scale=activation_scale.float(),
        bits=bits,
        group_size=group_size,
        original_shape=(out_features, in_features),
    )


def awq_dequantize_weight(q: AwqQuantized) -> torch.Tensor:
    """Invert ``awq_quantize_weight`` to recover the original weight matrix."""
    out_features, in_features = q.original_shape
    n_groups = in_features // q.group_size

    quantized = q.quantized.reshape(out_features, n_groups, q.group_size).float()
    scales = q.scales.reshape(out_features, n_groups, 1).float()
    zero = q.zero_points.reshape(out_features, n_groups, 1).float()

    scaled_recon = (quantized - zero) * scales
    recon = scaled_recon.reshape(out_features, in_features)

    # Invert the activation-aware column scaling
    return recon / q.activation_scale.unsqueeze(0)


def awq_round_trip_error(
    weight: torch.Tensor,
    sample_activations: Optional[torch.Tensor] = None,
    bits: int = 4,
    group_size: int = 128,
) -> dict:
    """Compute round-trip quantization error metrics for a weight matrix.

    When ``sample_activations`` is provided, the test measures both the
    raw weight error and the activation-weighted error (closer to what
    matters for inference quality).
    """
    if sample_activations is not None:
        a_scale = compute_activation_scale(weight, sample_activations)
    else:
        a_scale = None

    q = awq_quantize_weight(weight, bits=bits, group_size=group_size, activation_scale=a_scale)
    recon = awq_dequantize_weight(q)
    err = weight.float() - recon
    max_err = float(err.abs().max())
    rmse = float(err.pow(2).mean().sqrt())

    out: dict = {
        "max_abs_error": max_err,
        "rmse": rmse,
    }

    if sample_activations is not None:
        # Activation-weighted reconstruction error (Wx vs W_qx)
        x = sample_activations.float()
        true_out = x @ weight.float().t()
        approx_out = x @ recon.t()
        act_err = (true_out - approx_out).abs()
        out["activation_weighted_max_error"] = float(act_err.max())
        out["activation_weighted_rmse"] = float(act_err.pow(2).mean().sqrt())

    return out


__all__ = [
    "AwqQuantized",
    "compute_activation_scale",
    "awq_quantize_weight",
    "awq_dequantize_weight",
    "awq_round_trip_error",
]

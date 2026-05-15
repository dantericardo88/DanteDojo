"""DanteDojo kernel dispatch — constitutional one-door wrappers.

All external consumers of custom kernels MUST go through these one-door
dispatch functions.  They select the best available backend via
``KernelRegistry`` and delegate to the appropriate kernel implementation.

Constitutional guarantees:
- Both dispatch functions are decorated with ``@one_door``.
- Kernel selection is driven by ``KernelRegistry.probe()`` so selection
  evidence is always emitted before the computation.
- No direct access to Triton or PyTorch internals is permitted outside
  ``attention.py`` and ``rms_norm.py``.
"""

from __future__ import annotations

from typing import Optional

import torch

from dojo.core.one_door import one_door
from dojo.kernels.attention import flash_attention_forward
from dojo.kernels.registry import KernelRegistry
from dojo.kernels.rms_norm import fused_rms_norm


# ---------------------------------------------------------------------------
# Attention dispatch
# ---------------------------------------------------------------------------

@one_door("kernels.dispatch_attention", organ="kernels")
def dispatch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Constitutional one-door for attention kernel dispatch.

    Selects Triton FlashAttention-2 or PyTorch SDPA based on hardware probe.
    Evidence of the selection is emitted by the underlying implementation.

    Args:
        q: Query tensor ``(B, H, M, D)``.
        k: Key tensor ``(B, H, N, D)``.
        v: Value tensor ``(B, H, N, D)``.
        causal: Apply causal lower-triangular mask.
        sm_scale: Softmax scale (default ``1/sqrt(D)``).

    Returns:
        Attention output ``(B, H, M, D)``.
    """
    # Probe ensures selection evidence is emitted; the flash_attention_forward
    # call re-emits its own per-call evidence as well.
    registry = KernelRegistry.get_instance()
    registry.probe()
    return flash_attention_forward(q, k, v, causal=causal, sm_scale=sm_scale)


# ---------------------------------------------------------------------------
# RMS-norm dispatch
# ---------------------------------------------------------------------------

@one_door("kernels.dispatch_rms_norm", organ="kernels")
def dispatch_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Constitutional one-door for RMS norm kernel dispatch.

    Selects Triton fused RMS-norm or PyTorch reference based on hardware probe.

    Args:
        x:      Input tensor — any shape; normalises over the last dim.
        weight: Scale (gamma) parameter of shape ``(H,)`` where ``H = x.shape[-1]``.
        eps:    Numerical stability epsilon.

    Returns:
        Normalised tensor of the same shape as ``x``.
    """
    registry = KernelRegistry.get_instance()
    registry.probe()
    return fused_rms_norm(x, weight, eps=eps)


__all__ = [
    "dispatch_attention",
    "dispatch_rms_norm",
]

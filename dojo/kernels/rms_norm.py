"""DanteDojo custom RMS-norm kernels — constitutional fused RMS normalisation.

When ``triton`` is importable AND ``DOJO_REAL_GPU=1``, a real Triton kernel
fuses the normalisation, weight multiplication, and (optionally) gradient
computation into a single GPU pass.  In CI mode the mathematically identical
PyTorch reference implementation runs on CPU tensors.

Constitutional guarantees:
- ``fused_rms_norm`` and ``fused_rms_norm_backward`` are the only public
  entry points.
- Every call emits ``TRAINING_BACKEND_SELECTED`` evidence into the Merkle chain.
- ``triton`` is imported only inside a try/except — never at module level.

RMS-norm formula (reference):
    y_i = x_i / sqrt( mean(x²) + eps ) * weight_i

The fused backward kernel computes ``dx`` and ``dweight`` in a single pass,
avoiding the re-computation of the normalised activations.
"""

from __future__ import annotations

import os

import torch

from dojo.core.evidence import EvidenceLogger, EvidenceType


# ---------------------------------------------------------------------------
# Availability helpers (shared with attention.py)
# ---------------------------------------------------------------------------

def _is_triton_available() -> bool:
    try:
        import importlib
        importlib.import_module("triton")
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _real_gpu_requested() -> bool:
    return os.environ.get("DOJO_REAL_GPU", "0") == "1"


# ---------------------------------------------------------------------------
# Triton kernel: fused RMS-norm forward
# ---------------------------------------------------------------------------

def _get_triton_rms_norm_kernel():  # noqa: ANN201
    """Return the compiled Triton RMS-norm forward kernel.

    The kernel loads a full row of ``x``, computes the RMS, normalises, and
    multiplies by ``weight`` — all in a single warp-synchronous pass with no
    intermediate global memory traffic.

    Grid: one program per row (token) — shape ``(B * S,)``.
    Each program handles a row of length ``H`` (hidden dim).
    """
    import triton  # noqa: PLC0415
    import triton.language as tl  # noqa: PLC0415

    @triton.jit
    def _rms_norm_fwd_kernel(
        X,          # input  (N_ROWS, H)
        W,          # weight (H,)
        Y,          # output (N_ROWS, H)
        stride_x,   # row stride of X in elements
        stride_y,   # row stride of Y in elements
        H: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused RMS-norm: Y[row] = X[row] / rms(X[row]) * W."""
        row = tl.program_id(0)

        # ---- Load one full row of X ----
        offs = tl.arange(0, BLOCK_SIZE)
        x_ptrs = X + row * stride_x + offs
        x = tl.load(x_ptrs, mask=offs < H, other=0.0).to(tl.float32)

        # ---- Compute RMS ----
        # mean(x²) + eps  →  rms = sqrt(mean_sq + eps)
        mean_sq = tl.sum(x * x, axis=0) / H
        rms_inv = tl.rsqrt(mean_sq + eps)

        # ---- Load weight and normalise ----
        w = tl.load(W + offs, mask=offs < H, other=1.0).to(tl.float32)
        y = x * rms_inv * w

        # ---- Store output ----
        y_ptrs = Y + row * stride_y + offs
        tl.store(y_ptrs, y, mask=offs < H)

    return _rms_norm_fwd_kernel


def _get_triton_rms_norm_bwd_kernel():  # noqa: ANN201
    """Return the compiled Triton RMS-norm backward kernel.

    The backward pass re-reads x and w to compute:
        dx_i = dy_i * w_i / rms  -  x_i * (sum_j dy_j w_j x_j) / (H * rms^3)
        dw_i = sum_rows  dy_i * x_i / rms

    This fused kernel avoids storing the normalised activations from the
    forward pass, trading a small amount of recomputation for memory savings
    — the standard RMS-norm backward fusion pattern.
    """
    import triton  # noqa: PLC0415
    import triton.language as tl  # noqa: PLC0415

    @triton.jit
    def _rms_norm_bwd_kernel(
        X,           # (N_ROWS, H) — saved from forward
        W,           # (H,)
        DY,          # upstream gradient (N_ROWS, H)
        DX,          # output: grad w.r.t. X (N_ROWS, H)
        DW_partial,  # output: partial grad w.r.t. W (N_ROWS, H) — sum later
        stride_x,
        stride_dy,
        stride_dx,
        stride_dw,
        H: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused RMS-norm backward for one row."""
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)

        x  = tl.load(X  + row * stride_x  + offs, mask=offs < H, other=0.0).to(tl.float32)
        dy = tl.load(DY + row * stride_dy + offs, mask=offs < H, other=0.0).to(tl.float32)
        w  = tl.load(W  + offs,                  mask=offs < H, other=1.0).to(tl.float32)

        # Recompute rms_inv from saved x
        mean_sq = tl.sum(x * x, axis=0) / H
        rms_inv = tl.rsqrt(mean_sq + eps)

        # x_hat = x * rms_inv
        x_hat = x * rms_inv

        # Gradient of the norm: dot(dy * w, x_hat) / H
        dot = tl.sum(dy * w * x_hat, axis=0) / H

        # dx = (dy * w - x_hat * dot) * rms_inv
        dx = (dy * w - x_hat * dot) * rms_inv

        # Partial dw = dy * x_hat  (sum over rows in the host later)
        dw_row = dy * x_hat

        tl.store(DX          + row * stride_dx + offs, dx,     mask=offs < H)
        tl.store(DW_partial  + row * stride_dw + offs, dw_row, mask=offs < H)

    return _rms_norm_bwd_kernel


# ---------------------------------------------------------------------------
# Triton dispatch helpers
# ---------------------------------------------------------------------------

def _triton_fused_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Run the Triton fused RMS-norm forward on a 2-D input."""
    import triton  # noqa: PLC0415

    assert x.ndim == 2, "Expected 2-D input (N, H)"
    N, H = x.shape
    x = x.contiguous().cuda()
    weight = weight.contiguous().cuda()
    y = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(H)
    kernel = _get_triton_rms_norm_kernel()
    kernel[(N,)](
        x, weight, y,
        x.stride(0), y.stride(0),
        H=H,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


def _triton_fused_rms_norm_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    dy: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the Triton fused RMS-norm backward; returns (dx, dw)."""
    import triton  # noqa: PLC0415

    assert x.ndim == 2, "Expected 2-D input (N, H)"
    N, H = x.shape
    x      = x.contiguous().cuda()
    weight = weight.contiguous().cuda()
    dy     = dy.contiguous().cuda()
    dx     = torch.empty_like(x)
    dw_partial = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(H)
    kernel = _get_triton_rms_norm_bwd_kernel()
    kernel[(N,)](
        x, weight, dy, dx, dw_partial,
        x.stride(0), dy.stride(0), dx.stride(0), dw_partial.stride(0),
        H=H,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    dw = dw_partial.sum(dim=0)
    return dx, dw


# ---------------------------------------------------------------------------
# PyTorch reference fallback (CPU-safe, mathematically correct)
# ---------------------------------------------------------------------------

def _pytorch_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference RMS-norm: x / sqrt(mean(x²) + eps) * weight."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).float()
    mean_sq = (x_2d * x_2d).mean(dim=-1, keepdim=True)
    rms_inv = torch.rsqrt(mean_sq + eps)
    y = x_2d * rms_inv * weight.float()
    return y.to(x.dtype).reshape(orig_shape)


def _pytorch_rms_norm_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    dy: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference backward pass for RMS-norm — returns (dx, dw)."""
    orig_shape = x.shape
    x_2d  = x.reshape(-1, x.shape[-1]).float()
    dy_2d = dy.reshape(-1, dy.shape[-1]).float()
    w     = weight.float()
    H     = x_2d.shape[-1]

    mean_sq = (x_2d * x_2d).mean(dim=-1, keepdim=True)
    rms_inv = torch.rsqrt(mean_sq + eps)
    x_hat   = x_2d * rms_inv           # normalised activations

    dot = (dy_2d * w * x_hat).sum(dim=-1, keepdim=True) / H
    dx  = (dy_2d * w - x_hat * dot) * rms_inv
    dw  = (dy_2d * x_hat).sum(dim=0)

    return dx.to(x.dtype).reshape(orig_shape), dw.to(weight.dtype)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def fused_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Constitutional one-door for fused RMS normalisation (forward pass).

    Dispatches to the Triton kernel when available + DOJO_REAL_GPU=1, else
    uses a PyTorch reference implementation.

    Args:
        x:      Input tensor — any shape; normalisation is over the last dim.
        weight: Scale (gamma) parameter of shape ``(H,)`` where ``H = x.shape[-1]``.
        eps:    Numerical stability epsilon (default 1e-6).

    Returns:
        Normalised tensor of the same shape as ``x``.
    """
    logger = EvidenceLogger.get_instance()
    use_triton = _is_triton_available() and _real_gpu_requested()
    kernel_name = "triton_rms_norm" if use_triton else "pytorch_fallback"

    logger.emit(
        {
            "type": EvidenceType.TRAINING_BACKEND_SELECTED.value,
            "data": {
                "kernel": kernel_name,
                "subsystem": "rms_norm",
                "triton_available": _is_triton_available(),
                "real_gpu": _real_gpu_requested(),
                "x_shape": list(x.shape),
                "dtype": str(x.dtype),
                "eps": eps,
            },
        },
        organ="kernels",
    )

    if use_triton:
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        y_2d = _triton_fused_rms_norm(x_2d, weight, eps=eps)
        return y_2d.reshape(orig_shape)

    return _pytorch_rms_norm(x, weight, eps=eps)


def fused_rms_norm_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    dy: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constitutional one-door for fused RMS normalisation (backward pass).

    Args:
        x:      Input from the forward pass.
        weight: Scale parameter from the forward pass.
        dy:     Upstream gradient (same shape as ``x``).
        eps:    Must match the value used in the forward pass.

    Returns:
        ``(dx, dw)`` — gradients with respect to ``x`` and ``weight``.
    """
    logger = EvidenceLogger.get_instance()
    use_triton = _is_triton_available() and _real_gpu_requested()
    kernel_name = "triton_rms_norm" if use_triton else "pytorch_fallback"

    logger.emit(
        {
            "type": EvidenceType.TRAINING_BACKEND_SELECTED.value,
            "data": {
                "kernel": kernel_name,
                "subsystem": "rms_norm_backward",
                "triton_available": _is_triton_available(),
                "real_gpu": _real_gpu_requested(),
                "x_shape": list(x.shape),
                "dtype": str(x.dtype),
                "eps": eps,
            },
        },
        organ="kernels",
    )

    if use_triton:
        orig_shape = x.shape
        x_2d  = x.reshape(-1, x.shape[-1])
        dy_2d = dy.reshape(-1, dy.shape[-1])
        dx_2d, dw = _triton_fused_rms_norm_backward(x_2d, weight, dy_2d, eps=eps)
        return dx_2d.reshape(orig_shape), dw

    return _pytorch_rms_norm_backward(x, weight, dy, eps=eps)


__all__ = [
    "fused_rms_norm",
    "fused_rms_norm_backward",
]

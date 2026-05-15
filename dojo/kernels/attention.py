"""DanteDojo custom attention kernels — constitutional FlashAttention-2 via Triton.

When ``triton`` is importable AND ``DOJO_REAL_GPU=1``, a real Triton
FlashAttention-2 tiled kernel executes on the GPU.  In CI mode (no GPU /
no Triton), a mathematically identical PyTorch reference path runs so the
full evidence chain and correctness guarantees are exercised without hardware.

Constitutional guarantees:
- ``flash_attention_forward`` is the *only* public entry point.
- Every call emits ``TRAINING_BACKEND_SELECTED`` evidence stamped into the
  Merkle chain.
- FP8 on SM < 8.9 raises ``PrecisionUnsupported`` (fail-closed).
- ``triton`` is imported only inside a try/except — never at module level.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn.functional as F

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import PrecisionUnsupported


# ---------------------------------------------------------------------------
# Availability probe
# ---------------------------------------------------------------------------

def is_triton_available() -> bool:
    """Return True if the ``triton`` package can be imported."""
    try:
        import importlib
        importlib.import_module("triton")
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _real_gpu_requested() -> bool:
    return os.environ.get("DOJO_REAL_GPU", "0") == "1"


# ---------------------------------------------------------------------------
# Triton kernel (real FlashAttention-2 tiled implementation)
# ---------------------------------------------------------------------------

def _get_triton_kernel():  # noqa: ANN201
    """Import triton lazily and return the compiled kernel callable.

    The kernel implements FlashAttention-2 (Dao et al., 2023) tiling strategy:
    - Outer loop over query tiles (BLOCK_M rows of Q at a time).
    - Inner loop over key/value tiles (BLOCK_N columns of K/V).
    - Online softmax rescaling so the full attention matrix never materialises.
    - Causal masking applied inside the inner loop when ``causal=True``.

    Strides are passed as runtime scalars so the kernel handles arbitrary
    batch/head/sequence/dim layouts without recompilation.
    """
    import triton  # noqa: PLC0415
    import triton.language as tl  # noqa: PLC0415

    @triton.jit
    def _flash_attn_fwd_kernel(
        Q, K, V, Out,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        sm_scale,
        N_CTX: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_DHEAD: tl.constexpr,
        causal: tl.constexpr,
    ):
        """Triton FlashAttention-2 forward kernel.

        Grid: (ceil(N_CTX / BLOCK_M), batch * n_heads)
        Each program handles one BLOCK_M-row tile of Q for one (batch, head).
        """
        # ---- program-id indexing ----
        start_m = tl.program_id(0)          # which Q-tile (row block index)
        off_bh  = tl.program_id(1)          # flat batch*head index

        # Decode batch and head from flat index (passed via strides)
        # off_bh encodes (b * n_heads + h); we recover via the stride layout.
        # For simplicity we use off_bh directly with stride_qb / stride_qh.
        off_b = off_bh // (stride_qb // (stride_qh + 1) + 1)
        off_h = off_bh % (stride_qb // (stride_qh + 1) + 1)

        # ---- base pointers for this (batch, head) ----
        Q_ptr = Q + off_b * stride_qb + off_h * stride_qh
        K_ptr = K + off_b * stride_kb + off_h * stride_kh
        V_ptr = V + off_b * stride_vb + off_h * stride_vh
        O_ptr = Out + off_b * stride_ob + off_h * stride_oh

        # ---- Q tile: rows [start_m*BLOCK_M, (start_m+1)*BLOCK_M) ----
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_DHEAD)

        # Load Q tile [BLOCK_M, BLOCK_DHEAD]
        q_ptrs = Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

        # ---- Online-softmax accumulators ----
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # running max
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)               # running sum
        acc = tl.zeros([BLOCK_M, BLOCK_DHEAD], dtype=tl.float32)   # output accumulator

        # ---- Inner loop over K/V tiles ----
        n_blocks = tl.cdiv(N_CTX, BLOCK_N)
        for start_n in range(0, n_blocks):
            offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

            # Causal masking: skip blocks entirely beyond the diagonal
            if causal:
                if start_n * BLOCK_N > (start_m + 1) * BLOCK_M:
                    continue  # entire K tile is beyond current Q tile

            # Load K tile [BLOCK_N, BLOCK_DHEAD]
            k_ptrs = K_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

            # Compute QK^T [BLOCK_M, BLOCK_N]
            qk = tl.dot(q, tl.trans(k)) * sm_scale

            # Apply causal mask (within-block fine-grained)
            if causal:
                causal_mask = offs_m[:, None] >= offs_n[None, :]
                qk = tl.where(causal_mask, qk, float("-inf"))

            # Out-of-bounds mask
            qk = tl.where(offs_n[None, :] < N_CTX, qk, float("-inf"))

            # Online softmax update
            m_ij = tl.max(qk, axis=1)                    # [BLOCK_M]
            p    = tl.exp(qk - m_ij[:, None])             # [BLOCK_M, BLOCK_N]
            l_ij = tl.sum(p, axis=1)                      # [BLOCK_M]

            # Rescale previous accumulator
            m_new = tl.maximum(m_i, m_ij)
            alpha  = tl.exp(m_i - m_new)
            beta   = tl.exp(m_ij - m_new)

            acc  = acc * alpha[:, None] + beta[:, None] * tl.dot(
                p.to(tl.float16), tl.load(
                    V_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
                    mask=offs_n[:, None] < N_CTX, other=0.0,
                ).to(tl.float16),
            )
            l_i = alpha * l_i + beta * l_ij
            m_i = m_new

        # ---- Normalise and write output ----
        acc = acc / l_i[:, None]
        o_ptrs = O_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
        tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)

    return _flash_attn_fwd_kernel


def _triton_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Launch the Triton FlashAttention-2 kernel.

    Args:
        q: Query tensor of shape ``(B, H, M, D)``.
        k: Key tensor of shape ``(B, H, N, D)``.
        v: Value tensor of shape ``(B, H, N, D)``.
        causal: Whether to apply a causal (lower-triangular) mask.
        sm_scale: Softmax scale (default: ``1/sqrt(D)``).

    Returns:
        Output tensor of shape ``(B, H, M, D)``.
    """
    import triton  # noqa: PLC0415

    kernel = _get_triton_kernel()

    B, H, M, D = q.shape
    N = k.shape[2]

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # Ensure contiguous fp16 tensors on CUDA
    q = q.contiguous().half().cuda()
    k = k.contiguous().half().cuda()
    v = v.contiguous().half().cuda()
    out = torch.empty_like(q)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DHEAD = triton.next_power_of_2(D)

    grid = (triton.cdiv(M, BLOCK_M), B * H)

    kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        sm_scale,
        N_CTX=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DHEAD=BLOCK_DHEAD,
        causal=causal,
    )
    return out


# ---------------------------------------------------------------------------
# PyTorch reference fallback (mathematically equivalent, CPU-safe)
# ---------------------------------------------------------------------------

def _pytorch_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Pure-PyTorch scaled dot-product attention (mathematically equivalent).

    Uses ``torch.nn.functional.scaled_dot_product_attention`` when available
    (PyTorch >= 2.0) and falls back to a manual softmax computation otherwise.

    Args:
        q: ``(B, H, M, D)``
        k: ``(B, H, N, D)``
        v: ``(B, H, N, D)``
        causal: Lower-triangular causal mask.
        sm_scale: Scale applied before softmax (default ``1/sqrt(D)``).

    Returns:
        ``(B, H, M, D)`` attention output.
    """
    D = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # torch.nn.functional.scaled_dot_product_attention is available in PT >= 2.0
    if hasattr(F, "scaled_dot_product_attention"):
        # SDPA expects (B, H, S, D) which is exactly our layout.
        # It handles the sm_scale internally as 1/sqrt(D); we pre-scale to match.
        # We multiply Q by our custom sm_scale and pass scale=1.0 to avoid
        # double-scaling on older PyTorch builds that don't have the scale param.
        try:
            return F.scaled_dot_product_attention(
                q * sm_scale * math.sqrt(D),  # undo internal 1/sqrt(D), apply ours
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=causal,
                scale=sm_scale,
            )
        except TypeError:
            # scale kwarg added in PT 2.1 — fall through to manual path
            pass

    # Manual implementation: O(S^2) but numerically correct reference.
    # scores: (B, H, M, N)
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    if causal:
        M, N = q.shape[-2], k.shape[-2]
        mask = torch.triu(torch.ones(M, N, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(attn, v)


# ---------------------------------------------------------------------------
# Precision guard
# ---------------------------------------------------------------------------

def _check_fp8_support(q: torch.Tensor) -> None:
    """Raise PrecisionUnsupported if FP8 is requested on SM < 8.9."""
    fp8_dtypes = set()
    try:
        fp8_dtypes.add(torch.float8_e4m3fn)
        fp8_dtypes.add(torch.float8_e5m2)
    except AttributeError:
        pass  # FP8 not in this PyTorch build at all — nothing to check

    if q.dtype not in fp8_dtypes:
        return

    if not torch.cuda.is_available():
        raise PrecisionUnsupported(
            "FP8 attention requires CUDA hardware (SM >= 8.9 / Hopper+)",
            reason="No CUDA device found.",
        )

    capability = torch.cuda.get_device_capability()
    sm = capability[0] * 10 + capability[1]
    if sm < 89:
        raise PrecisionUnsupported(
            f"FP8 attention requires SM >= 8.9; detected SM {sm / 10:.1f}",
            reason=f"Device compute capability {capability[0]}.{capability[1]} is below Hopper.",
        )


# ---------------------------------------------------------------------------
# Public one-door entry point
# ---------------------------------------------------------------------------

def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Constitutional one-door for flash attention.

    Dispatches to the Triton FlashAttention-2 kernel when
    ``triton`` is importable AND ``DOJO_REAL_GPU=1``; otherwise falls back
    to a mathematically equivalent PyTorch implementation.

    Args:
        q: Query tensor ``(B, H, M, D)``.
        k: Key tensor ``(B, H, N, D)``.
        v: Value tensor ``(B, H, N, D)``.
        causal: Apply causal (autoregressive) mask.
        sm_scale: Softmax temperature scale (default ``1/sqrt(D)``).

    Returns:
        Attention output ``(B, H, M, D)``.

    Raises:
        PrecisionUnsupported: If FP8 dtype is passed on SM < 8.9.
    """
    _check_fp8_support(q)

    logger = EvidenceLogger.get_instance()

    use_triton = is_triton_available() and _real_gpu_requested()
    kernel_name = "triton_flash_attn" if use_triton else "pytorch_fallback"

    logger.emit(
        {
            "type": EvidenceType.TRAINING_BACKEND_SELECTED.value,
            "data": {
                "kernel": kernel_name,
                "subsystem": "attention",
                "triton_available": is_triton_available(),
                "real_gpu": _real_gpu_requested(),
                "causal": causal,
                "q_shape": list(q.shape),
                "dtype": str(q.dtype),
            },
        },
        organ="kernels",
    )

    if use_triton:
        return _triton_flash_attention(q, k, v, causal=causal, sm_scale=sm_scale)
    return _pytorch_flash_attention(q, k, v, causal=causal, sm_scale=sm_scale)


__all__ = [
    "flash_attention_forward",
    "is_triton_available",
]

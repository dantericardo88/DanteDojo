"""Tests for DanteDojo custom CUDA/Triton kernels (d15).

All tests run in CI mode (no GPU, no Triton) using small CPU tensors.
The kernel implementations fall back to mathematically correct PyTorch
reference paths so the full evidence chain and correctness guarantees
are exercised without real hardware.

Tensor shape convention throughout: (B=1, H=2, S=8, D=16).
"""

from __future__ import annotations

import math

import pytest
import torch

from dojo.core.evidence import EvidenceLogger, EvidenceType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_qkv(dtype=torch.float32):
    """Return small (1, 2, 8, 16) Q, K, V tensors for testing."""
    B, H, S, D = 1, 2, 8, 16
    q = torch.randn(B, H, S, D, dtype=dtype)
    k = torch.randn(B, H, S, D, dtype=dtype)
    v = torch.randn(B, H, S, D, dtype=dtype)
    return q, k, v


def _small_x(dtype=torch.float32):
    """Return small (4, 16) input tensor and weight for RMS-norm testing."""
    x = torch.randn(4, 16, dtype=dtype)
    w = torch.ones(16, dtype=dtype)
    return x, w


def _last_events(n: int = 5):
    """Return the last n raw evidence events."""
    return EvidenceLogger._events[-n:]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_evidence():
    """Reset evidence logger between tests."""
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture(autouse=True)
def _no_real_gpu(monkeypatch):
    """Ensure DOJO_REAL_GPU is not set so tests stay in CI/PyTorch mode."""
    monkeypatch.delenv("DOJO_REAL_GPU", raising=False)


# ===========================================================================
# 1. Availability and probing tests (5)
# ===========================================================================

class TestKernelRegistryProbe:
    def test_kernel_registry_probe_returns_dict(self):
        """probe() must return a dict."""
        from dojo.kernels.registry import KernelRegistry
        KernelRegistry.clear()
        reg = KernelRegistry()
        result = reg.probe()
        assert isinstance(result, dict)

    def test_kernel_registry_has_triton_key(self):
        """Probe result must include a 'triton' key."""
        from dojo.kernels.registry import KernelRegistry
        KernelRegistry.clear()
        reg = KernelRegistry()
        result = reg.probe()
        assert "triton" in result

    def test_kernel_registry_has_pytorch_fallback(self):
        """Probe result must mark pytorch_flash_attn and pytorch_rms_norm as True."""
        from dojo.kernels.registry import KernelRegistry
        KernelRegistry.clear()
        reg = KernelRegistry()
        result = reg.probe()
        assert result["pytorch_flash_attn"] is True
        assert result["pytorch_rms_norm"] is True

    def test_is_triton_available_returns_bool(self):
        """is_triton_available() must return a plain bool."""
        from dojo.kernels.attention import is_triton_available
        result = is_triton_available()
        assert isinstance(result, bool)

    def test_kernel_registry_emits_evidence(self):
        """probe() must emit at least one TRAINING_BACKEND_SELECTED evidence event."""
        from dojo.kernels.registry import KernelRegistry
        KernelRegistry.clear()
        logger = EvidenceLogger.get_instance()
        reg = KernelRegistry()
        reg.probe()
        events = EvidenceLogger._events
        types = [e["evidence"].get("type") for e in events]
        assert EvidenceType.TRAINING_BACKEND_SELECTED.value in types


# ===========================================================================
# 2. Attention kernel tests (7)
# ===========================================================================

class TestFlashAttention:
    def test_flash_attention_fallback_runs_ci_mode(self):
        """flash_attention_forward must run without GPU in CI mode."""
        from dojo.kernels.attention import flash_attention_forward
        q, k, v = _small_qkv()
        out = flash_attention_forward(q, k, v)
        assert out is not None

    def test_flash_attention_output_shape_matches_input(self):
        """Output shape must equal Q shape."""
        from dojo.kernels.attention import flash_attention_forward
        q, k, v = _small_qkv()
        out = flash_attention_forward(q, k, v)
        assert out.shape == q.shape

    def test_flash_attention_causal_mask_reduces_output(self):
        """Causal and non-causal outputs must differ (the mask changes values)."""
        from dojo.kernels.attention import flash_attention_forward
        torch.manual_seed(42)
        q, k, v = _small_qkv()
        out_full   = flash_attention_forward(q, k, v, causal=False)
        out_causal = flash_attention_forward(q, k, v, causal=True)
        # They should not be identical (causal mask zeros out future tokens)
        assert not torch.allclose(out_full, out_causal, atol=1e-4)

    def test_flash_attention_emits_evidence(self):
        """flash_attention_forward must emit evidence."""
        from dojo.kernels.attention import flash_attention_forward
        q, k, v = _small_qkv()
        flash_attention_forward(q, k, v)
        events = EvidenceLogger._events
        assert len(events) >= 1

    def test_flash_attention_evidence_has_kernel_name(self):
        """Emitted evidence must include a 'kernel' field."""
        from dojo.kernels.attention import flash_attention_forward
        q, k, v = _small_qkv()
        flash_attention_forward(q, k, v)
        events = EvidenceLogger._events
        # Find the TRAINING_BACKEND_SELECTED event
        found = False
        for e in events:
            ev = e.get("evidence", {})
            if ev.get("type") == EvidenceType.TRAINING_BACKEND_SELECTED.value:
                assert "kernel" in ev.get("data", {})
                found = True
                break
        assert found, "No TRAINING_BACKEND_SELECTED event emitted"

    def test_flash_attention_fallback_mathematically_correct(self):
        """CI fallback must match O(N²) reference: softmax(QK^T / sqrt(D)) @ V."""
        from dojo.kernels.attention import flash_attention_forward
        torch.manual_seed(7)
        q, k, v = _small_qkv()
        D = q.shape[-1]
        sm_scale = 1.0 / math.sqrt(D)

        # Reference: manual scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
        attn   = torch.softmax(scores, dim=-1)
        ref    = torch.matmul(attn, v)

        out = flash_attention_forward(q, k, v, sm_scale=sm_scale)
        assert torch.allclose(out.float(), ref.float(), atol=1e-4), (
            f"Max diff: {(out.float() - ref.float()).abs().max().item()}"
        )

    def test_flash_attention_dispatch_via_one_door(self):
        """dispatch_attention must call through and return correct shape."""
        from dojo.kernels.dispatch import dispatch_attention
        from dojo.kernels.registry import KernelRegistry
        KernelRegistry.clear()
        q, k, v = _small_qkv()
        out = dispatch_attention(q, k, v)
        assert out.shape == q.shape


# ===========================================================================
# 3. RMS norm kernel tests (5)
# ===========================================================================

class TestRmsNorm:
    def test_rms_norm_fallback_runs_ci_mode(self):
        """fused_rms_norm must run without GPU in CI mode."""
        from dojo.kernels.rms_norm import fused_rms_norm
        x, w = _small_x()
        out = fused_rms_norm(x, w)
        assert out is not None

    def test_rms_norm_output_shape_correct(self):
        """Output shape must equal input shape."""
        from dojo.kernels.rms_norm import fused_rms_norm
        x, w = _small_x()
        out = fused_rms_norm(x, w)
        assert out.shape == x.shape

    def test_rms_norm_numerically_correct(self):
        """CI fallback must match reference: x / sqrt(mean(x²) + eps) * weight."""
        from dojo.kernels.rms_norm import fused_rms_norm
        torch.manual_seed(99)
        x, w = _small_x()
        eps = 1e-6

        # Reference computation
        x_f   = x.float()
        mean_sq = (x_f * x_f).mean(dim=-1, keepdim=True)
        rms_inv = torch.rsqrt(mean_sq + eps)
        ref   = (x_f * rms_inv * w.float()).to(x.dtype)

        out = fused_rms_norm(x, w, eps=eps)
        assert torch.allclose(out.float(), ref.float(), atol=1e-5), (
            f"Max diff: {(out.float() - ref.float()).abs().max().item()}"
        )

    def test_rms_norm_emits_evidence(self):
        """fused_rms_norm must emit TRAINING_BACKEND_SELECTED evidence."""
        from dojo.kernels.rms_norm import fused_rms_norm
        x, w = _small_x()
        fused_rms_norm(x, w)
        events = EvidenceLogger._events
        types = [e["evidence"].get("type") for e in events]
        assert EvidenceType.TRAINING_BACKEND_SELECTED.value in types

    def test_rms_norm_dispatch_via_one_door(self):
        """dispatch_rms_norm must call through and return correct shape."""
        from dojo.kernels.dispatch import dispatch_rms_norm
        from dojo.kernels.registry import KernelRegistry
        KernelRegistry.clear()
        x, w = _small_x()
        out = dispatch_rms_norm(x, w)
        assert out.shape == x.shape


# ===========================================================================
# 4. Dispatch tests (3)
# ===========================================================================

class TestDispatch:
    def test_dispatch_attention_one_door_wired(self):
        """dispatch_attention must have the __one_door__ attribute."""
        from dojo.kernels.dispatch import dispatch_attention
        assert hasattr(dispatch_attention, "__one_door__")
        assert dispatch_attention.__one_door__ == "kernels.dispatch_attention"

    def test_dispatch_rms_norm_one_door_wired(self):
        """dispatch_rms_norm must have the __one_door__ attribute."""
        from dojo.kernels.dispatch import dispatch_rms_norm
        assert hasattr(dispatch_rms_norm, "__one_door__")
        assert dispatch_rms_norm.__one_door__ == "kernels.dispatch_rms_norm"

    def test_dispatch_selects_pytorch_when_no_triton(self):
        """In CI mode (no DOJO_REAL_GPU), registry must select pytorch backend."""
        from dojo.kernels.registry import KernelRegistry
        KernelRegistry.clear()
        reg = KernelRegistry()
        reg.probe()
        assert reg.get_best_attention() == "pytorch"
        assert reg.get_best_rms_norm()  == "pytorch"


# ===========================================================================
# 5. Additional correctness and edge-case tests (bring total >= 20)
# ===========================================================================

class TestAdditionalCoverage:
    def test_rms_norm_backward_output_shapes(self):
        """fused_rms_norm_backward must return (dx, dw) with correct shapes."""
        from dojo.kernels.rms_norm import fused_rms_norm_backward
        x, w = _small_x()
        dy   = torch.randn_like(x)
        dx, dw = fused_rms_norm_backward(x, w, dy)
        assert dx.shape == x.shape
        assert dw.shape == w.shape

    def test_rms_norm_backward_emits_evidence(self):
        """fused_rms_norm_backward must emit evidence."""
        from dojo.kernels.rms_norm import fused_rms_norm_backward
        x, w = _small_x()
        dy   = torch.randn_like(x)
        fused_rms_norm_backward(x, w, dy)
        events = EvidenceLogger._events
        types = [e["evidence"].get("type") for e in events]
        assert EvidenceType.TRAINING_BACKEND_SELECTED.value in types

    def test_rms_norm_backward_numerically_correct(self):
        """dx from backward must match analytically-derived reference.

        For y = x / rms(x) * w where rms(x) = sqrt(mean(x²) + eps):
            dx_i = (dy_i * w_i - x_i * sum_j(dy_j * w_j * x_j) / (H * rms²)) / rms
        This is the closed-form gradient — no autograd dependency needed.
        """
        from dojo.kernels.rms_norm import fused_rms_norm_backward
        torch.manual_seed(5)
        x   = torch.randn(4, 16)
        w   = torch.ones(16)
        dy  = torch.randn(4, 16)
        eps = 1e-6
        H   = 16

        x_f  = x.float()
        dy_f = dy.float()
        w_f  = w.float()

        mean_sq = (x_f * x_f).mean(dim=-1, keepdim=True)
        rms_inv = torch.rsqrt(mean_sq + eps)
        x_hat   = x_f * rms_inv

        # Closed-form dx: (dy*w - x_hat * dot(dy*w, x_hat)/H) * rms_inv
        dot     = (dy_f * w_f * x_hat).sum(dim=-1, keepdim=True) / H
        dx_ref  = (dy_f * w_f - x_hat * dot) * rms_inv

        dx, _ = fused_rms_norm_backward(x, w, dy, eps=eps)
        assert torch.allclose(dx.float(), dx_ref, atol=1e-4), (
            f"Max dx diff: {(dx.float() - dx_ref).abs().max().item()}"
        )

    def test_flash_attention_non_square_seq(self):
        """Attention must handle non-square (M != N) query/key lengths."""
        from dojo.kernels.attention import flash_attention_forward
        B, H, M, N, D = 1, 2, 6, 10, 16
        q = torch.randn(B, H, M, D)
        k = torch.randn(B, H, N, D)
        v = torch.randn(B, H, N, D)
        out = flash_attention_forward(q, k, v)
        assert out.shape == (B, H, M, D)

    def test_flash_attention_evidence_kernel_is_pytorch_fallback(self):
        """In CI mode the evidence kernel field must say 'pytorch_fallback'."""
        from dojo.kernels.attention import flash_attention_forward
        q, k, v = _small_qkv()
        flash_attention_forward(q, k, v)
        for e in reversed(EvidenceLogger._events):
            ev = e.get("evidence", {})
            if ev.get("type") == EvidenceType.TRAINING_BACKEND_SELECTED.value:
                assert ev["data"]["kernel"] == "pytorch_fallback"
                break

    def test_registry_probe_idempotent(self):
        """Calling probe() twice must return the same keys both times."""
        from dojo.kernels.registry import KernelRegistry
        KernelRegistry.clear()
        reg = KernelRegistry()
        r1 = reg.probe()
        r2 = reg.probe()
        assert set(r1.keys()) == set(r2.keys())

    def test_registry_singleton(self):
        """get_instance() must return the same object on repeated calls."""
        from dojo.kernels.registry import KernelRegistry
        KernelRegistry.clear()
        a = KernelRegistry.get_instance()
        b = KernelRegistry.get_instance()
        assert a is b

    def test_dispatch_attention_emits_one_door_evidence(self):
        """dispatch_attention must emit one_door.entry evidence."""
        from dojo.kernels.dispatch import dispatch_attention
        from dojo.kernels.registry import KernelRegistry
        KernelRegistry.clear()
        q, k, v = _small_qkv()
        dispatch_attention(q, k, v)
        types = [
            e["evidence"].get("type")
            for e in EvidenceLogger._events
        ]
        assert EvidenceType.ONE_DOOR_ENTRY.value in types

    def test_dispatch_rms_norm_emits_one_door_evidence(self):
        """dispatch_rms_norm must emit one_door.entry evidence."""
        from dojo.kernels.dispatch import dispatch_rms_norm
        from dojo.kernels.registry import KernelRegistry
        KernelRegistry.clear()
        x, w = _small_x()
        dispatch_rms_norm(x, w)
        types = [
            e["evidence"].get("type")
            for e in EvidenceLogger._events
        ]
        assert EvidenceType.ONE_DOOR_ENTRY.value in types

    def test_precision_unsupported_raised_for_fp8_no_cuda(self):
        """PrecisionUnsupported must be raised for FP8 when CUDA is absent."""
        from dojo.kernels.attention import flash_attention_forward
        from dojo.core.exceptions import PrecisionUnsupported

        # Only run if FP8 dtype exists in this PyTorch build
        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("FP8 dtype not available in this PyTorch build")

        q = torch.randn(1, 2, 8, 16).to(torch.float8_e4m3fn)
        k = torch.randn(1, 2, 8, 16).to(torch.float8_e4m3fn)
        v = torch.randn(1, 2, 8, 16).to(torch.float8_e4m3fn)

        # On CPU (no CUDA), FP8 must raise PrecisionUnsupported
        if not torch.cuda.is_available():
            with pytest.raises(PrecisionUnsupported):
                flash_attention_forward(q, k, v)
        else:
            pytest.skip("CUDA available — FP8 guard test not applicable here")

    def test_rms_norm_3d_input_shape_preserved(self):
        """fused_rms_norm must handle 3-D inputs (B, S, H) and preserve shape."""
        from dojo.kernels.rms_norm import fused_rms_norm
        x = torch.randn(2, 8, 16)
        w = torch.ones(16)
        out = fused_rms_norm(x, w)
        assert out.shape == x.shape

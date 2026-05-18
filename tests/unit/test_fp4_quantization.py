"""Tests for d12: real FP4 E2M1 quantization math (CPU-compatible).

Validates the 16-value FP4 grid, quantize/dequantize round-trip, error metrics,
and the fp4_matmul emulation path.
"""
from __future__ import annotations

import math

import pytest
import torch

from dojo.core.exceptions import PrecisionUnsupported
from dojo.quant.fp4 import (
    _FP4_GRID,
    _FP4_POSITIVE_VALUES,
    compute_fp4_quantization_error,
    dequantize_tensor_fp4,
    fp4_matmul,
    quantize_tensor_fp4,
    requires_blackwell,
)


# ---------------------------------------------------------------------------
# FP4 grid
# ---------------------------------------------------------------------------

class TestFp4Grid:
    def test_positive_grid_has_8_values(self):
        assert _FP4_POSITIVE_VALUES.numel() == 8

    def test_full_grid_has_16_values(self):
        assert _FP4_GRID.numel() == 16

    def test_positive_grid_contains_zero(self):
        assert 0.0 in _FP4_POSITIVE_VALUES.tolist()

    def test_positive_grid_max_is_six(self):
        assert _FP4_POSITIVE_VALUES.max().item() == 6.0

    def test_positive_grid_includes_canonical_values(self):
        vals = set(_FP4_POSITIVE_VALUES.tolist())
        for v in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
            assert v in vals

    def test_full_grid_symmetric_around_zero(self):
        vals = _FP4_GRID.tolist()
        for v in _FP4_POSITIVE_VALUES.tolist():
            if v != 0.0:
                assert -v in vals


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

class TestQuantize:
    def test_quantized_values_are_on_grid(self):
        t = torch.randn(20)
        q, _ = quantize_tensor_fp4(t)
        for v in q.flatten().tolist():
            assert any(math.isclose(v, g, abs_tol=1e-6) for g in _FP4_GRID.tolist()), (
                f"Value {v} not on FP4 grid"
            )

    def test_quantize_preserves_shape(self):
        t = torch.randn(3, 4, 5)
        q, _ = quantize_tensor_fp4(t)
        assert q.shape == t.shape

    def test_scale_is_positive_for_nonzero_input(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        _, scale = quantize_tensor_fp4(t)
        assert scale > 0

    def test_explicit_scale_is_used(self):
        t = torch.tensor([0.0, 1.0])
        _, scale = quantize_tensor_fp4(t, scale=2.5)
        assert scale == 2.5

    def test_zero_tensor_quantizes_to_zero(self):
        t = torch.zeros(5)
        q, _ = quantize_tensor_fp4(t)
        assert torch.all(q == 0.0)

    def test_rejects_integer_tensor(self):
        with pytest.raises(PrecisionUnsupported):
            quantize_tensor_fp4(torch.tensor([1, 2, 3]))

    def test_grid_value_quantizes_to_itself(self):
        for v in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
            t = torch.tensor([v])
            q, _ = quantize_tensor_fp4(t, scale=1.0)
            assert q[0].item() == v


# ---------------------------------------------------------------------------
# Dequantization
# ---------------------------------------------------------------------------

class TestDequantize:
    def test_dequantize_inverts_scale(self):
        t = torch.tensor([0.0, 1.0, 2.0])
        q, scale = quantize_tensor_fp4(t, scale=0.5)
        r = dequantize_tensor_fp4(q, scale)
        assert torch.allclose(r, q * 0.5)

    def test_dequantize_preserves_shape(self):
        t = torch.randn(2, 3, 4)
        q, scale = quantize_tensor_fp4(t)
        r = dequantize_tensor_fp4(q, scale)
        assert r.shape == t.shape

    def test_dequantize_target_dtype_respected(self):
        t = torch.randn(5)
        q, scale = quantize_tensor_fp4(t)
        r = dequantize_tensor_fp4(q, scale, target_dtype=torch.float64)
        assert r.dtype == torch.float64


# ---------------------------------------------------------------------------
# Round-trip error
# ---------------------------------------------------------------------------

class TestQuantizationError:
    def test_zero_tensor_has_zero_error(self):
        err = compute_fp4_quantization_error(torch.zeros(10))
        assert err["max_abs_error"] == 0.0
        assert err["rmse"] == 0.0

    def test_random_tensor_error_is_finite(self):
        err = compute_fp4_quantization_error(torch.randn(100))
        assert math.isfinite(err["max_abs_error"])
        assert math.isfinite(err["rmse"])

    def test_error_dict_has_required_keys(self):
        err = compute_fp4_quantization_error(torch.randn(10))
        for k in ("max_abs_error", "rmse", "snr_db", "scale"):
            assert k in err

    def test_snr_positive_for_random_signal(self):
        torch.manual_seed(0)
        err = compute_fp4_quantization_error(torch.randn(1000))
        assert err["snr_db"] > 0


# ---------------------------------------------------------------------------
# fp4_matmul
# ---------------------------------------------------------------------------

class TestFp4Matmul:
    def test_matmul_output_shape(self):
        a = torch.randn(4, 8)
        b = torch.randn(8, 5)
        out, _ = fp4_matmul(a, b)
        assert out.shape == (4, 5)

    def test_matmul_metadata_has_scales(self):
        a = torch.randn(2, 3)
        b = torch.randn(3, 4)
        _, meta = fp4_matmul(a, b)
        assert "scale_a" in meta
        assert "scale_b" in meta

    def test_zero_matmul_returns_zero(self):
        a = torch.zeros(3, 4)
        b = torch.zeros(4, 5)
        out, _ = fp4_matmul(a, b)
        assert torch.all(out == 0)


# ---------------------------------------------------------------------------
# Hardware capability gate
# ---------------------------------------------------------------------------

class TestBlackwellGate:
    def test_blackwell_required_for_sm_100(self):
        assert requires_blackwell(10.0)

    def test_blackwell_not_required_for_ada(self):
        assert not requires_blackwell(8.9)

    def test_blackwell_not_required_for_hopper(self):
        assert not requires_blackwell(9.0)

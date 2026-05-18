"""Tests for d22: pure-Python GPTQ block quantization emulation."""
from __future__ import annotations

import pytest
import torch

from dojo.core.exceptions import QuantizationError
from dojo.quant.gptq_emulation import (
    GptqQuantized,
    gptq_dequantize_weight,
    gptq_quantize_weight,
    gptq_round_trip_error,
)


def _weight(out=8, inp=128) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(out, inp)


class TestInputValidation:
    def test_rejects_1d_weight(self):
        with pytest.raises(QuantizationError):
            gptq_quantize_weight(torch.randn(128))

    def test_rejects_bad_bits(self):
        with pytest.raises(QuantizationError):
            gptq_quantize_weight(_weight(), bits=9)

    def test_rejects_zero_group_size(self):
        with pytest.raises(QuantizationError):
            gptq_quantize_weight(_weight(), group_size=0)

    def test_rejects_misaligned_group_size(self):
        with pytest.raises(QuantizationError):
            gptq_quantize_weight(_weight(out=4, inp=100), group_size=128)


class TestQuantize:
    def test_returns_gptq_quantized_dataclass(self):
        q = gptq_quantize_weight(_weight())
        assert isinstance(q, GptqQuantized)

    def test_quantized_values_in_4bit_range(self):
        q = gptq_quantize_weight(_weight(), bits=4)
        assert int(q.quantized.min()) >= 0
        assert int(q.quantized.max()) <= 15

    def test_quantized_values_in_8bit_range(self):
        q = gptq_quantize_weight(_weight(), bits=8)
        assert int(q.quantized.min()) >= 0
        assert int(q.quantized.max()) <= 255

    def test_scales_shape(self):
        q = gptq_quantize_weight(_weight(out=8, inp=128), group_size=128)
        assert q.scales.shape == (8, 1)

    def test_zero_points_shape(self):
        q = gptq_quantize_weight(_weight(out=8, inp=128), group_size=128)
        assert q.zero_points.shape == (8, 1)

    def test_preserves_shape(self):
        w = _weight(out=8, inp=256)
        q = gptq_quantize_weight(w)
        assert q.quantized.shape == w.shape


class TestRoundTrip:
    def test_dequantize_preserves_shape(self):
        w = _weight()
        q = gptq_quantize_weight(w)
        r = gptq_dequantize_weight(q)
        assert r.shape == w.shape

    def test_round_trip_error_finite(self):
        err = gptq_round_trip_error(_weight())
        assert all(err[k] >= 0 for k in ["max_abs_error", "rmse", "relative_error"])

    def test_higher_bits_lower_error(self):
        w = _weight()
        e4 = gptq_round_trip_error(w, bits=4)
        e8 = gptq_round_trip_error(w, bits=8)
        assert e8["rmse"] < e4["rmse"]

    def test_compression_ratio_above_one_for_4bit(self):
        err = gptq_round_trip_error(_weight(out=8, inp=128), bits=4)
        assert err["compression_ratio"] > 1.0


class TestEvidence:
    def test_evidence_emitted(self):
        from dojo.core.evidence import EvidenceLogger
        EvidenceLogger.clear()
        gptq_quantize_weight(_weight())
        events = EvidenceLogger.get_events()
        assert any(e["data"].get("operation") == "gptq_quantize" for e in events)

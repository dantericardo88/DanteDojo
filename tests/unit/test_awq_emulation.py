"""Tests for d23: pure-Python AWQ activation-aware quantization emulation."""
from __future__ import annotations

import pytest
import torch

from dojo.core.exceptions import QuantizationError
from dojo.quant.awq_emulation import (
    AwqQuantized,
    awq_dequantize_weight,
    awq_quantize_weight,
    awq_round_trip_error,
    compute_activation_scale,
)


def _weight(out=8, inp=128) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(out, inp)


def _activations(n_samples=32, inp=128) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(n_samples, inp)


class TestInputValidation:
    def test_rejects_1d_weight(self):
        with pytest.raises(QuantizationError):
            awq_quantize_weight(torch.randn(128))

    def test_rejects_bad_bits(self):
        with pytest.raises(QuantizationError):
            awq_quantize_weight(_weight(), bits=1)

    def test_rejects_misaligned_group_size(self):
        with pytest.raises(QuantizationError):
            awq_quantize_weight(_weight(out=4, inp=100), group_size=128)


class TestActivationScale:
    def test_computes_per_channel_scale(self):
        s = compute_activation_scale(_weight(), _activations())
        assert s.shape == (128,)

    def test_scale_is_positive(self):
        s = compute_activation_scale(_weight(), _activations())
        assert (s > 0).all()

    def test_rejects_mismatched_in_features(self):
        with pytest.raises(QuantizationError):
            compute_activation_scale(_weight(), torch.randn(32, 64))

    def test_rejects_1d_activations(self):
        with pytest.raises(QuantizationError):
            compute_activation_scale(_weight(), torch.randn(128))


class TestQuantize:
    def test_returns_awq_quantized(self):
        q = awq_quantize_weight(_weight())
        assert isinstance(q, AwqQuantized)

    def test_quantized_values_in_4bit_range(self):
        q = awq_quantize_weight(_weight(), bits=4)
        assert int(q.quantized.min()) >= 0
        assert int(q.quantized.max()) <= 15

    def test_with_activation_scale(self):
        w = _weight()
        a = compute_activation_scale(w, _activations())
        q = awq_quantize_weight(w, activation_scale=a)
        assert q.activation_scale.shape == (128,)

    def test_default_activation_scale_is_ones(self):
        q = awq_quantize_weight(_weight())
        assert torch.allclose(q.activation_scale, torch.ones(128))


class TestRoundTrip:
    def test_dequantize_preserves_shape(self):
        w = _weight()
        q = awq_quantize_weight(w)
        r = awq_dequantize_weight(q)
        assert r.shape == w.shape

    def test_round_trip_error_finite(self):
        err = awq_round_trip_error(_weight())
        assert err["max_abs_error"] >= 0
        assert err["rmse"] >= 0

    def test_activation_weighted_error_reported(self):
        err = awq_round_trip_error(_weight(), sample_activations=_activations())
        assert "activation_weighted_rmse" in err
        assert "activation_weighted_max_error" in err

    def test_higher_bits_lower_error(self):
        w = _weight()
        e4 = awq_round_trip_error(w, bits=4)
        e8 = awq_round_trip_error(w, bits=8)
        assert e8["rmse"] < e4["rmse"]


class TestEvidence:
    def test_evidence_emitted(self):
        from dojo.core.evidence import EvidenceLogger
        EvidenceLogger.clear()
        awq_quantize_weight(_weight())
        events = EvidenceLogger.get_events()
        assert any(e["data"].get("operation") == "awq_quantize" for e in events)

    def test_evidence_flags_activation_aware(self):
        from dojo.core.evidence import EvidenceLogger
        w = _weight()
        a = compute_activation_scale(w, _activations())
        EvidenceLogger.clear()
        awq_quantize_weight(w, activation_scale=a)
        events = EvidenceLogger.get_events()
        awq_evs = [e for e in events if e["data"].get("operation") == "awq_quantize"]
        assert awq_evs[0]["data"]["activation_aware"] is True

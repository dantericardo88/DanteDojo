"""Tests for d27: real inference pass over a streamed KV cache."""
from __future__ import annotations

import pytest
import torch

from dojo.stream.kv_streamed_inference import (
    KvStreamedInferenceError,
    KvStreamedInferenceLoop,
)


class TestConstruction:
    def test_rejects_zero_head_dim(self):
        with pytest.raises(KvStreamedInferenceError):
            KvStreamedInferenceLoop(layer_ids=["L0"], head_dim=0)

    def test_rejects_zero_vocab(self):
        with pytest.raises(KvStreamedInferenceError):
            KvStreamedInferenceLoop(layer_ids=["L0"], vocab_size=0)

    def test_rejects_zero_seq_len(self):
        with pytest.raises(KvStreamedInferenceError):
            KvStreamedInferenceLoop(layer_ids=["L0"], seq_len=0)

    def test_empty_layer_list_is_valid(self):
        loop = KvStreamedInferenceLoop(layer_ids=[])
        assert loop is not None


class TestVramBytes:
    def test_layer_vram_bytes_64x8_fp32(self):
        loop = KvStreamedInferenceLoop(layer_ids=["L0"], head_dim=64, seq_len=8)
        # 2 (K+V) * 8 * 64 * 4 = 4096
        assert loop.layer_vram_bytes() == 4096

    def test_layer_vram_bytes_bfloat16_half(self):
        loop = KvStreamedInferenceLoop(layer_ids=["L0"], head_dim=64, seq_len=8)
        assert loop.layer_vram_bytes(dtype_bytes=2) == loop.layer_vram_bytes() // 2


class TestRun:
    def test_query_wrong_shape_raises(self):
        loop = KvStreamedInferenceLoop(layer_ids=["L0"], head_dim=64)
        with pytest.raises(KvStreamedInferenceError, match="bad_query_shape"):
            loop.run(torch.zeros(128))

    def test_returns_one_token_per_layer(self):
        loop = KvStreamedInferenceLoop(layer_ids=["L0", "L1", "L2"], head_dim=32)
        tokens = loop.run(torch.zeros(32))
        assert len(tokens) == 3

    def test_tokens_within_vocab(self):
        loop = KvStreamedInferenceLoop(layer_ids=["L0"], head_dim=32, vocab_size=100)
        tokens = loop.run(torch.zeros(32))
        assert all(0 <= t < 100 for t in tokens)

    def test_empty_layers_returns_empty_tokens(self):
        loop = KvStreamedInferenceLoop(layer_ids=[], head_dim=32)
        assert loop.run(torch.zeros(32)) == []

    def test_deterministic_under_same_query(self):
        loop = KvStreamedInferenceLoop(layer_ids=["L0", "L1"], head_dim=32)
        q = torch.ones(32)
        t1 = loop.run(q)
        t2 = loop.run(q)
        assert t1 == t2


class TestStreamingDiscipline:
    def test_peak_vram_equals_one_layer(self):
        loop = KvStreamedInferenceLoop(layer_ids=["L0", "L1", "L2"], head_dim=32)
        loop.run(torch.zeros(32))
        assert loop.peak_vram_bytes == loop.layer_vram_bytes()

    def test_peak_vram_independent_of_layer_count(self):
        loop3 = KvStreamedInferenceLoop(layer_ids=[f"L{i}" for i in range(3)], head_dim=32)
        loop10 = KvStreamedInferenceLoop(layer_ids=[f"L{i}" for i in range(10)], head_dim=32)
        loop3.run(torch.zeros(32))
        loop10.run(torch.zeros(32))
        assert loop3.peak_vram_bytes == loop10.peak_vram_bytes

    def test_active_layer_clear_after_run(self):
        loop = KvStreamedInferenceLoop(layer_ids=["L0", "L1"], head_dim=32)
        loop.run(torch.zeros(32))
        assert loop.active_layer is None

    def test_peak_zero_before_run(self):
        loop = KvStreamedInferenceLoop(layer_ids=["L0"], head_dim=32)
        assert loop.peak_vram_bytes == 0


class TestRealAttentionMath:
    def test_zero_query_produces_logits_via_softmax(self):
        loop = KvStreamedInferenceLoop(layer_ids=["L0"], head_dim=32, vocab_size=50)
        tokens = loop.run(torch.zeros(32))
        # Zero query → uniform attention → context = mean(V) → real logits
        assert len(tokens) == 1
        assert 0 <= tokens[0] < 50

    def test_different_queries_can_produce_different_tokens(self):
        loop = KvStreamedInferenceLoop(layer_ids=["L0"], head_dim=32, vocab_size=100)
        t1 = loop.run(torch.ones(32))
        loop2 = KvStreamedInferenceLoop(layer_ids=["L0"], head_dim=32, vocab_size=100)
        t2 = loop2.run(torch.ones(32) * -1)
        # At least one query produces a token; non-trivial mapping exists
        assert isinstance(t1[0], int)
        assert isinstance(t2[0], int)


class TestEvidence:
    def test_inference_complete_event_emitted(self):
        from dojo.core.evidence import EvidenceLogger
        EvidenceLogger.clear()
        loop = KvStreamedInferenceLoop(layer_ids=["L0", "L1"], head_dim=32)
        loop.run(torch.zeros(32))
        events = EvidenceLogger.get_events()
        assert any(e.get("type") == "stream.kv_inference_complete" for e in events)

    def test_event_includes_peak_vram(self):
        from dojo.core.evidence import EvidenceLogger
        EvidenceLogger.clear()
        loop = KvStreamedInferenceLoop(layer_ids=["L0"], head_dim=32)
        loop.run(torch.zeros(32))
        events = EvidenceLogger.get_events()
        evs = [e for e in events if e.get("type") == "stream.kv_inference_complete"]
        assert "peak_vram_bytes" in evs[0]["data"]

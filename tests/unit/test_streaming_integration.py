"""Integration tests for LayerStreamingInferenceLoop, KVCacheStreamer, and PrefetchGovernor."""
from __future__ import annotations

from typing import List, Optional
from unittest.mock import patch

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.stream.inference_loop import LayerStreamingInferenceLoop
from dojo.stream.kv_cache_streamer import KVCacheStreamer
from dojo.stream.prefetch_governor import PrefetchGovernor
from dojo.stream.weight_streamer import WeightStreamer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    """Clear EvidenceLogger before and after every test."""
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


def _make_loop(num_layers: int = 4) -> LayerStreamingInferenceLoop:
    """Return a LayerStreamingInferenceLoop with mock WeightStreamer."""
    layer_ids = [f"layer_{i}" for i in range(num_layers)]
    streamer = WeightStreamer(mock_mode=True)
    return LayerStreamingInferenceLoop(
        layer_ids=layer_ids,
        streamer=streamer,
        mock_mode=True,
    )


def _get_events_of_type(event_type: str) -> List[dict]:
    return [e for e in EvidenceLogger.get_events() if e.get("type") == event_type]


# ===========================================================================
# TestLayerStreamingInferenceLoopIntegration
# ===========================================================================

class TestLayerStreamingInferenceLoopIntegration:
    """Integration tests for LayerStreamingInferenceLoop.run()."""

    def test_run_returns_output_tokens(self):
        """run([1,2,3]) returns a non-empty list."""
        loop = _make_loop(num_layers=4)
        result = loop.run([1, 2, 3])
        assert isinstance(result, list)
        assert len(result) > 0

    def test_output_is_list_of_ints(self):
        """Every element of the output is an int."""
        loop = _make_loop(num_layers=4)
        result = loop.run([1, 2, 3])
        for token in result:
            assert isinstance(token, int), f"Expected int, got {type(token)}: {token}"

    def test_run_emits_serve_start_evidence(self):
        """After run(), EvidenceLogger has a 'serve.start' event with mode='layer_streaming'."""
        loop = _make_loop(num_layers=3)
        loop.run([10, 20])

        events = _get_events_of_type("serve.start")
        assert len(events) >= 1, "Expected at least one serve.start event"
        data = events[0].get("data", {})
        assert data.get("mode") == "layer_streaming", f"mode mismatch: {data}"

    def test_run_emits_serve_ready_evidence(self):
        """After run(), EvidenceLogger has a 'serve.ready' event."""
        loop = _make_loop(num_layers=3)
        loop.run([5, 6, 7])

        events = _get_events_of_type("serve.ready")
        assert len(events) >= 1, "Expected at least one serve.ready event"

    def test_layers_are_evicted_after_run(self):
        """WeightStreamer has 0 loaded layers (active_layer is None) after run() completes."""
        streamer = WeightStreamer(mock_mode=True)
        loop = LayerStreamingInferenceLoop(
            layer_ids=["l0", "l1", "l2"],
            streamer=streamer,
            mock_mode=True,
        )
        loop.run([1, 2, 3])
        assert streamer.active_layer is None, (
            f"Expected active_layer to be None after run, got {streamer.active_layer!r}"
        )

    def test_vram_budget_one_layer_max(self):
        """At no point during run() are more than 1 layer loaded simultaneously."""
        layer_ids = ["l0", "l1", "l2", "l3"]
        streamer = WeightStreamer(mock_mode=True)

        max_concurrent = 0
        original_stream = streamer.stream_layer
        original_evict = streamer.evict_current

        concurrent_count = [0]

        def tracking_stream(layer_id, source_tier="disk"):
            concurrent_count[0] += 1
            nonlocal max_concurrent
            if concurrent_count[0] > max_concurrent:
                max_concurrent = concurrent_count[0]
            return original_stream(layer_id, source_tier)

        def tracking_evict():
            result = original_evict()
            if concurrent_count[0] > 0:
                concurrent_count[0] -= 1
            return result

        streamer.stream_layer = tracking_stream
        streamer.evict_current = tracking_evict

        loop = LayerStreamingInferenceLoop(
            layer_ids=layer_ids,
            streamer=streamer,
            mock_mode=True,
        )
        loop.run([1])

        assert max_concurrent <= 1, (
            f"VRAM budget breached: {max_concurrent} layers loaded simultaneously"
        )

    def test_run_with_many_layers(self):
        """run() with 32 layers completes without error."""
        layer_ids = [f"layer_{i}" for i in range(32)]
        streamer = WeightStreamer(mock_mode=True)
        loop = LayerStreamingInferenceLoop(
            layer_ids=layer_ids,
            streamer=streamer,
            mock_mode=True,
        )
        result = loop.run([1, 2, 3, 4])
        assert isinstance(result, list)
        assert len(result) == 32  # one output token per layer

    def test_empty_prompt_tokens_handled(self):
        """run([]) returns a list and must not raise."""
        loop = _make_loop(num_layers=4)
        result = loop.run([])
        assert isinstance(result, list)


# ===========================================================================
# TestKVCacheStreamerIntegration
# ===========================================================================

class TestKVCacheStreamerIntegration:
    """Integration tests for KVCacheStreamer."""

    def test_construction_with_budget(self):
        """KVCacheStreamer can be constructed with a custom budget."""
        kv = KVCacheStreamer(max_cache_size=512, mock_mode=True)
        assert kv.max_cache_size == 512
        assert kv.current_size == 0

    def test_evict_stays_within_budget(self):
        """KVCacheStreamer can evict entries to stay within budget."""
        kv = KVCacheStreamer(max_cache_size=3, mock_mode=True)
        kv.page_kv_entry("k0", "v0", 1)
        kv.page_kv_entry("k1", "v1", 1)
        kv.page_kv_entry("k2", "v2", 1)
        assert kv.current_size == 3

        # Now the cache is full; evict one to make room
        evicted = kv.evict_oldest(n=1)
        assert evicted == 1
        assert kv.current_size == 2

        # Should be able to add a new entry
        kv.page_kv_entry("k3", "v3", 1)
        assert kv.current_size == 3

    def test_eviction_emits_evidence(self):
        """KVCacheStreamer emits evidence events on eviction."""
        kv = KVCacheStreamer(max_cache_size=10, mock_mode=True)
        kv.page_kv_entry("key_a", "val_a", 3)
        kv.page_kv_entry("key_b", "val_b", 3)

        kv.evict_oldest(n=1)

        eviction_events = _get_events_of_type("stream.layer.evicted")
        assert len(eviction_events) >= 1, "Expected at least one eviction evidence event"

    def test_integrates_with_inference_loop(self):
        """KVCacheStreamer and LayerStreamingInferenceLoop can run side-by-side without conflict."""
        kv = KVCacheStreamer(max_cache_size=50, mock_mode=True)
        loop = _make_loop(num_layers=4)

        # Run inference loop
        output = loop.run([1, 2, 3])
        assert isinstance(output, list)

        # Store KV entries for each output token
        for i, token in enumerate(output):
            kv.page_kv_entry(f"out_token_{i}", str(token), 1)

        assert kv.current_size == len(output)

        # Verify evidence was emitted from both subsystems
        serve_events = _get_events_of_type("serve.start")
        kv_events = _get_events_of_type("stream.kv.stored")
        assert len(serve_events) >= 1
        assert len(kv_events) >= 1


# ===========================================================================
# TestPrefetchGovernorIntegration
# ===========================================================================

class TestPrefetchGovernorIntegration:
    """Integration tests for PrefetchGovernor."""

    def test_construction(self):
        """PrefetchGovernor can be constructed with a lookahead."""
        gov = PrefetchGovernor(max_lookahead=4, mock_mode=True)
        assert gov.max_lookahead == 4
        assert gov.pending == []

    def test_schedules_prefetch_for_upcoming_layers(self):
        """PrefetchGovernor schedules prefetch for upcoming layers and tracks them in pending."""
        gov = PrefetchGovernor(max_lookahead=4, mock_mode=True)
        layers = ["layer_1", "layer_2", "layer_3"]
        result = gov.issue_prefetch(layers)

        assert result.status == "PrefetchIssued"
        assert set(result.layers_issued) == set(layers)
        # Pending should contain the issued layers
        for layer in layers:
            assert layer in gov.pending

    def test_does_not_exceed_vram_budget(self):
        """PrefetchGovernor raises PrefetchGovernorError when lookahead window would be exceeded."""
        from dojo.stream.prefetch_governor import PrefetchGovernorError
        from dojo.core.exceptions import SovereignHalt

        gov = PrefetchGovernor(max_lookahead=2, mock_mode=True)
        gov.issue_prefetch(["l0", "l1"])  # fills the window

        # Now any additional prefetch request must be rejected
        with pytest.raises((PrefetchGovernorError, SovereignHalt)):
            gov.issue_prefetch(["l2"])

    def test_integrates_with_weight_streamer(self):
        """PrefetchGovernor and WeightStreamer work together: prefetched layers can be streamed."""
        gov = PrefetchGovernor(max_lookahead=4, mock_mode=True)
        streamer = WeightStreamer(mock_mode=True)

        layers_to_prefetch = ["layer_0", "layer_1"]
        gov.issue_prefetch(layers_to_prefetch)

        # Simulate consuming a prefetched layer and streaming it
        consumed = gov.consume_prefetch("layer_0")
        assert consumed is True

        # Now stream that layer via WeightStreamer
        stream_result = streamer.stream_layer("layer_0", source_tier="ram")
        assert stream_result.status == "LayerReady"
        assert stream_result.layer_id == "layer_0"

        # Evict after use
        evicted = streamer.evict_current()
        assert evicted == "layer_0"
        assert streamer.active_layer is None

        # Verify evidence from both subsystems
        prefetch_events = _get_events_of_type("stream.prefetch.issued")
        stream_events = _get_events_of_type("stream.layer.loaded")
        assert len(prefetch_events) >= 1
        assert len(stream_events) >= 1

"""Tests for VRAM-bounded layer streaming inference loop (d26).

All tests use mock_mode=True and are CI-safe (no GPU required).
"""

from __future__ import annotations

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.stream.inference_loop import InferenceLoopError, LayerStreamingInferenceLoop
from dojo.stream.weight_streamer import WeightStreamer, WeightStreamerError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    """Reset the singleton EvidenceLogger before each test."""
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


LAYER_IDS = ["layer_0", "layer_1", "layer_2", "layer_3"]
PROMPT = [1, 2, 3]


# ---------------------------------------------------------------------------
# 1. All layers produce output tokens
# ---------------------------------------------------------------------------

def test_layer_streaming_loop_processes_all_layers():
    loop = LayerStreamingInferenceLoop(layer_ids=LAYER_IDS, mock_mode=True)
    output = loop.run(PROMPT)
    assert len(output) == 4, f"Expected 4 output tokens, got {len(output)}"


# ---------------------------------------------------------------------------
# 2. active_layer is always None after each step
# ---------------------------------------------------------------------------

def test_layer_streaming_loop_never_has_two_active_layers():
    streamer = WeightStreamer(mock_mode=True)
    evictions_checked: list[bool] = []

    loop = LayerStreamingInferenceLoop(
        layer_ids=LAYER_IDS,
        streamer=streamer,
        mock_mode=True,
    )

    # Monkey-patch evict_current to assert state immediately after eviction
    original_evict = streamer.evict_current

    def patched_evict():
        result = original_evict()
        evictions_checked.append(streamer.active_layer is None)
        return result

    streamer.evict_current = patched_evict  # type: ignore[method-assign]

    loop.run(PROMPT)

    assert len(evictions_checked) == len(LAYER_IDS)
    assert all(evictions_checked), "VRAM active_layer was not None after eviction"


# ---------------------------------------------------------------------------
# 3. LAYER_STREAM_EVICTED fires for each layer
# ---------------------------------------------------------------------------

def test_layer_streaming_loop_eviction_fires_each_layer():
    loop = LayerStreamingInferenceLoop(layer_ids=LAYER_IDS, mock_mode=True)
    loop.run(PROMPT)

    events = EvidenceLogger.get_events()
    eviction_events = [e for e in events if e["type"] == EvidenceType.LAYER_STREAM_EVICTED.value]
    assert len(eviction_events) == len(LAYER_IDS), (
        f"Expected {len(LAYER_IDS)} eviction events, got {len(eviction_events)}"
    )


# ---------------------------------------------------------------------------
# 4. SERVE_START emitted at the beginning
# ---------------------------------------------------------------------------

def test_layer_streaming_loop_emits_serve_start():
    loop = LayerStreamingInferenceLoop(layer_ids=LAYER_IDS, mock_mode=True)
    loop.run(PROMPT)

    events = EvidenceLogger.get_events()
    serve_start_events = [e for e in events if e["type"] == EvidenceType.SERVE_START.value]
    assert len(serve_start_events) >= 1, "Expected at least one SERVE_START event"
    # First SERVE_START should carry the layer_streaming mode
    first = serve_start_events[0]
    assert first["data"].get("mode") == "layer_streaming"
    assert first["data"].get("vram_budget_layers") == 1


# ---------------------------------------------------------------------------
# 5. SERVE_READY emitted at the end with layers_processed
# ---------------------------------------------------------------------------

def test_layer_streaming_loop_emits_serve_ready():
    loop = LayerStreamingInferenceLoop(layer_ids=LAYER_IDS, mock_mode=True)
    loop.run(PROMPT)

    events = EvidenceLogger.get_events()
    serve_ready_events = [e for e in events if e["type"] == EvidenceType.SERVE_READY.value]
    assert len(serve_ready_events) >= 1, "Expected at least one SERVE_READY event"
    last = serve_ready_events[-1]
    assert last["data"].get("layers_processed") == len(LAYER_IDS)
    assert last["data"].get("output_tokens") == len(LAYER_IDS)


# ---------------------------------------------------------------------------
# 6. vram_budget_check reports budget_ok=True when no layer active
# ---------------------------------------------------------------------------

def test_vram_budget_check_reports_ok():
    streamer = WeightStreamer(mock_mode=True)
    result = streamer.vram_budget_check()
    assert result["budget_ok"] is True
    assert result["active_layer"] is None
    assert result["max_active_layers"] == 1


# ---------------------------------------------------------------------------
# 7. stream_layer() twice without evict raises WeightStreamerError
# ---------------------------------------------------------------------------

def test_weight_streamer_blocks_second_layer_load():
    streamer = WeightStreamer(mock_mode=True)
    streamer.stream_layer("layer_0", source_tier="ram")
    with pytest.raises(WeightStreamerError, match="VRAM residency breach"):
        streamer.stream_layer("layer_1", source_tier="ram")


# ---------------------------------------------------------------------------
# 8. Complete evidence chain in order
# ---------------------------------------------------------------------------

def test_layer_streaming_loop_evidence_chain_complete():
    loop = LayerStreamingInferenceLoop(layer_ids=LAYER_IDS, mock_mode=True)
    loop.run(PROMPT)

    events = EvidenceLogger.get_events()
    types = [e["type"] for e in events]

    # SERVE_START must appear before any streaming events
    serve_start_idx = next(
        (i for i, t in enumerate(types) if t == EvidenceType.SERVE_START.value), None
    )
    assert serve_start_idx is not None, "SERVE_START missing"

    # SERVE_READY must appear last among our key events
    serve_ready_idx = next(
        (i for i, t in reversed(list(enumerate(types))) if t == EvidenceType.SERVE_READY.value),
        None,
    )
    assert serve_ready_idx is not None, "SERVE_READY missing"
    assert serve_ready_idx > serve_start_idx, "SERVE_READY must come after SERVE_START"

    # N LAYER_STREAM_LOADED events (one per layer)
    loaded_events = [t for t in types if t == EvidenceType.LAYER_STREAM_LOADED.value]
    assert len(loaded_events) >= len(LAYER_IDS), (
        f"Expected >= {len(LAYER_IDS)} LAYER_STREAM_LOADED, got {len(loaded_events)}"
    )

    # N LAYER_STREAM_EVICTED events (one per layer)
    evicted_events = [t for t in types if t == EvidenceType.LAYER_STREAM_EVICTED.value]
    assert len(evicted_events) == len(LAYER_IDS), (
        f"Expected {len(LAYER_IDS)} LAYER_STREAM_EVICTED, got {len(evicted_events)}"
    )

    # All LAYER_STREAM_LOADED events must come after SERVE_START
    loaded_indices = [i for i, t in enumerate(types) if t == EvidenceType.LAYER_STREAM_LOADED.value]
    assert all(i > serve_start_idx for i in loaded_indices), (
        "Some LAYER_STREAM_LOADED events appear before SERVE_START"
    )

    # All LAYER_STREAM_EVICTED events must come before SERVE_READY
    evicted_indices = [i for i, t in enumerate(types) if t == EvidenceType.LAYER_STREAM_EVICTED.value]
    assert all(i < serve_ready_idx for i in evicted_indices), (
        "Some LAYER_STREAM_EVICTED events appear after SERVE_READY"
    )

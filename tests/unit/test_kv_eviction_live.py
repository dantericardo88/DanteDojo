"""Tests for KV cache eviction during 512-token generation simulation (d27)."""

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.stream.kv_cache_streamer import KVCacheStreamer, KVCacheStreamerError


def _make_streamer(max_cache_size: int = 100) -> KVCacheStreamer:
    EvidenceLogger.clear()
    return KVCacheStreamer(max_cache_size=max_cache_size, mock_mode=True)


# ---------------------------------------------------------------------------
# 1. Eviction fires during 512-token generation
# ---------------------------------------------------------------------------

def test_kv_eviction_fires_during_512_token_generation():
    """With a small cache (64 tokens), 512 tokens must trigger evictions."""
    streamer = _make_streamer(max_cache_size=64)
    result = streamer.simulate_generation(n_tokens=512, tokens_per_page=1)
    assert result["evictions_fired"] > 0, (
        f"Expected evictions but none fired. result={result}"
    )


# ---------------------------------------------------------------------------
# 2. Cache stays within budget after generation
# ---------------------------------------------------------------------------

def test_kv_eviction_keeps_cache_within_budget():
    """final_cache_size must never exceed max_cache_size."""
    max_size = 50
    streamer = _make_streamer(max_cache_size=max_size)
    result = streamer.simulate_generation(n_tokens=512, tokens_per_page=1)
    assert result["final_cache_size"] <= max_size, (
        f"Cache exceeded budget: {result['final_cache_size']} > {max_size}"
    )
    # Also verify the streamer's own current_size agrees
    assert streamer.current_size <= max_size


# ---------------------------------------------------------------------------
# 3. Evidence events are emitted during generation
# ---------------------------------------------------------------------------

def test_kv_eviction_emits_evidence_events():
    """Evidence events must be emitted during simulation (every 64 tokens)."""
    streamer = _make_streamer(max_cache_size=64)
    streamer.simulate_generation(n_tokens=512, tokens_per_page=1)
    events = EvidenceLogger.get_events()
    # Expect at least 7 checkpoint events (i=64,128,192,256,320,384,448)
    checkpoint_events = [
        e for e in events
        if e.get("type") == EvidenceType.LAYER_STREAM_EVICTED.value
        and isinstance(e.get("data"), dict)
        and "tokens_processed" in e.get("data", {})
    ]
    assert len(checkpoint_events) >= 7, (
        f"Expected >= 7 evidence checkpoints, got {len(checkpoint_events)}"
    )


# ---------------------------------------------------------------------------
# 4. Evictions definitely needed with cache smaller than 500 tokens
# ---------------------------------------------------------------------------

def test_kv_eviction_500_tokens_needed():
    """Cache of 32 tokens must require evictions to fit 500 generated tokens."""
    streamer = _make_streamer(max_cache_size=32)
    result = streamer.simulate_generation(n_tokens=500, tokens_per_page=1)
    assert result["evictions_fired"] > 0
    assert result["final_cache_size"] <= 32


# ---------------------------------------------------------------------------
# 5. No eviction when cache is large enough
# ---------------------------------------------------------------------------

def test_kv_no_eviction_when_cache_large():
    """With a 10 000-token cache, 512 tokens must not trigger any evictions."""
    streamer = _make_streamer(max_cache_size=10_000)
    result = streamer.simulate_generation(n_tokens=512, tokens_per_page=1)
    assert result["evictions_fired"] == 0, (
        f"Unexpected evictions: {result['evictions_fired']}"
    )
    assert result["final_cache_size"] == 512


# ---------------------------------------------------------------------------
# 6. Eviction count is plausible (proportional to overflow)
# ---------------------------------------------------------------------------

def test_kv_eviction_fire_count_plausible():
    """Eviction count should be proportional to the overflow volume.

    With max_cache_size=10 and 512 tokens, each overflow evicts 1 entry,
    so we expect roughly 502 evictions (512 - 10 = 502 overflow insertions).
    Allow ±1 for boundary effects.
    """
    max_size = 10
    n_tokens = 512
    streamer = _make_streamer(max_cache_size=max_size)
    result = streamer.simulate_generation(n_tokens=n_tokens, tokens_per_page=1)
    expected = n_tokens - max_size
    # Allow a small tolerance
    assert abs(result["evictions_fired"] - expected) <= 2, (
        f"Eviction count {result['evictions_fired']} deviates too far from expected {expected}"
    )


# ---------------------------------------------------------------------------
# 7. Return value contains the correct token count
# ---------------------------------------------------------------------------

def test_kv_eviction_returns_correct_token_count():
    """simulate_generation must always report tokens_generated == n_tokens."""
    streamer = _make_streamer(max_cache_size=64)
    result = streamer.simulate_generation(n_tokens=512)
    assert result["tokens_generated"] == 512


# ---------------------------------------------------------------------------
# 8. page_kv_entry raises KVCacheStreamerError past limit (no eviction retry)
# ---------------------------------------------------------------------------

def test_kv_cache_streamer_page_kv_entry_respects_limit():
    """page_kv_entry must raise KVCacheStreamerError when cache would overflow."""
    streamer = _make_streamer(max_cache_size=5)
    # Fill to capacity
    for i in range(5):
        streamer.page_kv_entry(f"k{i}", f"v{i}", 1)
    assert streamer.current_size == 5
    # One more must raise
    with pytest.raises(KVCacheStreamerError):
        streamer.page_kv_entry("overflow", "v", 1)
    # Cache size must be unchanged
    assert streamer.current_size == 5

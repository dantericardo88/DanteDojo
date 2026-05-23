"""d27: KVCacheStreamer real value storage and cache-hit tests.

Verifies that:
- page_kv_entry() stores the value string (not just size metadata)
- get_cached_value() returns stored value on hit, None on miss
- get_cached_value() emits KV_PAGE_HIT evidence on hit
- evict_oldest() removes value from _cache_values
- LocalGGUFRunner.generate() returns cached text on second call
- LocalGGUFRunner.generate() returns backend="kv_cache" on hit
- LocalGGUFRunner.generate() does NOT call stub/Ollama on cache hit
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.stream.kv_cache_streamer import KVCacheStreamer, KVCacheStreamerError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ---------------------------------------------------------------------------
# KVCacheStreamer value storage and retrieval
# ---------------------------------------------------------------------------

class TestKVCacheValueStorage:
    def test_cache_miss_returns_none(self):
        streamer = KVCacheStreamer(max_cache_size=100, mock_mode=True)
        assert streamer.get_cached_value("nonexistent") is None

    def test_get_cached_value_returns_stored_value(self):
        streamer = KVCacheStreamer(max_cache_size=100, mock_mode=True)
        streamer.page_kv_entry("k1", "hello world", size=4)
        assert streamer.get_cached_value("k1") == "hello world"

    def test_stored_value_exact_match(self):
        streamer = KVCacheStreamer(max_cache_size=100, mock_mode=True)
        long_value = "the quick brown fox jumps over the lazy dog " * 5
        streamer.page_kv_entry("k_long", long_value, size=10)
        assert streamer.get_cached_value("k_long") == long_value

    def test_multiple_keys_independent(self):
        streamer = KVCacheStreamer(max_cache_size=100, mock_mode=True)
        streamer.page_kv_entry("a", "alpha text", size=2)
        streamer.page_kv_entry("b", "beta text", size=2)
        assert streamer.get_cached_value("a") == "alpha text"
        assert streamer.get_cached_value("b") == "beta text"

    def test_overwrite_key_updates_value(self):
        streamer = KVCacheStreamer(max_cache_size=100, mock_mode=True)
        streamer.page_kv_entry("k", "first value", size=4)
        # Adding a second entry with same key overwrites (dict semantics)
        # We need to free space first by using a smaller cache to avoid overflow
        streamer2 = KVCacheStreamer(max_cache_size=100, mock_mode=True)
        streamer2.page_kv_entry("k", "first value", size=4)
        streamer2.page_kv_entry("k", "second value", size=4)
        assert streamer2.get_cached_value("k") == "second value"


class TestKVCacheHitEvidence:
    def test_hit_emits_kv_page_hit_evidence(self):
        streamer = KVCacheStreamer(max_cache_size=100, mock_mode=True)
        streamer.page_kv_entry("ev_key", "cached response", size=3)
        EvidenceLogger.clear()  # clear the stored event, watch only the hit
        streamer.get_cached_value("ev_key")
        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        assert EvidenceType.KV_PAGE_HIT.value in types

    def test_miss_does_not_emit_kv_page_hit(self):
        streamer = KVCacheStreamer(max_cache_size=100, mock_mode=True)
        EvidenceLogger.clear()
        streamer.get_cached_value("missing_key")
        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        assert EvidenceType.KV_PAGE_HIT.value not in types

    def test_hit_evidence_contains_key(self):
        streamer = KVCacheStreamer(max_cache_size=100, mock_mode=True)
        streamer.page_kv_entry("target_key", "some text", size=5)
        EvidenceLogger.clear()
        streamer.get_cached_value("target_key")
        events = EvidenceLogger.get_events()
        hit_events = [e for e in events if e.get("type") == EvidenceType.KV_PAGE_HIT.value]
        assert hit_events, "KV_PAGE_HIT event must be present"
        assert hit_events[0]["data"]["key"] == "target_key"


class TestKVCacheEviction:
    def test_eviction_removes_value(self):
        streamer = KVCacheStreamer(max_cache_size=10, mock_mode=True)
        streamer.page_kv_entry("evict_me", "evicted content", size=5)
        streamer.evict_oldest(1)
        assert streamer.get_cached_value("evict_me") is None

    def test_eviction_preserves_remaining_values(self):
        streamer = KVCacheStreamer(max_cache_size=20, mock_mode=True)
        streamer.page_kv_entry("first", "content first", size=5)
        streamer.page_kv_entry("second", "content second", size=5)
        streamer.evict_oldest(1)  # evicts "first"
        assert streamer.get_cached_value("first") is None
        assert streamer.get_cached_value("second") == "content second"


# ---------------------------------------------------------------------------
# LocalGGUFRunner cache-hit integration
# ---------------------------------------------------------------------------

class TestLocalGGUFRunnerCacheHit:
    """generate() must serve from KV cache on second call with identical prompt."""

    @pytest.fixture(autouse=True)
    def _no_ollama(self, monkeypatch):
        """Force stub path — disable Ollama so tests are deterministic."""
        monkeypatch.setattr(
            "dojo.serve.llamacpp_backend._ollama_available",
            lambda *_args, **_kwargs: False,
        )

    def _make_runner(self):
        from dojo.serve.llamacpp_backend import LocalGGUFRunner
        return LocalGGUFRunner(
            model_path=Path("/tmp/fake.gguf"),
            llm=None,
            ollama_url="http://localhost:11434",
        )

    def test_second_call_returns_cached_text(self):
        """After first generate(), second call with same prompt hits cache."""
        runner = self._make_runner()
        # First call goes to stub (Ollama not running in CI)
        result1 = runner.generate("What is 2+2?")
        first_text = result1["text"]
        assert first_text  # first call produces some text

        # Second call — must hit the cache and return the SAME text
        result2 = runner.generate("What is 2+2?")
        assert result2["text"] == first_text

    def test_second_call_backend_is_kv_cache(self):
        """Second generate() with same prompt must report backend='kv_cache'."""
        runner = self._make_runner()
        runner.generate("Hello cache test")
        result2 = runner.generate("Hello cache test")
        assert result2["backend"] == "kv_cache"

    def test_first_call_not_kv_cache(self):
        """First generate() must NOT be served from cache (nothing stored yet)."""
        runner = self._make_runner()
        result = runner.generate("Fresh prompt unique xyz")
        assert result["backend"] != "kv_cache"

    def test_different_prompt_different_cache_key(self):
        """Different prompts have different cache keys — no cross-contamination."""
        runner = self._make_runner()
        r1 = runner.generate("prompt alpha")
        r2 = runner.generate("prompt beta")
        # Both are first calls for their respective prompts — neither from cache
        assert r1["backend"] != "kv_cache"
        assert r2["backend"] != "kv_cache"

    def test_cache_hit_emits_kv_page_hit_evidence(self):
        """Evidence chain must contain KV_PAGE_HIT after a cache hit."""
        EvidenceLogger.clear()
        runner = self._make_runner()
        runner.generate("evidence test prompt")
        EvidenceLogger.clear()  # reset: track only the second call
        runner.generate("evidence test prompt")
        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        assert EvidenceType.KV_PAGE_HIT.value in types

    def test_stub_not_called_on_cache_hit(self):
        """generate() must return before reaching stub code on a cache hit."""
        runner = self._make_runner()
        runner.generate("stub bypass test")  # first call — populates cache

        # Patch the stub branch to detect if it's reached
        with patch("dojo.serve.llamacpp_backend._ollama_available", return_value=False):
            with patch("dojo.serve.llamacpp_backend._call_ollama") as mock_ollama:
                result = runner.generate("stub bypass test")
                # Ollama should NOT be called — we returned from cache before reaching it
                mock_ollama.assert_not_called()
                assert result["backend"] == "kv_cache"

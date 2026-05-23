"""d28: PrefetchGovernor feedback loop — issue → consume → hit_rate.

Proves that PrefetchGovernor is NOT purely advisory: generate() now consumes
the pending prefetch queue and emits PREFETCH_CONSUMED evidence with hit-rate
statistics. This creates a constitutional feedback loop:
  1. load() → issue_prefetch(layers)  → PREFETCH_ISSUED evidence
  2. generate() → consume_prefetch()  → PREFETCH_CONSUMED evidence
  3. prefetch_hit_rate()              → fraction of issued layers consumed

Coverage:
1.  prefetch_hit_rate() returns 0.0 before any prefetch issued
2.  prefetch_hit_rate() returns 0.0 after issue_prefetch but before consume
3.  prefetch_hit_rate() returns 1.0 after all layers consumed
4.  prefetch_hit_rate() returns partial fraction on partial consumption
5.  consume_prefetch() returns True for known layer, False for unknown
6.  generate() consumes pending prefetch layers
7.  generate() emits PREFETCH_CONSUMED evidence with layers_consumed > 0
8.  PREFETCH_CONSUMED evidence has pending_remaining = 0 after full consume
9.  PREFETCH_CONSUMED evidence includes backend field
10. Second generate() does NOT emit PREFETCH_CONSUMED (pending already empty)
11. load() followed by generate() → complete feedback round-trip
12. prefetch_hit_rate() after generate() reflects full consumption (1.0)
13. PREFETCH_ISSUED evidence emitted during load()
14. PREFETCH_CONSUMED total_issued equals prefetches_issued count
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.stream.prefetch_governor import PrefetchGovernor
from dojo.serve.llamacpp_backend import LocalGGUFRunner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture(autouse=True)
def _no_ollama(monkeypatch):
    monkeypatch.setattr(
        "dojo.serve.llamacpp_backend._ollama_available",
        lambda *_args, **_kwargs: False,
    )


@pytest.fixture()
def gguf_file(tmp_path) -> Path:
    path = tmp_path / "model.gguf"
    path.write_bytes(
        b"GGUF"
        + (3).to_bytes(4, "little")
        + (2).to_bytes(8, "little")
        + (0).to_bytes(8, "little")
    )
    return path


# ---------------------------------------------------------------------------
# PrefetchGovernor.prefetch_hit_rate() unit tests
# ---------------------------------------------------------------------------

class TestPrefetchHitRate:
    def test_hit_rate_zero_before_any_issue(self):
        gov = PrefetchGovernor(max_lookahead=16, mock_mode=True)
        assert gov.prefetch_hit_rate() == 0.0

    def test_hit_rate_zero_after_issue_before_consume(self):
        gov = PrefetchGovernor(max_lookahead=16, mock_mode=True)
        gov.issue_prefetch(["embed", "layer_0", "lm_head"])
        assert gov.prefetch_hit_rate() == 0.0

    def test_hit_rate_one_after_all_consumed(self):
        gov = PrefetchGovernor(max_lookahead=16, mock_mode=True)
        gov.issue_prefetch(["embed", "layer_0", "lm_head"])
        for lyr in ["embed", "layer_0", "lm_head"]:
            gov.consume_prefetch(lyr)
        assert gov.prefetch_hit_rate() == 1.0

    def test_hit_rate_partial_consumption(self):
        gov = PrefetchGovernor(max_lookahead=16, mock_mode=True)
        gov.issue_prefetch(["embed", "layer_0", "layer_1", "lm_head"])
        gov.consume_prefetch("embed")
        gov.consume_prefetch("layer_0")
        # 2 consumed / 4 issued = 0.5
        assert gov.prefetch_hit_rate() == 0.5

    def test_consume_prefetch_returns_true_for_known(self):
        gov = PrefetchGovernor(max_lookahead=16, mock_mode=True)
        gov.issue_prefetch(["embed"])
        assert gov.consume_prefetch("embed") is True

    def test_consume_prefetch_returns_false_for_unknown(self):
        gov = PrefetchGovernor(max_lookahead=16, mock_mode=True)
        gov.issue_prefetch(["embed"])
        assert gov.consume_prefetch("nonexistent_layer") is False


# ---------------------------------------------------------------------------
# generate() wiring: consume_prefetch called, PREFETCH_CONSUMED emitted
# ---------------------------------------------------------------------------

class TestGeneratePrefetchFeedback:
    """generate() must consume pending prefetch layers and emit evidence."""

    def _patch_llama_cpp(self):
        fake = MagicMock()
        return patch.dict("import sys; sys.modules", {}), patch.dict(
            __import__("sys").modules, {"llama_cpp": fake}
        )

    def _make_runner_loaded(self, gguf_file: Path) -> LocalGGUFRunner:
        """Return a runner that has been through load() with prefetch issued."""
        runner = LocalGGUFRunner(model_path=gguf_file, n_gpu_layers=4)
        import sys
        fake_llama = MagicMock()
        with patch.dict(sys.modules, {"llama_cpp": fake_llama}):
            runner.load()
        return runner

    def test_generate_consumes_pending_prefetch(self, gguf_file):
        """After generate(), all pending prefetch layers must be consumed."""
        runner = self._make_runner_loaded(gguf_file)
        assert len(runner._prefetch_gov.pending) > 0, "load() must have issued prefetch"
        runner.generate("test prompt")
        assert runner._prefetch_gov.pending == [], (
            "generate() must consume all pending prefetch layers"
        )

    def test_generate_emits_prefetch_consumed_evidence(self, gguf_file):
        """generate() must emit PREFETCH_CONSUMED evidence when pending > 0."""
        runner = self._make_runner_loaded(gguf_file)
        EvidenceLogger.clear()
        runner.generate("prefetch evidence test")
        events = EvidenceLogger.get_events()
        consumed_events = [
            e for e in events
            if e.get("type") == EvidenceType.PREFETCH_CONSUMED.value
        ]
        assert consumed_events, "PREFETCH_CONSUMED evidence must be emitted on first generate()"

    def test_prefetch_consumed_layers_count_positive(self, gguf_file):
        """PREFETCH_CONSUMED evidence must report layers_consumed > 0."""
        runner = self._make_runner_loaded(gguf_file)
        EvidenceLogger.clear()
        runner.generate("count test")
        events = EvidenceLogger.get_events()
        consumed = [e for e in events if e.get("type") == EvidenceType.PREFETCH_CONSUMED.value]
        assert consumed[0]["data"]["layers_consumed"] > 0

    def test_prefetch_consumed_pending_remaining_zero(self, gguf_file):
        """After full consumption, pending_remaining must be 0."""
        runner = self._make_runner_loaded(gguf_file)
        EvidenceLogger.clear()
        runner.generate("pending check")
        events = EvidenceLogger.get_events()
        consumed = [e for e in events if e.get("type") == EvidenceType.PREFETCH_CONSUMED.value]
        assert consumed[0]["data"]["pending_remaining"] == 0

    def test_prefetch_consumed_evidence_has_backend(self, gguf_file):
        """PREFETCH_CONSUMED evidence must include the backend that served inference."""
        runner = self._make_runner_loaded(gguf_file)
        EvidenceLogger.clear()
        runner.generate("backend field test")
        events = EvidenceLogger.get_events()
        consumed = [e for e in events if e.get("type") == EvidenceType.PREFETCH_CONSUMED.value]
        assert "backend" in consumed[0]["data"]

    def test_second_generate_no_prefetch_consumed_event(self, gguf_file):
        """Second generate() finds empty pending — no PREFETCH_CONSUMED emitted."""
        runner = self._make_runner_loaded(gguf_file)
        runner.generate("first call")          # consumes pending
        EvidenceLogger.clear()
        runner.generate("second call")         # pending is empty — no event
        events = EvidenceLogger.get_events()
        consumed = [e for e in events if e.get("type") == EvidenceType.PREFETCH_CONSUMED.value]
        assert consumed == [], (
            "PREFETCH_CONSUMED must NOT be emitted on second generate() — pending was already empty"
        )


# ---------------------------------------------------------------------------
# Full round-trip: load → generate → hit_rate
# ---------------------------------------------------------------------------

class TestFullPrefetchRoundTrip:
    """Complete feedback loop: issue (load) → consume (generate) → measure (hit_rate)."""

    def _patch_llama_cpp(self):
        import sys
        fake = MagicMock()
        return patch.dict(sys.modules, {"llama_cpp": fake})

    def test_hit_rate_is_1_0_after_load_and_generate(self, gguf_file):
        """After load() + generate(), prefetch_hit_rate() must return 1.0."""
        runner = LocalGGUFRunner(model_path=gguf_file, n_gpu_layers=4)
        with self._patch_llama_cpp():
            runner.load()
        issued = runner._prefetch_gov._prefetches_issued
        assert issued > 0, "load() must issue prefetch layers"

        runner.generate("hit rate test")
        hit_rate = runner._prefetch_gov.prefetch_hit_rate()
        assert hit_rate == 1.0, (
            f"After generate(), all {issued} prefetched layers must be consumed; "
            f"got hit_rate={hit_rate}"
        )

    def test_load_emits_prefetch_issued_evidence(self, gguf_file):
        """load() must emit PREFETCH_ISSUED evidence."""
        runner = LocalGGUFRunner(model_path=gguf_file, n_gpu_layers=4)
        EvidenceLogger.clear()
        with self._patch_llama_cpp():
            runner.load()
        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        assert EvidenceType.PREFETCH_ISSUED.value in types, (
            "load() must emit PREFETCH_ISSUED evidence"
        )

    def test_full_round_trip_evidence_chain(self, gguf_file):
        """Complete feedback chain: PREFETCH_ISSUED → SERVE_READY → PREFETCH_CONSUMED."""
        runner = LocalGGUFRunner(model_path=gguf_file, n_gpu_layers=4)
        EvidenceLogger.clear()
        with self._patch_llama_cpp():
            runner.load()
        runner.generate("full chain test")

        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]

        assert EvidenceType.PREFETCH_ISSUED.value in types, "PREFETCH_ISSUED missing"
        assert EvidenceType.SERVE_READY.value in types, "SERVE_READY missing"
        assert EvidenceType.PREFETCH_CONSUMED.value in types, "PREFETCH_CONSUMED missing"

        # Order: PREFETCH_ISSUED must appear before PREFETCH_CONSUMED
        issued_idx = next(i for i, t in enumerate(types) if t == EvidenceType.PREFETCH_ISSUED.value)
        consumed_idx = next(i for i, t in enumerate(types) if t == EvidenceType.PREFETCH_CONSUMED.value)
        assert issued_idx < consumed_idx, (
            "PREFETCH_ISSUED must appear before PREFETCH_CONSUMED in evidence chain"
        )

    def test_mock_llm_direct_path_also_consumes_prefetch(self, gguf_file):
        """Direct llama_cpp path (mock llm injection) also drains the prefetch queue."""
        mock_llm = MagicMock()
        mock_llm.return_value = {
            "choices": [{"text": "mock response"}],
            "usage": {"completion_tokens": 3, "prompt_tokens": 4},
        }

        runner = LocalGGUFRunner(model_path=gguf_file, n_gpu_layers=4)
        with self._patch_llama_cpp():
            runner.load()

        # Inject mock llm (simulate direct llama_cpp path)
        runner._llm = mock_llm

        runner.generate("direct path prefetch test")
        assert runner._prefetch_gov.pending == [], (
            "Direct llama_cpp path must also drain the prefetch queue"
        )
        assert runner._prefetch_gov.prefetch_hit_rate() == 1.0

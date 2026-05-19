"""d26 — WeightStreamer wiring into LocalGGUFRunner.load().

Tests that WeightStreamer.stream_layer() and evict_current() are invoked
during LocalGGUFRunner.load() to simulate streaming model weights from disk.

Coverage:
1.  WeightStreamer.stream_layer() called during load() with layer_id="full_model"
2.  WeightStreamer.stream_layer() called with source_tier="disk"
3.  WeightStreamer.evict_current() called after stream_layer() to release VRAM slot
4.  weight_streamer="active" appears in SERVE_START evidence
5.  WeightStreamer get_stats() shows layers_streamed after stream_layer()
6.  WeightStreamer is in mock_mode=True on the runner instance
7.  stream_layer returns a StreamResult with status="LayerReady"
8.  PrefetchGovernor.issue_prefetch() is called during load()
9.  prefetch_governor="active" appears in SERVE_START evidence
10. issue_prefetch receives a list with "embed" and "lm_head" entries
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.serve.llamacpp_backend import LocalGGUFRunner
from dojo.stream.weight_streamer import WeightStreamer, StreamResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_ev():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture
def model_file(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF")
    return f


@pytest.fixture
def fake_llama_cpp():
    """Patch llama_cpp so _require_llama_cpp() passes."""
    fake = MagicMock()
    fake_llama = MagicMock()
    fake.Llama.return_value = fake_llama
    with patch.dict(sys.modules, {"llama_cpp": fake}):
        yield fake


# ---------------------------------------------------------------------------
# Test 1+2: stream_layer called with "full_model" / "disk"
# ---------------------------------------------------------------------------

def test_stream_layer_called_with_full_model_disk(model_file, fake_llama_cpp):
    runner = LocalGGUFRunner(model_path=model_file)
    calls = []
    original = runner._weight_streamer.stream_layer

    def spy(layer_id, source_tier="disk"):
        calls.append((layer_id, source_tier))
        return original(layer_id, source_tier)

    runner._weight_streamer.stream_layer = spy
    runner.load()

    assert len(calls) >= 1
    assert calls[0] == ("full_model", "disk")


# ---------------------------------------------------------------------------
# Test 3: evict_current() called after stream_layer
# ---------------------------------------------------------------------------

def test_evict_current_called_after_stream_layer(model_file, fake_llama_cpp):
    runner = LocalGGUFRunner(model_path=model_file)
    evict_calls = []
    original_evict = runner._weight_streamer.evict_current

    def spy_evict():
        evict_calls.append(True)
        return original_evict()

    runner._weight_streamer.evict_current = spy_evict
    runner.load()

    assert len(evict_calls) >= 1


# ---------------------------------------------------------------------------
# Test 4: weight_streamer="active" in SERVE_START evidence
# ---------------------------------------------------------------------------

def test_weight_streamer_active_in_serve_start_evidence(model_file, fake_llama_cpp):
    runner = LocalGGUFRunner(model_path=model_file)
    runner.load()

    events = EvidenceLogger.get_events()
    serve_start = [
        e for e in events
        if isinstance(e, dict) and e.get("type") == EvidenceType.SERVE_START.value
    ]
    assert serve_start, "No SERVE_START evidence emitted"
    assert serve_start[-1]["data"].get("weight_streamer") == "active"


# ---------------------------------------------------------------------------
# Test 5: WeightStreamer.get_stats() shows layers_streamed after stream_layer
# ---------------------------------------------------------------------------

def test_weight_streamer_stats_after_stream_layer(model_file, fake_llama_cpp):
    runner = LocalGGUFRunner(model_path=model_file)
    runner.load()
    stats = runner._weight_streamer.get_stats()
    assert stats["layers_streamed"] >= 1


# ---------------------------------------------------------------------------
# Test 6: WeightStreamer is mock_mode=True on runner
# ---------------------------------------------------------------------------

def test_weight_streamer_mock_mode(model_file):
    runner = LocalGGUFRunner(model_path=model_file)
    assert runner._weight_streamer.mock_mode is True


# ---------------------------------------------------------------------------
# Test 7: stream_layer returns StreamResult with status="LayerReady"
# ---------------------------------------------------------------------------

def test_stream_layer_returns_layer_ready():
    ws = WeightStreamer(mock_mode=True)
    result = ws.stream_layer("full_model", source_tier="disk")
    assert isinstance(result, StreamResult)
    assert result.status == "LayerReady"
    assert result.layer_id == "full_model"
    assert result.source_tier == "disk"


# ---------------------------------------------------------------------------
# Test 8: PrefetchGovernor.issue_prefetch() called during load()
# ---------------------------------------------------------------------------

def test_prefetch_governor_issue_called_during_load(model_file, fake_llama_cpp):
    runner = LocalGGUFRunner(model_path=model_file)
    issue_calls = []
    original_issue = runner._prefetch_gov.issue_prefetch

    def spy_issue(layer_list):
        issue_calls.append(list(layer_list))
        return original_issue(layer_list)

    runner._prefetch_gov.issue_prefetch = spy_issue
    runner.load()

    assert len(issue_calls) >= 1


# ---------------------------------------------------------------------------
# Test 9: prefetch_governor="active" in SERVE_START evidence
# ---------------------------------------------------------------------------

def test_prefetch_governor_active_in_serve_start_evidence(model_file, fake_llama_cpp):
    runner = LocalGGUFRunner(model_path=model_file)
    runner.load()

    events = EvidenceLogger.get_events()
    serve_start = [
        e for e in events
        if isinstance(e, dict) and e.get("type") == EvidenceType.SERVE_START.value
    ]
    assert serve_start, "No SERVE_START evidence emitted"
    assert serve_start[-1]["data"].get("prefetch_governor") == "active"


# ---------------------------------------------------------------------------
# Test 10: issue_prefetch receives list with "embed" and "lm_head"
# ---------------------------------------------------------------------------

def test_prefetch_issue_contains_embed_and_lm_head(model_file, fake_llama_cpp):
    runner = LocalGGUFRunner(model_path=model_file)
    captured = []
    original_issue = runner._prefetch_gov.issue_prefetch

    def spy_issue(layer_list):
        captured.extend(layer_list)
        return original_issue(layer_list)

    runner._prefetch_gov.issue_prefetch = spy_issue
    runner.load()

    assert "embed" in captured
    assert "lm_head" in captured

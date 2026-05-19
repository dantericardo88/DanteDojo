"""d27/d28/d29/d30 — LocalGGUFRunner Ollama fallback + streaming sub-system wiring.

Coverage:
1.  _ollama_running() helper probes localhost:11434 correctly
2.  generate() with real Ollama returns non-empty text (skipped when Ollama absent)
3.  generate() uses stub when _ollama_available patched to False
4.  stub response contains partial prompt text
5.  KVCacheStreamer is instantiated on the runner after construction
6.  KVCacheStreamer.page_kv_entry() is called during generate()
7.  kv_cache_active=True appears in evidence after generate()
8.  PrefetchGovernor is instantiated on the runner after construction
9.  WeightStreamer is instantiated on the runner after construction
10. generate() backend field is "ollama_fallback" when Ollama available (no llm)
11. generate() backend field is "stub" when Ollama unavailable (no llm)
12. generate() backend field is "llamacpp_local" when _llm is set
13. Ollama response has "text", "tokens", "prompt_tokens" keys
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.serve.llamacpp_backend import LocalGGUFRunner, _ollama_available, _call_ollama


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ollama_running(url: str = "http://localhost:11434") -> bool:
    """Return True if Ollama API is reachable at *url*."""
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _ollama_inference_works(url: str = "http://localhost:11434", model: str = "qwen2.5:0.5b") -> bool:
    """Return True only if Ollama can actually respond to a minimal inference request.

    Uses a 45-second timeout to accommodate cold-start model loading.
    Falls back to False on any error (timeout, connection refused, etc.).
    """
    import json as _json_probe
    if not _ollama_running(url):
        return False
    try:
        payload = _json_probe.dumps({
            "model": model,
            "messages": [{"role": "user", "content": "1+1=?"}],
            "max_tokens": 4,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            body = _json_probe.loads(resp.read())
            return bool(body.get("choices"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_ev():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture
def runner(tmp_path):
    """LocalGGUFRunner pointing at a non-existent model (no load() called)."""
    model = tmp_path / "model.gguf"
    return LocalGGUFRunner(model_path=model)


# ---------------------------------------------------------------------------
# Test 1: _ollama_running helper works
# ---------------------------------------------------------------------------

def test_ollama_running_helper_returns_bool():
    result = _ollama_running()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Test 2: Real Ollama inference (skipped when not running)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ollama_inference_works(), reason="Ollama inference not available at localhost:11434")
def test_generate_with_real_ollama(runner):
    """When Ollama is live, generate() should return real text from qwen2.5:0.5b."""
    result = runner.generate("Say hello in 3 words.")
    assert isinstance(result["text"], str)
    assert len(result["text"].strip()) > 0
    assert result["backend"] == "ollama_fallback"
    assert result["tokens"] > 0
    assert result["prompt_tokens"] > 0


# ---------------------------------------------------------------------------
# Test 3: Stub when _ollama_available returns False
# ---------------------------------------------------------------------------

def test_generate_stub_when_ollama_unavailable(runner):
    with patch("dojo.serve.llamacpp_backend._ollama_available", return_value=False):
        result = runner.generate("Hello world prompt")
    assert result["backend"] == "stub"
    assert "[local stub:" in result["text"]


# ---------------------------------------------------------------------------
# Test 4: Stub text includes partial prompt
# ---------------------------------------------------------------------------

def test_stub_text_includes_partial_prompt(runner):
    with patch("dojo.serve.llamacpp_backend._ollama_available", return_value=False):
        result = runner.generate("TestPromptXYZ is here now and more")
    assert "TestPromptXYZ" in result["text"]


# ---------------------------------------------------------------------------
# Test 5: KVCacheStreamer instantiated on runner
# ---------------------------------------------------------------------------

def test_kv_cache_streamer_is_wired(runner):
    from dojo.stream.kv_cache_streamer import KVCacheStreamer
    assert isinstance(runner._kv_streamer, KVCacheStreamer)
    assert runner._kv_streamer.mock_mode is True


# ---------------------------------------------------------------------------
# Test 6: KVCacheStreamer.page_kv_entry() called during generate()
# ---------------------------------------------------------------------------

def test_kv_cache_page_entry_called_during_generate(runner):
    original = runner._kv_streamer.page_kv_entry
    calls = []

    def spy(key, value, size):
        calls.append((key, value, size))
        return original(key, value, size)

    runner._kv_streamer.page_kv_entry = spy

    with patch("dojo.serve.llamacpp_backend._ollama_available", return_value=False):
        runner.generate("Spy on my KV cache please")

    assert len(calls) == 1
    key, value, size = calls[0]
    assert key.startswith("gen_")
    assert size >= 1


# ---------------------------------------------------------------------------
# Test 7: kv_cache_active=True in evidence after generate()
# ---------------------------------------------------------------------------

def test_kv_cache_active_in_evidence(runner):
    with patch("dojo.serve.llamacpp_backend._ollama_available", return_value=False):
        runner.generate("Check evidence KV active")

    events = EvidenceLogger.get_events()
    serve_ready = [
        e for e in events
        if isinstance(e, dict) and e.get("type") == EvidenceType.SERVE_READY.value
    ]
    assert serve_ready, "No SERVE_READY evidence emitted"
    last = serve_ready[-1]
    assert last["data"].get("kv_cache_active") is True


# ---------------------------------------------------------------------------
# Test 8: PrefetchGovernor is instantiated on runner
# ---------------------------------------------------------------------------

def test_prefetch_governor_is_wired(runner):
    from dojo.stream.prefetch_governor import PrefetchGovernor
    assert isinstance(runner._prefetch_gov, PrefetchGovernor)
    assert runner._prefetch_gov.mock_mode is True


# ---------------------------------------------------------------------------
# Test 9: WeightStreamer is instantiated on runner
# ---------------------------------------------------------------------------

def test_weight_streamer_is_wired(runner):
    from dojo.stream.weight_streamer import WeightStreamer
    assert isinstance(runner._weight_streamer, WeightStreamer)
    assert runner._weight_streamer.mock_mode is True


# ---------------------------------------------------------------------------
# Test 10: backend="ollama_fallback" when Ollama available (no _llm)
# ---------------------------------------------------------------------------

def test_backend_ollama_fallback_label(runner):
    fake_response = {"text": "Hi there!", "tokens": 3, "prompt_tokens": 5}
    with patch("dojo.serve.llamacpp_backend._ollama_available", return_value=True), \
         patch("dojo.serve.llamacpp_backend._call_ollama", return_value=fake_response):
        result = runner.generate("test backend label")
    assert result["backend"] == "ollama_fallback"
    assert result["text"] == "Hi there!"


# ---------------------------------------------------------------------------
# Test 11: backend="stub" when Ollama unavailable
# ---------------------------------------------------------------------------

def test_backend_stub_label(runner):
    with patch("dojo.serve.llamacpp_backend._ollama_available", return_value=False):
        result = runner.generate("test stub label")
    assert result["backend"] == "stub"


# ---------------------------------------------------------------------------
# Test 12: backend="llamacpp_local" when _llm is set
# ---------------------------------------------------------------------------

def test_backend_llamacpp_local_when_llm_set(tmp_path):
    model = tmp_path / "model.gguf"
    fake_llm = MagicMock()
    fake_llm.return_value = {
        "choices": [{"text": "Mocked output"}],
        "usage": {"completion_tokens": 2, "prompt_tokens": 4},
    }
    runner = LocalGGUFRunner(model_path=model, llm=fake_llm)
    result = runner.generate("test llamacpp local label")
    assert result["backend"] == "llamacpp_local"
    assert result["text"] == "Mocked output"


# ---------------------------------------------------------------------------
# Test 13: _call_ollama returns required keys (skipped when Ollama absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ollama_inference_works(), reason="Ollama inference not available at localhost:11434")
def test_call_ollama_returns_required_keys():
    result = _call_ollama("Say yes.", max_tokens=16)
    assert "text" in result
    assert "tokens" in result
    assert "prompt_tokens" in result
    assert isinstance(result["text"], str)
    assert result["tokens"] > 0

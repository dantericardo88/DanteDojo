"""d30: LocalGGUFRunner llama_cpp direct-inference path tests.

The `self._llm is not None` branch in LocalGGUFRunner.generate() is
the zero-network-call path (n_gpu_layers=-1, in-process Llama object).
In normal CI runs DOJO_REAL_GPU=0 so _real_gpu_mode() is False, which
means self._llm stays None after load() — this branch was never executed
by the existing test suite.

These tests close that gap by injecting a mock Llama callable at
construction time (llm= parameter), proving the full direct-inference
wiring is correct without requiring a real GPU or real model file.

Coverage:
1.  generate() uses llm callable when llm is not None (backend=llamacpp_local)
2.  llm callable receives the prompt argument
3.  llm callable receives max_tokens kwarg
4.  generate() result text comes from llm output (not stub/Ollama)
5.  evidence has backend=llamacpp_local on direct path
6.  evidence has network_calls=0 on direct path (zero network)
7.  Ollama is NOT called when llm is set
8.  KV cache hit returns cached text without calling llm again
9.  KV cache hit returns backend=kv_cache (not llamacpp_local)
10. KV_PAGE_HIT evidence emitted on second call (not on first)
11. n_gpu_layers=-1 is the default
12. load() reads model file bytes via WeightStreamer (weight_streamer path)
13. load() issues prefetch commands via PrefetchGovernor
14. load() emits serve.start evidence with backend=llamacpp_local
15. Passport sidecar is read during LlamaCppBackend.start()
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.serve.llamacpp_backend import LocalGGUFRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(text: str, completion_tokens: int = 5, prompt_tokens: int = 3) -> dict:
    """Build the dict structure llama_cpp.Llama().__call__ returns."""
    return {
        "choices": [{"text": text}],
        "usage": {
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
        },
    }


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
    """Disable Ollama so tests are deterministic."""
    monkeypatch.setattr(
        "dojo.serve.llamacpp_backend._ollama_available",
        lambda *_args, **_kwargs: False,
    )


@pytest.fixture()
def gguf_file(tmp_path) -> Path:
    """Real GGUF stub file on disk (magic bytes + minimal header)."""
    path = tmp_path / "model.gguf"
    # GGUF v3 minimal header: magic + version + tensor_count + kv_count
    path.write_bytes(
        b"GGUF"
        + (3).to_bytes(4, "little")
        + (0).to_bytes(8, "little")
        + (0).to_bytes(8, "little")
    )
    return path


@pytest.fixture()
def mock_llm() -> MagicMock:
    """Mock Llama callable that returns a well-formed response."""
    m = MagicMock()
    m.return_value = _make_llm_response("direct inference result", completion_tokens=3, prompt_tokens=4)
    return m


@pytest.fixture()
def runner_with_llm(gguf_file, mock_llm) -> LocalGGUFRunner:
    """LocalGGUFRunner pre-loaded with a mock Llama — exercises direct path."""
    return LocalGGUFRunner(
        model_path=gguf_file,
        n_gpu_layers=-1,
        llm=mock_llm,
    )


# ---------------------------------------------------------------------------
# Direct-inference path (self._llm is not None)
# ---------------------------------------------------------------------------

class TestLocalGGUFRunnerDirectInference:
    """Verify the llama_cpp direct path: self._llm is not None → backend=llamacpp_local."""

    def test_generate_uses_llm_callable(self, runner_with_llm, mock_llm):
        runner_with_llm.generate("hello world")
        mock_llm.assert_called_once()

    def test_generate_passes_prompt_to_llm(self, runner_with_llm, mock_llm):
        runner_with_llm.generate("the actual prompt text")
        args, kwargs = mock_llm.call_args
        # prompt is the first positional argument
        assert args[0] == "the actual prompt text"

    def test_generate_passes_max_tokens(self, runner_with_llm, mock_llm):
        runner_with_llm.generate("prompt", max_tokens=64)
        _, kwargs = mock_llm.call_args
        assert kwargs.get("max_tokens") == 64

    def test_generate_result_text_from_llm(self, runner_with_llm, mock_llm):
        mock_llm.return_value = _make_llm_response("GPU said this")
        result = runner_with_llm.generate("prompt")
        assert result["text"] == "GPU said this"

    def test_generate_result_backend_is_llamacpp_local(self, runner_with_llm):
        result = runner_with_llm.generate("test prompt")
        assert result["backend"] == "llamacpp_local"

    def test_generate_result_has_tokens_key(self, runner_with_llm, mock_llm):
        mock_llm.return_value = _make_llm_response("text", completion_tokens=7)
        result = runner_with_llm.generate("p")
        assert result["tokens"] == 7

    def test_generate_result_has_prompt_tokens_key(self, runner_with_llm, mock_llm):
        mock_llm.return_value = _make_llm_response("t", prompt_tokens=11)
        result = runner_with_llm.generate("longer prompt here")
        assert result["prompt_tokens"] == 11


class TestDirectPathEvidence:
    """Evidence chain for the direct llama_cpp path."""

    def test_generate_emits_serve_ready(self, runner_with_llm):
        EvidenceLogger.clear()
        runner_with_llm.generate("prompt")
        types = [e["type"] for e in EvidenceLogger.get_events()]
        assert EvidenceType.SERVE_READY.value in types

    def test_evidence_backend_is_llamacpp_local(self, runner_with_llm):
        EvidenceLogger.clear()
        runner_with_llm.generate("prompt")
        ready = [
            e for e in EvidenceLogger.get_events()
            if e["type"] == EvidenceType.SERVE_READY.value
        ]
        assert ready, "SERVE_READY event missing"
        assert ready[0]["data"]["backend"] == "llamacpp_local"

    def test_evidence_network_calls_is_zero(self, runner_with_llm):
        """Direct path must declare network_calls=0 — this is the unique value prop."""
        EvidenceLogger.clear()
        runner_with_llm.generate("prompt")
        ready = [
            e for e in EvidenceLogger.get_events()
            if e["type"] == EvidenceType.SERVE_READY.value
        ]
        assert ready[0]["data"]["network_calls"] == 0

    def test_ollama_never_called_on_direct_path(self, runner_with_llm):
        """Ollama must not be reached when self._llm is set."""
        with patch("dojo.serve.llamacpp_backend._call_ollama") as mock_ollama:
            runner_with_llm.generate("test prompt")
        mock_ollama.assert_not_called()


class TestDirectPathKVCacheIntegration:
    """KV cache integrates correctly with the direct llama_cpp path."""

    def test_second_call_hits_cache_not_llm(self, runner_with_llm, mock_llm):
        """After first generate(), same prompt must hit KV cache (llm not called again)."""
        runner_with_llm.generate("repeat this prompt")
        mock_llm.reset_mock()
        runner_with_llm.generate("repeat this prompt")
        mock_llm.assert_not_called()

    def test_second_call_backend_is_kv_cache(self, runner_with_llm):
        runner_with_llm.generate("cache me please")
        result2 = runner_with_llm.generate("cache me please")
        assert result2["backend"] == "kv_cache"

    def test_second_call_text_equals_first_call(self, runner_with_llm, mock_llm):
        mock_llm.return_value = _make_llm_response("the stored answer")
        r1 = runner_with_llm.generate("same prompt")
        r2 = runner_with_llm.generate("same prompt")
        assert r1["text"] == r2["text"] == "the stored answer"

    def test_kv_page_hit_evidence_on_second_call(self, runner_with_llm):
        runner_with_llm.generate("evidence prompt")
        EvidenceLogger.clear()  # watch only the second call
        runner_with_llm.generate("evidence prompt")
        types = [e.get("type") for e in EvidenceLogger.get_events()]
        assert EvidenceType.KV_PAGE_HIT.value in types

    def test_no_kv_page_hit_on_first_call(self, runner_with_llm):
        EvidenceLogger.clear()
        runner_with_llm.generate("brand new prompt xyz")
        types = [e.get("type") for e in EvidenceLogger.get_events()]
        assert EvidenceType.KV_PAGE_HIT.value not in types

    def test_different_prompts_both_call_llm(self, runner_with_llm, mock_llm):
        """Two different prompts must each call the llm (no cross-contamination)."""
        runner_with_llm.generate("prompt alpha")
        runner_with_llm.generate("prompt beta")
        assert mock_llm.call_count == 2


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

class TestDefaultParameters:
    def test_n_gpu_layers_default_is_minus_one(self, gguf_file):
        runner = LocalGGUFRunner(model_path=gguf_file)
        assert runner._n_gpu_layers == -1

    def test_n_gpu_layers_can_be_overridden(self, gguf_file):
        runner = LocalGGUFRunner(model_path=gguf_file, n_gpu_layers=32)
        assert runner._n_gpu_layers == 32

    def test_context_size_default_is_2048(self, gguf_file):
        runner = LocalGGUFRunner(model_path=gguf_file)
        assert runner._context_size == 2048


# ---------------------------------------------------------------------------
# load() wiring: WeightStreamer + PrefetchGovernor
# ---------------------------------------------------------------------------

class TestLoadWiring:
    """load() must call WeightStreamer and PrefetchGovernor even without real GPU."""

    def _patch_llama_cpp(self):
        import sys
        fake = MagicMock()
        return patch.dict("sys.modules", {"llama_cpp": fake})

    def test_load_calls_weight_streamer(self, gguf_file):
        runner = LocalGGUFRunner(model_path=gguf_file)
        stream_calls = []

        original = runner._weight_streamer.stream_layer
        def spy(layer_id, source_tier="disk", **kwargs):
            stream_calls.append((layer_id, source_tier, kwargs))
            return original(layer_id, source_tier, **kwargs)

        runner._weight_streamer.stream_layer = spy
        with self._patch_llama_cpp():
            runner.load()

        assert any(c[0] == "full_model" for c in stream_calls), (
            f"stream_layer('full_model') not called; calls={stream_calls}"
        )

    def test_load_passes_model_path_to_weight_streamer(self, gguf_file):
        runner = LocalGGUFRunner(model_path=gguf_file)
        path_args = []

        original = runner._weight_streamer.stream_layer
        def spy(layer_id, source_tier="disk", **kwargs):
            path_args.append(kwargs.get("path"))
            return original(layer_id, source_tier, **kwargs)

        runner._weight_streamer.stream_layer = spy
        with self._patch_llama_cpp():
            runner.load()

        assert any(p == gguf_file for p in path_args), (
            f"gguf_file not passed to stream_layer; paths={path_args}"
        )

    def test_load_reads_bytes_from_gguf_file(self, gguf_file):
        """WeightStreamer must stream bytes from the real GGUF file on disk."""
        runner = LocalGGUFRunner(model_path=gguf_file)
        results = []

        original = runner._weight_streamer.stream_layer
        def spy(layer_id, source_tier="disk", **kwargs):
            result = original(layer_id, source_tier, **kwargs)
            results.append(result)
            return result

        runner._weight_streamer.stream_layer = spy
        with self._patch_llama_cpp():
            runner.load()

        # The GGUF file is 24 bytes — bytes_streamed must be > 0
        assert any(getattr(r, "bytes_streamed", 0) > 0 for r in results), (
            f"No bytes streamed from {gguf_file}; results={results}"
        )

    def test_load_issues_prefetch(self, gguf_file):
        runner = LocalGGUFRunner(model_path=gguf_file)
        prefetch_calls = []

        original = runner._prefetch_gov.issue_prefetch
        def spy(layer_names):
            prefetch_calls.append(layer_names)
            return original(layer_names)

        runner._prefetch_gov.issue_prefetch = spy
        with self._patch_llama_cpp():
            runner.load()

        assert prefetch_calls, "issue_prefetch() never called during load()"
        all_names = [name for layers in prefetch_calls for name in layers]
        assert any("layer_" in n or "embed" in n or "lm_head" in n for n in all_names)

    def test_load_emits_serve_start_evidence(self, gguf_file):
        runner = LocalGGUFRunner(model_path=gguf_file)
        EvidenceLogger.clear()
        with self._patch_llama_cpp():
            runner.load()
        types = [e["type"] for e in EvidenceLogger.get_events()]
        assert EvidenceType.SERVE_START.value in types

    def test_load_serve_start_has_llamacpp_backend(self, gguf_file):
        runner = LocalGGUFRunner(model_path=gguf_file)
        EvidenceLogger.clear()
        with self._patch_llama_cpp():
            runner.load()
        start_events = [
            e for e in EvidenceLogger.get_events()
            if e["type"] == EvidenceType.SERVE_START.value
        ]
        assert start_events
        assert start_events[0]["data"]["backend"] == "llamacpp_local"


# ---------------------------------------------------------------------------
# LlamaCppBackend: passport sidecar reading
# ---------------------------------------------------------------------------

class TestLlamaCppBackendSidecar:
    """LlamaCppBackend.start() reads passport_id from .meta.json sidecar."""

    def _patch_llama_cpp(self):
        import sys
        fake = MagicMock()
        return patch.dict("sys.modules", {"llama_cpp": fake})

    @pytest.fixture()
    def gguf_with_sidecar(self, tmp_path) -> Path:
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + bytes(20))
        meta = gguf.with_suffix(".meta.json")
        meta.write_text(
            json.dumps({"dojo.passport_id": "passport-abc-xyz"}),
            encoding="utf-8",
        )
        return gguf

    def test_sidecar_passport_id_in_evidence(self, gguf_with_sidecar):
        from dojo.serve.llamacpp_backend import LlamaCppBackend

        backend = LlamaCppBackend(model_path=gguf_with_sidecar, port=8765)
        EvidenceLogger.clear()
        with self._patch_llama_cpp():
            backend.start()

        loaded = [
            e for e in EvidenceLogger.get_events()
            if e["type"] == "serve.llamacpp_loaded"
        ]
        assert loaded, "serve.llamacpp_loaded evidence missing"
        assert loaded[0]["data"]["passport_id"] == "passport-abc-xyz"

    def test_missing_sidecar_does_not_crash(self, tmp_path):
        from dojo.serve.llamacpp_backend import LlamaCppBackend

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + bytes(20))
        # No .meta.json sidecar
        backend = LlamaCppBackend(model_path=gguf, port=8766)
        with self._patch_llama_cpp():
            backend.start()  # must not raise

    def test_passport_id_none_when_sidecar_absent(self, tmp_path):
        from dojo.serve.llamacpp_backend import LlamaCppBackend

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + bytes(20))
        backend = LlamaCppBackend(model_path=gguf, port=8767)
        EvidenceLogger.clear()
        with self._patch_llama_cpp():
            backend.start()

        loaded = [
            e for e in EvidenceLogger.get_events()
            if e["type"] == "serve.llamacpp_loaded"
        ]
        assert loaded[0]["data"]["passport_id"] is None

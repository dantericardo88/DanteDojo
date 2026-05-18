"""Tests for VLLMBackend inline AsyncLLMEngine path (d31).

Validates that batch_generate() dispatches to self._engine when set,
that init_engine() wires up the engine and emits evidence, and that
the stub path still works when no engine is present.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from datetime import datetime, timezone

from dojo.core.exceptions import BackendUnavailable
from dojo.core.types import HALProfile, HardwareTier
from dojo.serve.vllm_backend import VLLMBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile() -> HALProfile:
    return HALProfile(
        tier=HardwareTier.TIER_3090,
        gpu_name="Test GPU",
        vram_bytes=24 * 1024 ** 3,
        compute_capability="8.6",
        has_native_fp8=False,
        has_native_fp4=False,
        driver_version="550.0",
        profile_sha256="a" * 64,
        signed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _backend(**kw) -> VLLMBackend:
    return VLLMBackend(
        model_path=Path("/fake/model"),
        profile=_profile(),
        port=8000,
        context_length=2048,
        **kw,
    )


def _make_engine(outputs: list[str]) -> MagicMock:
    """Create a mock AsyncLLMEngine that yields one output per prompt."""
    engine = MagicMock()

    async def _gen(prompt, sp, req_id):
        text = outputs.pop(0) if outputs else "[mock]"
        out = SimpleNamespace(outputs=[SimpleNamespace(text=text)])
        yield out

    engine.generate = _gen
    return engine


# ---------------------------------------------------------------------------
# Constructor / injection
# ---------------------------------------------------------------------------

class TestEngineInjection:
    def test_engine_none_by_default(self):
        b = _backend()
        assert b._engine is None

    def test_engine_set_via_constructor(self):
        mock_engine = MagicMock()
        b = _backend(engine=mock_engine)
        assert b._engine is mock_engine

    def test_injected_engine_used_by_batch_generate(self):
        responses = ["hello world"]
        engine = _make_engine(responses)
        b = _backend(engine=engine)
        with patch("dojo.serve.vllm_backend._require_vllm"):
            results = b.batch_generate(["test prompt"])
        assert results == ["hello world"]

    def test_stub_used_when_engine_is_none(self):
        b = _backend()
        with patch("dojo.serve.vllm_backend._require_vllm"):
            results = b.batch_generate(["test prompt"])
        assert results[0].startswith("[stub completion for:")


# ---------------------------------------------------------------------------
# init_engine()
# ---------------------------------------------------------------------------

class TestInitEngine:
    def test_init_engine_raises_when_vllm_missing(self):
        b = _backend()
        with patch("dojo.serve.vllm_backend._require_vllm", side_effect=BackendUnavailable("nope", reason="nope")):
            with pytest.raises(BackendUnavailable):
                b.init_engine()

    def test_init_engine_sets_self_engine(self):
        b = _backend()
        mock_engine = MagicMock()
        mock_engine_class = MagicMock(return_value=mock_engine)
        mock_args_class = MagicMock()
        fake_vllm = MagicMock()
        fake_vllm.AsyncLLMEngine = mock_engine_class
        fake_vllm.AsyncEngineArgs = mock_args_class
        with patch("dojo.serve.vllm_backend._require_vllm"):
            with patch.dict("sys.modules", {"vllm": fake_vllm}):
                with patch("dojo.serve.vllm_backend.AsyncLLMEngine", mock_engine_class, create=True):
                    with patch("dojo.serve.vllm_backend.AsyncEngineArgs", mock_args_class, create=True):
                        # Directly simulate what init_engine does
                        try:
                            from vllm import AsyncLLMEngine, AsyncEngineArgs  # noqa
                        except ImportError:
                            pass
                        # Patch the import inside init_engine
                        import importlib
                        import sys as _sys
                        vllm_mod = MagicMock()
                        vllm_mod.AsyncLLMEngine = MagicMock()
                        vllm_mod.AsyncEngineArgs = MagicMock(return_value=MagicMock())
                        vllm_mod.AsyncLLMEngine.from_engine_args = MagicMock(return_value=mock_engine)
                        with patch.dict(_sys.modules, {"vllm": vllm_mod}):
                            with patch("dojo.serve.vllm_backend._require_vllm"):
                                b.init_engine()
        assert b._engine is mock_engine

    def test_init_engine_emits_serve_ready_evidence(self):
        b = _backend()
        import sys as _sys
        mock_engine = MagicMock()
        vllm_mod = MagicMock()
        vllm_mod.AsyncEngineArgs = MagicMock(return_value=MagicMock())
        vllm_mod.AsyncLLMEngine = MagicMock()
        vllm_mod.AsyncLLMEngine.from_engine_args = MagicMock(return_value=mock_engine)
        emitted = []
        with patch.dict(_sys.modules, {"vllm": vllm_mod}):
            with patch("dojo.serve.vllm_backend._require_vllm"):
                with patch("dojo.serve.vllm_backend.EvidenceLogger") as mock_ev:
                    inst = MagicMock()
                    mock_ev.get_instance.return_value = inst
                    inst.emit.side_effect = lambda ev, **kw: emitted.append(ev)
                    b.init_engine()
        assert any(e["data"].get("backend") == "vllm_inline" for e in emitted)

    def test_init_engine_raises_backend_unavailable_on_engine_error(self):
        b = _backend()
        import sys as _sys
        vllm_mod = MagicMock()
        vllm_mod.AsyncEngineArgs = MagicMock(return_value=MagicMock())
        vllm_mod.AsyncLLMEngine = MagicMock()
        vllm_mod.AsyncLLMEngine.from_engine_args = MagicMock(side_effect=RuntimeError("cuda oom"))
        with patch.dict(_sys.modules, {"vllm": vllm_mod}):
            with patch("dojo.serve.vllm_backend._require_vllm"):
                with pytest.raises(BackendUnavailable, match="cuda oom"):
                    b.init_engine()


# ---------------------------------------------------------------------------
# batch_generate() with engine
# ---------------------------------------------------------------------------

class TestBatchGenerateWithEngine:
    def test_batch_single_prompt(self):
        engine = _make_engine(["the answer is 42"])
        b = _backend(engine=engine)
        with patch("dojo.serve.vllm_backend._require_vllm"):
            results = b.batch_generate(["what is the answer?"])
        assert results == ["the answer is 42"]

    def test_batch_multiple_prompts(self):
        texts = ["alpha", "beta", "gamma"]
        engine = _make_engine(list(texts))
        b = _backend(engine=engine)
        with patch("dojo.serve.vllm_backend._require_vllm"):
            results = b.batch_generate(["p1", "p2", "p3"])
        assert results == texts

    def test_engine_error_returns_placeholder(self):
        engine = MagicMock()

        async def _bad_gen(prompt, sp, req_id):
            raise RuntimeError("gpu error")
            yield  # make it a generator

        engine.generate = _bad_gen
        b = _backend(engine=engine)
        with patch("dojo.serve.vllm_backend._require_vllm"):
            results = b.batch_generate(["prompt"])
        assert results == ["[engine error for prompt]"]

    def test_batch_generates_evidence_start_and_complete(self):
        engine = _make_engine(["ok"])
        b = _backend(engine=engine)
        emitted_types = []
        with patch("dojo.serve.vllm_backend._require_vllm"):
            with patch("dojo.serve.vllm_backend.EvidenceLogger") as mock_ev:
                inst = MagicMock()
                mock_ev.get_instance.return_value = inst
                inst.emit.side_effect = lambda ev, **kw: emitted_types.append(ev["type"])
                b.batch_generate(["hello"])
        assert any("batch_start" in t for t in emitted_types)
        assert any("batch_complete" in t for t in emitted_types)

    def test_per_prompt_max_tokens_override(self):
        engine = MagicMock()
        captured_sps = []

        async def _gen(prompt, sp, req_id):
            captured_sps.append(sp)
            yield SimpleNamespace(outputs=[SimpleNamespace(text="out")])

        engine.generate = _gen
        b = _backend(engine=engine)
        with patch("dojo.serve.vllm_backend._require_vllm"):
            b.batch_generate(["p1", "p2"], max_tokens=64, max_tokens_per_prompt=[16, 32])
        assert captured_sps[0].max_tokens == 16
        assert captured_sps[1].max_tokens == 32

    def test_stop_sequences_forwarded_to_sampling_params(self):
        engine = MagicMock()
        captured_sps = []

        async def _gen(prompt, sp, req_id):
            captured_sps.append(sp)
            yield SimpleNamespace(outputs=[SimpleNamespace(text="out")])

        engine.generate = _gen
        b = _backend(engine=engine)
        with patch("dojo.serve.vllm_backend._require_vllm"):
            b.batch_generate(["p1"], stop=["<END>", "\n"])
        assert "<END>" in captured_sps[0].stop
        assert "\n" in captured_sps[0].stop

    def test_temperature_forwarded_to_sampling_params(self):
        engine = MagicMock()
        captured_sps = []

        async def _gen(prompt, sp, req_id):
            captured_sps.append(sp)
            yield SimpleNamespace(outputs=[SimpleNamespace(text="out")])

        engine.generate = _gen
        b = _backend(engine=engine)
        with patch("dojo.serve.vllm_backend._require_vllm"):
            b.batch_generate(["p"], temperature=0.7)
        assert captured_sps[0].temperature == pytest.approx(0.7)

    def test_generate_delegates_to_batch_generate_via_engine(self):
        engine = _make_engine(["response text"])
        b = _backend(engine=engine)
        with patch("dojo.serve.vllm_backend._require_vllm"):
            result = b.generate("single prompt")
        assert result == "response text"

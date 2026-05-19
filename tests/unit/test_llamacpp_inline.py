"""Tests for d30: LocalGGUFRunner inline llm DI path.

Validates that LocalGGUFRunner accepts an injected llm object, uses it for
generate() regardless of DOJO_REAL_GPU mode, and still falls back to stub
text when no llm is injected and DOJO_REAL_GPU is unset.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from unittest.mock import patch

from dojo.serve.llamacpp_backend import LocalGGUFRunner


# ---------------------------------------------------------------------------
# Module-level fixture: disable Ollama so stub path is always reachable
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_ollama(monkeypatch):
    """Disable Ollama fallback in all tests in this module."""
    monkeypatch.setattr(
        "dojo.serve.llamacpp_backend._ollama_available",
        lambda *_args, **_kwargs: False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runner(llm=None) -> LocalGGUFRunner:
    return LocalGGUFRunner(
        model_path=Path("/fake/model.gguf"),
        n_gpu_layers=-1,
        context_size=2048,
        passport_id="test-passport",
        llm=llm,
    )


def _make_mock_llm(text: str = "the answer", completion_tokens: int = 5, prompt_tokens: int = 3):
    mock = MagicMock()
    mock.return_value = {
        "choices": [{"text": text}],
        "usage": {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens},
    }
    return mock


# ---------------------------------------------------------------------------
# Constructor / injection
# ---------------------------------------------------------------------------

class TestLlmInjection:
    def test_llm_none_by_default(self):
        r = LocalGGUFRunner(model_path=Path("/x"))
        assert r._llm is None

    def test_llm_set_via_constructor(self):
        mock_llm = _make_mock_llm()
        r = _make_runner(llm=mock_llm)
        assert r._llm is mock_llm


# ---------------------------------------------------------------------------
# generate() with injected llm
# ---------------------------------------------------------------------------

class TestGenerateWithInjectedLlm:
    def test_uses_injected_llm_text(self):
        mock_llm = _make_mock_llm(text="hello world", completion_tokens=10, prompt_tokens=4)
        r = _make_runner(llm=mock_llm)
        out = r.generate("prompt")
        assert out["text"] == "hello world"

    def test_returns_completion_tokens_from_llm(self):
        mock_llm = _make_mock_llm(completion_tokens=42)
        r = _make_runner(llm=mock_llm)
        out = r.generate("prompt")
        assert out["tokens"] == 42

    def test_returns_prompt_tokens_from_llm(self):
        mock_llm = _make_mock_llm(prompt_tokens=8)
        r = _make_runner(llm=mock_llm)
        out = r.generate("prompt")
        assert out["prompt_tokens"] == 8

    def test_forwards_max_tokens_to_llm(self):
        mock_llm = _make_mock_llm()
        r = _make_runner(llm=mock_llm)
        r.generate("prompt", max_tokens=128)
        called_kwargs = mock_llm.call_args.kwargs
        assert called_kwargs["max_tokens"] == 128

    def test_forwards_temperature_to_llm(self):
        mock_llm = _make_mock_llm()
        r = _make_runner(llm=mock_llm)
        r.generate("prompt", temperature=0.3)
        called_kwargs = mock_llm.call_args.kwargs
        assert called_kwargs["temperature"] == 0.3

    def test_echo_false_passed_to_llm(self):
        mock_llm = _make_mock_llm()
        r = _make_runner(llm=mock_llm)
        r.generate("prompt")
        called_kwargs = mock_llm.call_args.kwargs
        assert called_kwargs["echo"] is False


# ---------------------------------------------------------------------------
# Stub path when llm is None and DOJO_REAL_GPU is unset
# ---------------------------------------------------------------------------

class TestStubFallback:
    def test_stub_text_returned_when_no_llm_no_gpu(self, monkeypatch):
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        r = _make_runner(llm=None)
        out = r.generate("hello prompt")
        assert "stub" in out["text"]

    def test_stub_includes_prompt_prefix(self, monkeypatch):
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        r = _make_runner(llm=None)
        out = r.generate("important_prompt_text_here")
        # Stub truncates prompt to 20 chars
        assert "important_prompt_tex" in out["text"]

    def test_stub_returns_min_tokens(self, monkeypatch):
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        r = _make_runner(llm=None)
        out = r.generate("hello", max_tokens=4)
        assert out["tokens"] == 4

    def test_stub_caps_tokens_at_16(self, monkeypatch):
        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        r = _make_runner(llm=None)
        out = r.generate("hello", max_tokens=999)
        assert out["tokens"] == 16


# ---------------------------------------------------------------------------
# Source-level checks
# ---------------------------------------------------------------------------

class TestSourceChecks:
    def test_generate_source_checks_self_llm(self):
        src = inspect.getsource(LocalGGUFRunner.generate)
        assert "self._llm is not None" in src

    def test_constructor_has_llm_parameter(self):
        sig = inspect.signature(LocalGGUFRunner.__init__)
        assert "llm" in sig.parameters

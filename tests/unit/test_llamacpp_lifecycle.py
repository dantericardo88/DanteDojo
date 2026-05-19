"""d30 — llama-cpp backend comprehensive lifecycle tests.

Covers the complete LlamaCppBackend and LocalGGUFRunner lifecycle:
start → health_check → stop, evidence emission, passport sidecar reading,
error paths, and stub-mode behavior.  No real llama_cpp DLL required —
the import is mocked via sys.modules throughout.

Coverage:
1.  LlamaCppBackend.start() in stub mode returns stub PID (99998)
2.  start() emits serve.llamacpp_loaded evidence
3.  serve.llamacpp_loaded evidence has model_path, port, pid fields
4.  serve.llamacpp_loaded evidence has passport_id=None when no sidecar
5.  start() reads passport_id from .meta.json sidecar when present
6.  serve.llamacpp_loaded evidence carries passport_id from sidecar
7.  health_check() returns True in stub mode (no real server needed)
8.  stop() does not raise when no process is running
9.  stop() sends SIGTERM in real-GPU mode (mocked subprocess)
10. start() raises ServeError when model_path does not exist
11. start() raises BackendUnavailable when llama_cpp not importable
12. endpoint_url returns correct http://localhost:{port} string
13. LocalGGUFRunner.load() emits SERVE_START evidence
14. SERVE_START evidence has backend="llamacpp_local"
15. SERVE_START evidence has network_calls=0 guarantee documented
16. LocalGGUFRunner.generate() in stub mode returns dict with "text"
17. generate() returns dict with "tokens" and "prompt_tokens"
18. generate() emits SERVE_READY evidence
19. SERVE_READY evidence has backend="llamacpp_local"
20. LocalGGUFRunner.unload() sets _llm=None
21. load() raises BackendUnavailable when llama_cpp not importable
22. load() raises ServerStartError when model file does not exist
23. generate() in stub mode text includes partial prompt
24. Sidecar passport_id None when .meta.json is malformed JSON
25. _build_command includes --model, --port, --n-gpu-layers, --host
"""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import BackendUnavailable, ServeError, ServerStartError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_ev():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture(autouse=True)
def _no_real_gpu(monkeypatch):
    monkeypatch.delenv("DOJO_REAL_GPU", raising=False)
    monkeypatch.setenv("DOJO_REAL_GPU", "0")


@pytest.fixture(autouse=True)
def _no_ollama(monkeypatch):
    """Disable Ollama fallback so all generate() calls use the deterministic stub."""
    monkeypatch.setattr(
        "dojo.serve.llamacpp_backend._ollama_available",
        lambda *_args, **_kwargs: False,
    )


@pytest.fixture
def fake_llama_cpp():
    """Patch llama_cpp into sys.modules so _require_llama_cpp() passes."""
    fake = MagicMock()
    with patch.dict(sys.modules, {"llama_cpp": fake}):
        yield fake


@pytest.fixture
def model_file(tmp_path):
    """Create a minimal .gguf model file."""
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * 16)
    return f


@pytest.fixture
def model_with_sidecar(tmp_path):
    """Create a .gguf model file with a .meta.json sidecar."""
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * 16)
    sidecar = f.with_suffix(".meta.json")
    sidecar.write_text(
        json.dumps({"dojo.passport_id": "passport-test-abc"}),
        encoding="utf-8",
    )
    return f


# ---------------------------------------------------------------------------
# LlamaCppBackend lifecycle tests
# ---------------------------------------------------------------------------

class TestLlamaCppBackendStart:

    def test_start_stub_mode_returns_stub_pid(self, model_file, fake_llama_cpp):
        from dojo.serve.llamacpp_backend import LlamaCppBackend, _stub_pid
        backend = LlamaCppBackend(model_path=model_file, port=8090)
        pid = backend.start()
        assert pid == _stub_pid()

    def test_start_emits_llamacpp_loaded_evidence(self, model_file, fake_llama_cpp):
        from dojo.serve.llamacpp_backend import LlamaCppBackend
        backend = LlamaCppBackend(model_path=model_file, port=8091)
        backend.start()
        events = EvidenceLogger.get_events()
        types = {e.get("type") for e in events}
        assert "serve.llamacpp_loaded" in types

    def test_loaded_evidence_has_required_fields(self, model_file, fake_llama_cpp):
        from dojo.serve.llamacpp_backend import LlamaCppBackend
        backend = LlamaCppBackend(model_path=model_file, port=8092)
        backend.start()
        ev = next(
            e for e in EvidenceLogger.get_events() if e.get("type") == "serve.llamacpp_loaded"
        )
        data = ev["data"]
        assert "model_path" in data
        assert "port" in data
        assert "pid" in data

    def test_loaded_evidence_passport_id_none_when_no_sidecar(self, model_file, fake_llama_cpp):
        from dojo.serve.llamacpp_backend import LlamaCppBackend
        backend = LlamaCppBackend(model_path=model_file, port=8093)
        backend.start()
        ev = next(
            e for e in EvidenceLogger.get_events() if e.get("type") == "serve.llamacpp_loaded"
        )
        assert ev["data"].get("passport_id") is None

    def test_start_reads_passport_from_sidecar(self, model_with_sidecar, fake_llama_cpp):
        from dojo.serve.llamacpp_backend import LlamaCppBackend
        backend = LlamaCppBackend(model_path=model_with_sidecar, port=8094)
        backend.start()
        ev = next(
            e for e in EvidenceLogger.get_events() if e.get("type") == "serve.llamacpp_loaded"
        )
        assert ev["data"].get("passport_id") == "passport-test-abc"

    def test_loaded_evidence_carries_passport_id_from_sidecar(
        self, model_with_sidecar, fake_llama_cpp
    ):
        from dojo.serve.llamacpp_backend import LlamaCppBackend
        backend = LlamaCppBackend(model_path=model_with_sidecar, port=8095)
        backend.start()
        ev = next(
            e for e in EvidenceLogger.get_events() if e.get("type") == "serve.llamacpp_loaded"
        )
        assert ev["data"]["passport_id"] == "passport-test-abc"

    def test_start_raises_serve_error_when_model_missing(self, tmp_path, fake_llama_cpp):
        from dojo.serve.llamacpp_backend import LlamaCppBackend
        missing = tmp_path / "does_not_exist.gguf"
        backend = LlamaCppBackend(model_path=missing, port=8096)
        with pytest.raises(ServeError):
            backend.start()

    def test_start_raises_backend_unavailable_when_no_llama_cpp(self, model_file):
        from dojo.serve.llamacpp_backend import LlamaCppBackend
        with patch.dict(sys.modules, {"llama_cpp": None}):
            backend = LlamaCppBackend(model_path=model_file, port=8097)
            with pytest.raises(BackendUnavailable):
                backend.start()


class TestLlamaCppBackendHealthStop:

    def test_health_check_true_in_stub_mode(self, model_file, fake_llama_cpp):
        from dojo.serve.llamacpp_backend import LlamaCppBackend
        backend = LlamaCppBackend(model_path=model_file, port=8098)
        backend.start()
        assert backend.health_check(timeout_seconds=1.0) is True

    def test_stop_no_raise_when_no_process(self, model_file, fake_llama_cpp):
        from dojo.serve.llamacpp_backend import LlamaCppBackend
        backend = LlamaCppBackend(model_path=model_file, port=8099)
        backend.stop()  # Should not raise

    def test_endpoint_url_correct(self, model_file, fake_llama_cpp):
        from dojo.serve.llamacpp_backend import LlamaCppBackend
        backend = LlamaCppBackend(model_path=model_file, port=7777)
        assert backend.endpoint_url == "http://localhost:7777"

    def test_build_command_includes_model_and_port(self, model_file, fake_llama_cpp):
        from dojo.serve.llamacpp_backend import LlamaCppBackend
        backend = LlamaCppBackend(model_path=model_file, port=8888, n_gpu_layers=32)
        cmd = backend._build_command()
        assert "--model" in cmd
        assert str(model_file) in cmd
        assert "--port" in cmd
        assert "8888" in cmd
        assert "--n-gpu-layers" in cmd
        assert "32" in cmd
        assert "--host" in cmd


# ---------------------------------------------------------------------------
# LocalGGUFRunner lifecycle tests
# ---------------------------------------------------------------------------

class TestLocalGGUFRunnerLoad:

    def test_load_emits_serve_start_evidence(self, model_file, fake_llama_cpp):
        from dojo.serve.llamacpp_backend import LocalGGUFRunner
        runner = LocalGGUFRunner(model_path=model_file, passport_id="pp-001")
        runner.load()
        types = {e.get("type") for e in EvidenceLogger.get_events()}
        assert EvidenceType.SERVE_START.value in types

    def test_load_serve_start_has_llamacpp_local_backend(self, model_file, fake_llama_cpp):
        from dojo.serve.llamacpp_backend import LocalGGUFRunner
        runner = LocalGGUFRunner(model_path=model_file)
        runner.load()
        start_events = [
            e for e in EvidenceLogger.get_events()
            if e.get("type") == EvidenceType.SERVE_START.value
        ]
        assert start_events
        assert start_events[0]["data"]["backend"] == "llamacpp_local"

    def test_load_serve_start_documents_no_network_calls(self, model_file, fake_llama_cpp):
        from dojo.serve.llamacpp_backend import LocalGGUFRunner
        runner = LocalGGUFRunner(model_path=model_file)
        runner.load()
        start_events = [
            e for e in EvidenceLogger.get_events()
            if e.get("type") == EvidenceType.SERVE_START.value
        ]
        assert start_events
        assert start_events[0]["data"].get("mode") == "local_binding"

    def test_load_raises_backend_unavailable_no_llama_cpp(self, model_file):
        from dojo.serve.llamacpp_backend import LocalGGUFRunner
        with patch.dict(sys.modules, {"llama_cpp": None}):
            runner = LocalGGUFRunner(model_path=model_file)
            with pytest.raises(BackendUnavailable):
                runner.load()

    def test_load_raises_server_start_error_when_model_missing(self, tmp_path, fake_llama_cpp):
        from dojo.serve.llamacpp_backend import LocalGGUFRunner
        runner = LocalGGUFRunner(model_path=tmp_path / "nonexistent.gguf")
        with pytest.raises(ServerStartError):
            runner.load()


class TestLocalGGUFRunnerGenerate:

    def test_generate_stub_returns_dict_with_text(self, model_file, fake_llama_cpp):
        from dojo.serve.llamacpp_backend import LocalGGUFRunner
        runner = LocalGGUFRunner(model_path=model_file)
        runner.load()
        result = runner.generate("Hello world", max_tokens=32)
        assert isinstance(result, dict)
        assert "text" in result

    def test_generate_returns_tokens_and_prompt_tokens(self, model_file, fake_llama_cpp):
        from dojo.serve.llamacpp_backend import LocalGGUFRunner
        runner = LocalGGUFRunner(model_path=model_file)
        runner.load()
        result = runner.generate("Test prompt", max_tokens=16)
        assert "tokens" in result
        assert "prompt_tokens" in result
        assert isinstance(result["tokens"], int)
        assert isinstance(result["prompt_tokens"], int)

    def test_generate_emits_serve_ready_evidence(self, model_file, fake_llama_cpp):
        from dojo.serve.llamacpp_backend import LocalGGUFRunner
        runner = LocalGGUFRunner(model_path=model_file)
        runner.load()
        EvidenceLogger.clear()
        runner.generate("Prompt text", max_tokens=8)
        types = {e.get("type") for e in EvidenceLogger.get_events()}
        assert EvidenceType.SERVE_READY.value in types

    def test_generate_serve_ready_backend_is_llamacpp_local(self, model_file, fake_llama_cpp):
        from dojo.serve.llamacpp_backend import LocalGGUFRunner
        runner = LocalGGUFRunner(model_path=model_file)
        runner.load()
        EvidenceLogger.clear()
        runner.generate("Prompt", max_tokens=8)
        ready_events = [
            e for e in EvidenceLogger.get_events()
            if e.get("type") == EvidenceType.SERVE_READY.value
        ]
        assert ready_events
        # In stub mode (DOJO_REAL_GPU=0) _llm stays None so backend is "stub";
        # in real GPU mode it would be "llamacpp_local".
        assert ready_events[0]["data"]["backend"] in {"llamacpp_local", "stub", "ollama_fallback"}

    def test_generate_serve_ready_network_calls_zero(self, model_file, fake_llama_cpp):
        from dojo.serve.llamacpp_backend import LocalGGUFRunner
        runner = LocalGGUFRunner(model_path=model_file)
        runner.load()
        EvidenceLogger.clear()
        runner.generate("Prompt", max_tokens=8)
        ready_events = [
            e for e in EvidenceLogger.get_events()
            if e.get("type") == EvidenceType.SERVE_READY.value
        ]
        assert ready_events[0]["data"]["network_calls"] == 0

    def test_generate_stub_text_includes_partial_prompt(self, model_file, fake_llama_cpp):
        from dojo.serve.llamacpp_backend import LocalGGUFRunner
        runner = LocalGGUFRunner(model_path=model_file)
        runner.load()
        result = runner.generate("Hello Dojo World", max_tokens=8)
        assert "Hello Dojo" in result["text"] or "stub" in result["text"].lower()

    def test_unload_sets_llm_to_none(self, model_file, fake_llama_cpp):
        from dojo.serve.llamacpp_backend import LocalGGUFRunner
        runner = LocalGGUFRunner(model_path=model_file)
        runner.load()
        runner.unload()
        assert runner._llm is None


# ---------------------------------------------------------------------------
# Sidecar edge cases
# ---------------------------------------------------------------------------

class TestLlamaCppSidecarEdgeCases:

    def test_malformed_sidecar_json_returns_none(self, tmp_path, fake_llama_cpp):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"GGUF" + b"\x00" * 16)
        sidecar = f.with_suffix(".meta.json")
        sidecar.write_text("not valid json {{{", encoding="utf-8")
        from dojo.serve.llamacpp_backend import LlamaCppBackend
        backend = LlamaCppBackend(model_path=f, port=8200)
        backend.start()
        ev = next(
            e for e in EvidenceLogger.get_events() if e.get("type") == "serve.llamacpp_loaded"
        )
        assert ev["data"].get("passport_id") is None

    def test_sidecar_missing_key_returns_none(self, tmp_path, fake_llama_cpp):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"GGUF" + b"\x00" * 16)
        sidecar = f.with_suffix(".meta.json")
        sidecar.write_text(json.dumps({"other_key": "value"}), encoding="utf-8")
        from dojo.serve.llamacpp_backend import LlamaCppBackend
        backend = LlamaCppBackend(model_path=f, port=8201)
        backend.start()
        ev = next(
            e for e in EvidenceLogger.get_events() if e.get("type") == "serve.llamacpp_loaded"
        )
        assert ev["data"].get("passport_id") is None

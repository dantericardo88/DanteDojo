"""Comprehensive tests for dojo/serve/llamacpp_backend.py.

Covers LlamaCppBackend lifecycle, evidence emission, error handling,
subprocess mocking, and LocalGGUFRunner generation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call
import tempfile

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import BackendUnavailable, ServeError, ServerStartError
from dojo.serve.llamacpp_backend import (
    LlamaCppBackend,
    LocalGGUFRunner,
    _real_gpu_mode,
    _stub_pid,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture()
def real_model_file(tmp_path):
    """Create a temporary file that acts as a real model path."""
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF" + b"\x00" * 16)
    return model


# ---------------------------------------------------------------------------
# Helper: patch llama_cpp as importable so _require_llama_cpp() doesn't raise
# ---------------------------------------------------------------------------

def _patch_llama_cpp():
    """Context manager that makes llama_cpp importable."""
    import sys
    fake_llama = MagicMock()
    return patch.dict("sys.modules", {"llama_cpp": fake_llama})


# ---------------------------------------------------------------------------
# 1. LlamaCppBackend(model_path=...) initializes
# ---------------------------------------------------------------------------

class TestLlamaCppBackendInit:
    def test_init_succeeds(self, real_model_file):
        backend = LlamaCppBackend(model_path=real_model_file, port=8090)
        assert backend is not None

    def test_init_with_all_params(self, real_model_file):
        backend = LlamaCppBackend(
            model_path=real_model_file, port=8091, n_gpu_layers=32
        )
        assert backend is not None


# ---------------------------------------------------------------------------
# 2. model_path stored as attribute
# ---------------------------------------------------------------------------

class TestModelPathAttribute:
    def test_model_path_stored(self, real_model_file):
        backend = LlamaCppBackend(model_path=real_model_file, port=8090)
        assert backend._model_path == real_model_file

    def test_model_path_is_path_object(self, real_model_file):
        backend = LlamaCppBackend(model_path=real_model_file, port=8090)
        assert isinstance(backend._model_path, Path)


# ---------------------------------------------------------------------------
# 3. Default port is accessible
# ---------------------------------------------------------------------------

class TestPortAttribute:
    def test_port_stored(self, real_model_file):
        backend = LlamaCppBackend(model_path=real_model_file, port=8765)
        assert backend._port == 8765

    def test_endpoint_url_includes_port(self, real_model_file):
        backend = LlamaCppBackend(model_path=real_model_file, port=8765)
        assert "8765" in backend.endpoint_url

    def test_endpoint_url_is_localhost(self, real_model_file):
        backend = LlamaCppBackend(model_path=real_model_file, port=8080)
        assert backend.endpoint_url == "http://localhost:8080"


# ---------------------------------------------------------------------------
# 4. start() calls subprocess.Popen (in real GPU mode)
# ---------------------------------------------------------------------------

class TestStartCallsPopen:
    def test_start_calls_popen_in_real_gpu_mode(self, real_model_file):
        mock_proc = MagicMock()
        mock_proc.pid = 12345

        with _patch_llama_cpp():
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "1"}):
                with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
                    backend = LlamaCppBackend(model_path=real_model_file, port=8090)
                    backend.start()

        mock_popen.assert_called_once()

    def test_start_returns_pid_in_real_gpu_mode(self, real_model_file):
        mock_proc = MagicMock()
        mock_proc.pid = 99999

        with _patch_llama_cpp():
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "1"}):
                with patch("subprocess.Popen", return_value=mock_proc):
                    backend = LlamaCppBackend(model_path=real_model_file, port=8090)
                    pid = backend.start()

        assert pid == 99999


# ---------------------------------------------------------------------------
# 5. start() passes model_path to Popen call arguments
# ---------------------------------------------------------------------------

class TestStartPassesModelPath:
    def test_popen_args_contain_model_path(self, real_model_file):
        mock_proc = MagicMock()
        mock_proc.pid = 42

        with _patch_llama_cpp():
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "1"}):
                with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
                    backend = LlamaCppBackend(model_path=real_model_file, port=8090)
                    backend.start()

        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        assert str(real_model_file) in cmd

    def test_popen_args_contain_port(self, real_model_file):
        mock_proc = MagicMock()
        mock_proc.pid = 42

        with _patch_llama_cpp():
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "1"}):
                with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
                    backend = LlamaCppBackend(model_path=real_model_file, port=8091)
                    backend.start()

        cmd = mock_popen.call_args[0][0]
        assert "8091" in cmd


# ---------------------------------------------------------------------------
# 6. health_check() returns False before start() (stub: returns True)
# ---------------------------------------------------------------------------

class TestHealthCheckBeforeStart:
    def test_health_check_in_stub_mode_returns_true(self, real_model_file):
        """In non-real-GPU mode, health_check always returns True (no real server)."""
        backend = LlamaCppBackend(model_path=real_model_file, port=8090)
        # No start() called; in stub mode health_check returns True immediately
        with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
            result = backend.health_check(timeout_seconds=1.0)
        assert result is True

    def test_health_check_with_no_process_in_real_mode_raises(self, real_model_file):
        """In real GPU mode with no running server, health_check should timeout."""
        backend = LlamaCppBackend(model_path=real_model_file, port=19998)
        with patch.dict(os.environ, {"DOJO_REAL_GPU": "1"}):
            from dojo.core.exceptions import ServerHealthTimeout
            with pytest.raises(ServerHealthTimeout):
                backend.health_check(timeout_seconds=0.1)


# ---------------------------------------------------------------------------
# 7. health_check() returns True when subprocess is running (mock)
# ---------------------------------------------------------------------------

class TestHealthCheckWhenRunning:
    def test_health_check_returns_true_in_stub_mode(self, real_model_file):
        with _patch_llama_cpp():
            backend = LlamaCppBackend(model_path=real_model_file, port=8090)
            backend.start()  # stub mode: no real subprocess
        with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
            result = backend.health_check()
        assert result is True

    def test_health_check_true_when_url_responds_200(self, real_model_file):
        """Mock urllib to simulate a healthy server in real-GPU mode."""
        import urllib.request
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with _patch_llama_cpp():
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "1"}):
                mock_proc = MagicMock()
                mock_proc.pid = 100
                with patch("subprocess.Popen", return_value=mock_proc):
                    backend = LlamaCppBackend(model_path=real_model_file, port=8090)
                    backend.start()

                with patch("urllib.request.urlopen", return_value=mock_resp):
                    result = backend.health_check(timeout_seconds=5.0)

        assert result is True


# ---------------------------------------------------------------------------
# 8. stop() calls terminate() on the subprocess
# ---------------------------------------------------------------------------

class TestStop:
    def test_stop_sends_sigterm(self, real_model_file):
        mock_proc = MagicMock()
        mock_proc.pid = 55555

        with _patch_llama_cpp():
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "1"}):
                with patch("subprocess.Popen", return_value=mock_proc):
                    backend = LlamaCppBackend(model_path=real_model_file, port=8090)
                    backend.start()

                with patch("os.kill") as mock_kill:
                    backend.stop()

        mock_kill.assert_called_once()
        args = mock_kill.call_args[0]
        assert args[0] == 55555  # PID sent

    def test_stop_with_no_process_no_crash(self, real_model_file):
        """stop() with no started process should silently do nothing."""
        backend = LlamaCppBackend(model_path=real_model_file, port=8090)
        # _process is None — must not raise
        backend.stop()


# ---------------------------------------------------------------------------
# 9. serve.llamacpp_loaded evidence emitted on start
# ---------------------------------------------------------------------------

class TestLlamacppLoadedEvidence:
    def test_llamacpp_loaded_emitted(self, real_model_file):
        with _patch_llama_cpp():
            backend = LlamaCppBackend(model_path=real_model_file, port=8090)
            backend.start()

        types = [e["type"] for e in EvidenceLogger.get_events()]
        assert "serve.llamacpp_loaded" in types

    def test_llamacpp_loaded_emitted_in_real_mode(self, real_model_file):
        mock_proc = MagicMock()
        mock_proc.pid = 77

        with _patch_llama_cpp():
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "1"}):
                with patch("subprocess.Popen", return_value=mock_proc):
                    backend = LlamaCppBackend(model_path=real_model_file, port=8090)
                    backend.start()

        types = [e["type"] for e in EvidenceLogger.get_events()]
        assert "serve.llamacpp_loaded" in types


# ---------------------------------------------------------------------------
# 10. Evidence has a model_path or port field
# ---------------------------------------------------------------------------

class TestLlamacppLoadedEvidenceFields:
    def test_loaded_evidence_has_model_path(self, real_model_file):
        with _patch_llama_cpp():
            backend = LlamaCppBackend(model_path=real_model_file, port=8090)
            backend.start()

        loaded_events = [
            e for e in EvidenceLogger.get_events()
            if e["type"] == "serve.llamacpp_loaded"
        ]
        assert len(loaded_events) >= 1
        assert "model_path" in loaded_events[0]["data"]

    def test_loaded_evidence_has_port(self, real_model_file):
        with _patch_llama_cpp():
            backend = LlamaCppBackend(model_path=real_model_file, port=8090)
            backend.start()

        loaded_events = [
            e for e in EvidenceLogger.get_events()
            if e["type"] == "serve.llamacpp_loaded"
        ]
        assert len(loaded_events) >= 1
        assert "port" in loaded_events[0]["data"]
        assert loaded_events[0]["data"]["port"] == 8090

    def test_loaded_evidence_model_path_matches(self, real_model_file):
        with _patch_llama_cpp():
            backend = LlamaCppBackend(model_path=real_model_file, port=8090)
            backend.start()

        loaded_events = [
            e for e in EvidenceLogger.get_events()
            if e["type"] == "serve.llamacpp_loaded"
        ]
        assert str(real_model_file) == loaded_events[0]["data"]["model_path"]


# ---------------------------------------------------------------------------
# 11. Model path not existing raises ServeError before subprocess spawn
# ---------------------------------------------------------------------------

class TestModelPathNotFound:
    def test_missing_model_path_raises_serve_error(self):
        nonexistent = Path("/nonexistent/path/model.gguf")
        backend = LlamaCppBackend(model_path=nonexistent, port=8090)
        with _patch_llama_cpp():
            with pytest.raises(ServeError):
                backend.start()

    def test_missing_model_path_no_subprocess_spawned(self):
        nonexistent = Path("/nonexistent/path/model.gguf")
        backend = LlamaCppBackend(model_path=nonexistent, port=8090)
        with _patch_llama_cpp():
            with patch("subprocess.Popen") as mock_popen:
                with pytest.raises(ServeError):
                    backend.start()
        mock_popen.assert_not_called()

    def test_serve_error_reason_contains_model_path(self):
        nonexistent = Path("/no/such/model.gguf")
        backend = LlamaCppBackend(model_path=nonexistent, port=8090)
        with _patch_llama_cpp():
            with pytest.raises(ServeError) as exc_info:
                backend.start()
        assert "model_path" in exc_info.value.reason or "not found" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 12. LlamaCppBackend has _process attribute (context-manager style check)
# ---------------------------------------------------------------------------

class TestBackendAttributes:
    def test_process_none_before_start(self, real_model_file):
        backend = LlamaCppBackend(model_path=real_model_file, port=8090)
        assert backend._process is None

    def test_process_set_after_start_in_real_mode(self, real_model_file):
        mock_proc = MagicMock()
        mock_proc.pid = 42

        with _patch_llama_cpp():
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "1"}):
                with patch("subprocess.Popen", return_value=mock_proc):
                    backend = LlamaCppBackend(model_path=real_model_file, port=8090)
                    backend.start()

        assert backend._process is mock_proc

    def test_process_none_after_stub_start(self, real_model_file):
        """In stub mode, _process remains None."""
        with _patch_llama_cpp():
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
                backend = LlamaCppBackend(model_path=real_model_file, port=8090)
                backend.start()

        assert backend._process is None


# ---------------------------------------------------------------------------
# 13. LocalGGUFRunner.generate() returns a dict with "text" key
# ---------------------------------------------------------------------------

class TestLocalGGUFRunnerGenerate:
    def test_generate_returns_dict(self, real_model_file):
        runner = LocalGGUFRunner(model_path=real_model_file)
        with _patch_llama_cpp():
            runner.load()
            result = runner.generate("Say hello")
        assert isinstance(result, dict)

    def test_generate_has_text_key(self, real_model_file):
        runner = LocalGGUFRunner(model_path=real_model_file)
        with _patch_llama_cpp():
            runner.load()
            result = runner.generate("Say hello")
        assert "text" in result

    def test_generate_text_is_string(self, real_model_file):
        runner = LocalGGUFRunner(model_path=real_model_file)
        with _patch_llama_cpp():
            runner.load()
            result = runner.generate("Say hello")
        assert isinstance(result["text"], str)

    def test_generate_has_tokens_key(self, real_model_file):
        runner = LocalGGUFRunner(model_path=real_model_file)
        with _patch_llama_cpp():
            runner.load()
            result = runner.generate("Hi", max_tokens=16)
        assert "tokens" in result
        assert isinstance(result["tokens"], int)

    def test_generate_stub_contains_prompt_prefix(self, real_model_file):
        runner = LocalGGUFRunner(model_path=real_model_file)
        with _patch_llama_cpp():
            runner.load()
            result = runner.generate("HelloWorld")
        # Stub returns first 20 chars of prompt
        assert "HelloWorld" in result["text"] or "local stub" in result["text"]


# ---------------------------------------------------------------------------
# 14. generate() evidence is emitted (serve.ready)
# ---------------------------------------------------------------------------

class TestGenerateEvidence:
    def test_generate_emits_serve_ready(self, real_model_file):
        runner = LocalGGUFRunner(model_path=real_model_file)
        with _patch_llama_cpp():
            runner.load()
            EvidenceLogger.clear()
            runner.generate("Test prompt")
        types = [e["type"] for e in EvidenceLogger.get_events()]
        assert "serve.ready" in types

    def test_generate_evidence_has_tokens_generated(self, real_model_file):
        runner = LocalGGUFRunner(model_path=real_model_file)
        with _patch_llama_cpp():
            runner.load()
            EvidenceLogger.clear()
            runner.generate("Test prompt")
        ready_events = [
            e for e in EvidenceLogger.get_events()
            if e["type"] == "serve.ready"
        ]
        assert len(ready_events) >= 1
        assert "tokens_generated" in ready_events[0]["data"]

    def test_generate_evidence_has_backend_field(self, real_model_file):
        runner = LocalGGUFRunner(model_path=real_model_file)
        with _patch_llama_cpp():
            runner.load()
            EvidenceLogger.clear()
            runner.generate("Test prompt")
        ready_events = [
            e for e in EvidenceLogger.get_events()
            if e["type"] == "serve.ready"
        ]
        assert len(ready_events) >= 1
        assert ready_events[0]["data"].get("backend") == "llamacpp_local"


# ---------------------------------------------------------------------------
# 15. stop() when already stopped doesn't crash
# ---------------------------------------------------------------------------

class TestStopIdempotent:
    def test_stop_when_already_stopped_no_error(self, real_model_file):
        backend = LlamaCppBackend(model_path=real_model_file, port=8090)
        # Never started — stop should be a no-op
        backend.stop()  # must not raise

    def test_stop_twice_no_error(self, real_model_file):
        mock_proc = MagicMock()
        mock_proc.pid = 55

        with _patch_llama_cpp():
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "1"}):
                with patch("subprocess.Popen", return_value=mock_proc):
                    backend = LlamaCppBackend(model_path=real_model_file, port=8090)
                    backend.start()

                with patch("os.kill", side_effect=ProcessLookupError):
                    backend.stop()  # first stop — PID gone → no crash
                    backend.stop()  # second stop — _process still set but os.kill suppressed


# ---------------------------------------------------------------------------
# 16. health_check() after stop returns True (stub) / times out (real)
# ---------------------------------------------------------------------------

class TestHealthCheckAfterStop:
    def test_health_check_stub_always_true(self, real_model_file):
        """In stub mode health_check always returns True regardless of lifecycle."""
        with _patch_llama_cpp():
            backend = LlamaCppBackend(model_path=real_model_file, port=8090)
            backend.start()
            backend.stop()

        with patch.dict(os.environ, {"DOJO_REAL_GPU": "0"}):
            result = backend.health_check(timeout_seconds=1.0)
        assert result is True

    def test_health_check_real_mode_after_stop_times_out(self, real_model_file):
        """After stopping, real-GPU mode health_check should timeout (no server)."""
        from dojo.core.exceptions import ServerHealthTimeout
        mock_proc = MagicMock()
        mock_proc.pid = 88

        with _patch_llama_cpp():
            with patch.dict(os.environ, {"DOJO_REAL_GPU": "1"}):
                with patch("subprocess.Popen", return_value=mock_proc):
                    backend = LlamaCppBackend(model_path=real_model_file, port=19997)
                    backend.start()

                with patch("os.kill"):
                    backend.stop()

                with pytest.raises(ServerHealthTimeout):
                    backend.health_check(timeout_seconds=0.1)


# ---------------------------------------------------------------------------
# Bonus: build_command includes expected flags
# ---------------------------------------------------------------------------

class TestBuildCommand:
    def test_build_command_has_model_flag(self, real_model_file):
        backend = LlamaCppBackend(model_path=real_model_file, port=8090)
        cmd = backend._build_command()
        assert "--model" in cmd
        assert str(real_model_file) in cmd

    def test_build_command_has_port_flag(self, real_model_file):
        backend = LlamaCppBackend(model_path=real_model_file, port=8765)
        cmd = backend._build_command()
        assert "--port" in cmd
        assert "8765" in cmd

    def test_build_command_starts_with_llama_server(self, real_model_file):
        backend = LlamaCppBackend(model_path=real_model_file, port=8090)
        cmd = backend._build_command()
        assert cmd[0] == "llama-server"


# ---------------------------------------------------------------------------
# Bonus: BackendUnavailable raised when llama_cpp not installed
# ---------------------------------------------------------------------------

class TestLlamaCppNotInstalled:
    def test_start_raises_backend_unavailable_when_no_llama_cpp(self, real_model_file):
        backend = LlamaCppBackend(model_path=real_model_file, port=8090)
        # Ensure llama_cpp import fails by removing it from sys.modules
        import sys
        # Patch builtins.__import__ via patch to simulate ImportError
        original = sys.modules.get("llama_cpp")
        sys.modules["llama_cpp"] = None  # type: ignore[assignment]
        try:
            with pytest.raises((BackendUnavailable, ImportError, TypeError)):
                backend.start()
        finally:
            if original is None:
                sys.modules.pop("llama_cpp", None)
            else:
                sys.modules["llama_cpp"] = original

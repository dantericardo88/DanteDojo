"""Unit tests for LocalGGUFRunner — local GGUF inference via llama-cpp-python bindings.

Covers:
- Stub-mode load() succeeds (DOJO_REAL_GPU=0)
- generate() returns correct dict shape
- tokens count is a positive integer
- evidence emission: SERVE_START on load(), SERVE_READY on generate()
- network_calls: 0 guaranteed in evidence
- ServerStartError on missing model file
- BackendUnavailable when llama_cpp not installed
- unload() clears the model reference
- Stub text includes the prompt prefix
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import BackendUnavailable, ServerStartError
from dojo.serve.llamacpp_backend import LocalGGUFRunner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture(autouse=True)
def stub_gpu_mode(monkeypatch):
    """Ensure DOJO_REAL_GPU=0 for all tests (CI-safe)."""
    monkeypatch.setenv("DOJO_REAL_GPU", "0")


@pytest.fixture()
def mock_llama_cpp():
    """Inject a mock llama_cpp module so _require_llama_cpp() passes."""
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"llama_cpp": mock_module}):
        yield mock_module


@pytest.fixture()
def runner(tmp_path, mock_llama_cpp):
    """A LocalGGUFRunner pointing at a real (empty) temp file."""
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"")  # empty file — load() only checks existence in stub mode
    return LocalGGUFRunner(model_path=model_file, passport_id="test-passport")


# ---------------------------------------------------------------------------
# Test 1: load() succeeds in stub mode
# ---------------------------------------------------------------------------

class TestLocalRunnerLoadStubMode:
    def test_load_stub_mode(self, runner):
        """load() must not raise in stub mode (DOJO_REAL_GPU=0)."""
        runner.load()  # should complete without error
        # In stub mode _llm stays None — that is expected
        assert runner._llm is None


# ---------------------------------------------------------------------------
# Test 2: generate() returns dict with "text" key
# ---------------------------------------------------------------------------

class TestLocalRunnerGenerateReturnsText:
    def test_generate_returns_text_key(self, runner):
        runner.load()
        result = runner.generate("Hello world")
        assert "text" in result
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0


# ---------------------------------------------------------------------------
# Test 3: generate() returns positive "tokens" count
# ---------------------------------------------------------------------------

class TestLocalRunnerGenerateReturnsTokensCount:
    def test_generate_tokens_is_positive_int(self, runner):
        runner.load()
        result = runner.generate("test prompt", max_tokens=8)
        assert "tokens" in result
        assert isinstance(result["tokens"], int)
        assert result["tokens"] > 0


# ---------------------------------------------------------------------------
# Test 4: evidence contains network_calls: 0
# ---------------------------------------------------------------------------

class TestLocalRunnerGenerateNetworkCallsZero:
    def test_network_calls_zero_in_evidence(self, runner):
        runner.load()
        runner.generate("prompt text")

        events = EvidenceLogger._events
        serve_ready_events = [
            e for e in events
            if isinstance(e.get("evidence"), dict)
            and e["evidence"].get("type") == "serve.ready"
        ]
        assert len(serve_ready_events) >= 1
        data = serve_ready_events[-1]["evidence"]["data"]
        assert data.get("network_calls") == 0


# ---------------------------------------------------------------------------
# Test 5: load() emits SERVE_START evidence
# ---------------------------------------------------------------------------

class TestLocalRunnerEmitsServeStartEvidence:
    def test_emits_serve_start_on_load(self, runner):
        runner.load()

        events = EvidenceLogger._events
        evidence_types = [
            e["evidence"].get("type")
            for e in events
            if isinstance(e.get("evidence"), dict)
        ]
        assert "serve.start" in evidence_types

    def test_serve_start_has_correct_backend(self, runner):
        runner.load()

        events = EvidenceLogger._events
        serve_start = next(
            (
                e["evidence"]["data"]
                for e in events
                if isinstance(e.get("evidence"), dict)
                and e["evidence"].get("type") == "serve.start"
            ),
            None,
        )
        assert serve_start is not None
        assert serve_start.get("backend") == "llamacpp_local"
        assert serve_start.get("mode") == "local_binding"


# ---------------------------------------------------------------------------
# Test 6: generate() emits SERVE_READY evidence
# ---------------------------------------------------------------------------

class TestLocalRunnerEmitsServeReadyEvidence:
    def test_emits_serve_ready_on_generate(self, runner):
        runner.load()
        runner.generate("some prompt")

        events = EvidenceLogger._events
        evidence_types = [
            e["evidence"].get("type")
            for e in events
            if isinstance(e.get("evidence"), dict)
        ]
        assert "serve.ready" in evidence_types

    def test_serve_ready_has_model_path(self, runner):
        runner.load()
        runner.generate("some prompt")

        events = EvidenceLogger._events
        serve_ready = next(
            (
                e["evidence"]["data"]
                for e in events
                if isinstance(e.get("evidence"), dict)
                and e["evidence"].get("type") == "serve.ready"
            ),
            None,
        )
        assert serve_ready is not None
        assert "model_path" in serve_ready


# ---------------------------------------------------------------------------
# Test 7: ServerStartError when model file not found
# ---------------------------------------------------------------------------

class TestLocalRunnerModelNotFoundRaises:
    def test_nonexistent_path_raises_server_start_error(self, tmp_path, mock_llama_cpp):
        runner = LocalGGUFRunner(model_path=tmp_path / "does_not_exist.gguf")
        with pytest.raises(ServerStartError):
            runner.load()

    def test_error_message_mentions_path(self, tmp_path, mock_llama_cpp):
        missing = tmp_path / "missing_model.gguf"
        runner = LocalGGUFRunner(model_path=missing)
        with pytest.raises(ServerStartError, match="missing_model.gguf"):
            runner.load()


# ---------------------------------------------------------------------------
# Test 8: BackendUnavailable when llama_cpp not installed
# ---------------------------------------------------------------------------

class TestLocalRunnerMissingLlamaCppRaises:
    def test_missing_llama_cpp_raises_backend_unavailable(self, tmp_path):
        """When llama_cpp is not importable, load() must raise BackendUnavailable."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"")
        runner = LocalGGUFRunner(model_path=model_file)

        # Ensure llama_cpp is NOT in sys.modules
        with patch.dict(sys.modules, {"llama_cpp": None}):
            with pytest.raises(BackendUnavailable):
                runner.load()


# ---------------------------------------------------------------------------
# Test 9: unload() sets _llm to None
# ---------------------------------------------------------------------------

class TestLocalRunnerUnloadClearsModel:
    def test_unload_sets_llm_to_none(self, runner):
        runner.load()
        # In stub mode _llm is already None, but let's set it to something
        runner._llm = MagicMock()
        assert runner._llm is not None

        runner.unload()
        assert runner._llm is None

    def test_unload_is_idempotent(self, runner):
        """Calling unload() twice must not raise."""
        runner.unload()
        runner.unload()
        assert runner._llm is None


# ---------------------------------------------------------------------------
# Test 10: stub text contains prompt prefix
# ---------------------------------------------------------------------------

class TestLocalRunnerStubTextContainsPrompt:
    def test_stub_text_contains_prompt_prefix(self, runner):
        runner.load()
        prompt = "Tell me about dragons"
        result = runner.generate(prompt)
        # Stub format: "[local stub: <first 20 chars>...]"
        assert "Tell me about dragon" in result["text"]

    def test_stub_text_format(self, runner):
        runner.load()
        result = runner.generate("Hello")
        assert result["text"].startswith("[local stub:")
        assert result["text"].endswith("...]")

    def test_prompt_tokens_reflect_word_count(self, runner):
        runner.load()
        prompt = "one two three four five"
        result = runner.generate(prompt)
        assert result["prompt_tokens"] == 5


# ---------------------------------------------------------------------------
# GGUF export tests (d24)
# ---------------------------------------------------------------------------

from unittest.mock import patch as _patch


@pytest.fixture(autouse=False)
def stub_gguf_env(monkeypatch):
    """Ensure DOJO_REAL_GPU=0 and DOJO_GGUF_STUB=1 for export tests."""
    monkeypatch.setenv("DOJO_REAL_GPU", "0")
    monkeypatch.setenv("DOJO_GGUF_STUB", "1")


class TestExportGgufStubMode:
    """export_gguf works without llama_cpp installed in stub mode."""

    def test_export_succeeds_without_llama_cpp(self, tmp_path, stub_gguf_env):
        """Stub mode must export successfully even if llama_cpp is not installed."""
        from dojo.quant.gguf import export_gguf

        model_dir = tmp_path / "mymodel"
        model_dir.mkdir()
        out_dir = tmp_path / "out"

        # Remove llama_cpp from sys.modules to simulate it being absent
        with _patch.dict(sys.modules, {"llama_cpp": None}):
            result = export_gguf(model_dir, out_dir, quant_type="Q4_K_M")

        assert result.exists(), f"Output GGUF not found: {result}"

    def test_stub_output_has_gguf_magic(self, tmp_path, stub_gguf_env):
        """Stub GGUF output must start with the GGUF magic bytes 0x47 0x47 0x55 0x46."""
        from dojo.quant.gguf import export_gguf

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        out_dir = tmp_path / "out"

        with _patch.dict(sys.modules, {"llama_cpp": None}):
            result = export_gguf(model_dir, out_dir)

        magic = result.read_bytes()[:4]
        assert magic == b"\x47\x47\x55\x46", f"Expected GGUF magic, got: {magic!r}"

    def test_stub_output_path_contains_quant_type(self, tmp_path, stub_gguf_env):
        """Output filename must include the quant_type string."""
        from dojo.quant.gguf import export_gguf

        model_dir = tmp_path / "mymodel"
        model_dir.mkdir()
        out_dir = tmp_path / "out"

        with _patch.dict(sys.modules, {"llama_cpp": None}):
            result = export_gguf(model_dir, out_dir, quant_type="Q5_K_M")

        assert "Q5_K_M" in result.name


class TestExportGgufPassportSidecar:
    """Passport metadata sidecar is written when passport_id is provided."""

    def test_sidecar_written_with_passport_id(self, tmp_path, stub_gguf_env):
        """When passport_id is given, a .meta.json sidecar must be created."""
        from dojo.quant.gguf import export_gguf

        model_dir = tmp_path / "m"
        model_dir.mkdir()
        out_dir = tmp_path / "out"

        with _patch.dict(sys.modules, {"llama_cpp": None}):
            result = export_gguf(model_dir, out_dir, passport_id="pp-test-123")

        sidecar = result.with_suffix(".meta.json")
        assert sidecar.exists(), "Expected .meta.json sidecar alongside GGUF"

    def test_sidecar_contains_passport_id(self, tmp_path, stub_gguf_env):
        """The sidecar JSON must have dojo.passport_id matching the provided value."""
        import json as _json
        from dojo.quant.gguf import export_gguf

        model_dir = tmp_path / "m"
        model_dir.mkdir()
        out_dir = tmp_path / "out"
        pid = "pp-constitutional-456"

        with _patch.dict(sys.modules, {"llama_cpp": None}):
            result = export_gguf(model_dir, out_dir, passport_id=pid)

        sidecar = result.with_suffix(".meta.json")
        meta = _json.loads(sidecar.read_text(encoding="utf-8"))
        assert meta.get("dojo.passport_id") == pid

    def test_no_sidecar_without_passport_id(self, tmp_path, stub_gguf_env):
        """When no passport_id is provided, no .meta.json sidecar should be written."""
        from dojo.quant.gguf import export_gguf

        model_dir = tmp_path / "m"
        model_dir.mkdir()
        out_dir = tmp_path / "out"

        with _patch.dict(sys.modules, {"llama_cpp": None}):
            result = export_gguf(model_dir, out_dir)

        sidecar = result.with_suffix(".meta.json")
        assert not sidecar.exists(), "Sidecar should not be written without passport_id"

    def test_sidecar_has_sha256(self, tmp_path, stub_gguf_env):
        """The sidecar must include a dojo.gguf_sha256 field."""
        import json as _json
        from dojo.quant.gguf import export_gguf

        model_dir = tmp_path / "m"
        model_dir.mkdir()
        out_dir = tmp_path / "out"

        with _patch.dict(sys.modules, {"llama_cpp": None}):
            result = export_gguf(model_dir, out_dir, passport_id="pp-sha")

        meta = _json.loads(result.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert "dojo.gguf_sha256" in meta
        assert len(meta["dojo.gguf_sha256"]) == 64  # sha256 hex digest length


class TestExportGgufRealConversionMocked:
    """Real conversion path: mock _find_convert_script + subprocess to verify integration."""

    def test_real_conversion_calls_subprocess(self, tmp_path, monkeypatch):
        """When a convert script is found, subprocess.run should be called."""
        import subprocess as _subprocess
        from dojo.quant.gguf import export_gguf

        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        monkeypatch.setenv("DOJO_GGUF_STUB", "0")

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        out_dir = tmp_path / "out"
        fake_script = tmp_path / "convert_hf_to_gguf.py"
        fake_script.write_text("# mock", encoding="utf-8")

        # subprocess.run returns success; also create the f16 output file
        f16_path = out_dir / f"{model_dir.name}-f16.gguf"

        def fake_run(cmd, **kwargs):
            # Create the f16 output when the convert script "runs"
            out_dir.mkdir(parents=True, exist_ok=True)
            f16_path.write_bytes(b"\x47\x47\x55\x46" + b"\x00" * 20)
            r = MagicMock()
            r.returncode = 0
            r.stderr = ""
            return r

        with _patch("dojo.quant.gguf._find_convert_script", return_value=fake_script):
            with _patch("subprocess.run", side_effect=fake_run) as mock_run:
                result = export_gguf(model_dir, out_dir, quant_type="Q4_K_M")

        assert mock_run.called, "subprocess.run should have been called for real conversion"
        assert result.exists()

    def test_real_mode_no_llama_cpp_stub_disabled_raises(self, tmp_path, monkeypatch):
        """When stub is disabled and no convert script found, raise QuantizationError."""
        from dojo.quant.gguf import export_gguf
        from dojo.core.exceptions import QuantizationError

        monkeypatch.setenv("DOJO_REAL_GPU", "0")
        monkeypatch.setenv("DOJO_GGUF_STUB", "0")

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        out_dir = tmp_path / "out"

        with _patch("dojo.quant.gguf._find_convert_script", return_value=None):
            with _patch.dict(sys.modules, {"llama_cpp": None}):
                with pytest.raises(QuantizationError):
                    export_gguf(model_dir, out_dir)


class TestExportGgufEvidenceEmission:
    """export_gguf emits QUANTIZATION_START and QUANTIZATION_COMPLETE evidence."""

    def test_emits_quantization_start(self, tmp_path, stub_gguf_env):
        from dojo.quant.gguf import export_gguf

        model_dir = tmp_path / "m"
        model_dir.mkdir()
        out_dir = tmp_path / "out"

        with _patch.dict(sys.modules, {"llama_cpp": None}):
            export_gguf(model_dir, out_dir)

        events = EvidenceLogger.get_events()
        types = [e["type"] for e in events]
        assert "quant.start" in types

    def test_emits_quantization_complete(self, tmp_path, stub_gguf_env):
        from dojo.quant.gguf import export_gguf

        model_dir = tmp_path / "m"
        model_dir.mkdir()
        out_dir = tmp_path / "out"

        with _patch.dict(sys.modules, {"llama_cpp": None}):
            export_gguf(model_dir, out_dir)

        events = EvidenceLogger.get_events()
        types = [e["type"] for e in events]
        assert "quant.complete" in types

    def test_complete_event_has_stub_flag(self, tmp_path, stub_gguf_env):
        from dojo.quant.gguf import export_gguf

        model_dir = tmp_path / "m"
        model_dir.mkdir()
        out_dir = tmp_path / "out"

        with _patch.dict(sys.modules, {"llama_cpp": None}):
            export_gguf(model_dir, out_dir)

        events = EvidenceLogger.get_events()
        complete_ev = next(
            (e for e in events if e["type"] == "quant.complete"),
            None,
        )
        assert complete_ev is not None
        assert complete_ev["data"].get("stub") is True

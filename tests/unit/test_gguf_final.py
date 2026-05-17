"""Final d24 GGUF conversion tests — sidecar, SHA256, stub mode, evidence.

Covers:
1.  _find_convert_script() returns None when no script found (no crash)
2.  _find_convert_script() returns a Path when script exists (mock filesystem)
3.  export_gguf in stub mode (DOJO_GGUF_STUB=1) completes without error
4.  Stub mode returns a path ending in .gguf
5.  export_gguf emits QUANTIZATION_COMPLETE evidence
6.  Evidence has output_path field
7.  Evidence has quant_type field
8.  Passport ID written to sidecar .meta.json when passport_id provided
9.  Sidecar JSON contains dojo.passport_id (or passport_id) field
10. Sidecar JSON contains model_path (or dojo.gguf_sha256) — provenance record
11. SHA256 sidecar / evidence has sha256 field
12. export_gguf with DOJO_GGUF_STUB=0 and no convert script raises QuantizationError
13. export_gguf with DOJO_GGUF_STUB=0 and convert script calls subprocess
14. output_dir is created if it doesn't exist
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import subprocess

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import QuantizationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture(autouse=True)
def stub_mode_env(monkeypatch):
    """Default: stub mode on, real GPU off."""
    monkeypatch.setenv("DOJO_GGUF_STUB", "1")
    monkeypatch.setenv("DOJO_REAL_GPU", "0")
    monkeypatch.setenv("DOJO_AUTO_DOWNLOAD_SCRIPTS", "0")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_complete_evidence_data() -> dict | None:
    """Return data dict from the first QUANTIZATION_COMPLETE evidence event."""
    for e in EvidenceLogger._events:
        ev = e.get("evidence", {})
        if isinstance(ev, dict) and ev.get("type") == "quant.complete":
            return ev.get("data", {})
    return None


# ---------------------------------------------------------------------------
# 1. _find_convert_script() returns None when no script found
# ---------------------------------------------------------------------------

class TestFindConvertScript:
    def test_returns_none_when_no_script(self, monkeypatch, tmp_path):
        """_find_convert_script() must return None (not crash) when nothing found."""
        from dojo.quant import gguf as gguf_module

        # Disable auto-download and point LLAMA_CPP_PATH to empty dir
        monkeypatch.setenv("DOJO_AUTO_DOWNLOAD_SCRIPTS", "0")
        monkeypatch.setenv("LLAMA_CPP_PATH", str(tmp_path))

        # Ensure llama_cpp is not importable
        with patch.dict(sys.modules, {"llama_cpp": None}):
            # Patch Path.home() to a non-existent path so no candidate matches
            with patch("dojo.quant.gguf.Path.home", return_value=tmp_path / "nonexistent_home"):
                result = gguf_module._find_convert_script()

        assert result is None, f"Expected None, got {result!r}"

    def test_no_crash_on_missing_script(self, monkeypatch, tmp_path):
        """_find_convert_script() must not raise an exception on any OS."""
        from dojo.quant import gguf as gguf_module

        monkeypatch.setenv("DOJO_AUTO_DOWNLOAD_SCRIPTS", "0")
        monkeypatch.setenv("LLAMA_CPP_PATH", "")

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with patch("dojo.quant.gguf.Path.home", return_value=tmp_path / "no_home"):
                # Should not raise
                result = gguf_module._find_convert_script()

        # result is either None or a Path — both are valid
        assert result is None or isinstance(result, Path)

    # -----------------------------------------------------------------------
    # 2. _find_convert_script() returns a Path when script exists
    # -----------------------------------------------------------------------

    def test_returns_path_when_script_exists_via_env(self, monkeypatch, tmp_path):
        """_find_convert_script() returns a Path when LLAMA_CPP_PATH points to a dir
        that contains convert_hf_to_gguf.py."""
        from dojo.quant import gguf as gguf_module

        monkeypatch.setenv("DOJO_AUTO_DOWNLOAD_SCRIPTS", "0")
        monkeypatch.setenv("LLAMA_CPP_PATH", str(tmp_path))

        # Create the script file
        script = tmp_path / "convert_hf_to_gguf.py"
        script.write_text("# fake script\n", encoding="utf-8")

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with patch("dojo.quant.gguf.Path.home", return_value=tmp_path / "no_home"):
                result = gguf_module._find_convert_script()

        assert result is not None, "Expected a Path, got None"
        assert isinstance(result, Path)
        assert result.name == "convert_hf_to_gguf.py"

    def test_returns_path_from_home_dir(self, monkeypatch, tmp_path):
        """_find_convert_script() returns Path when ~/llama.cpp/convert_hf_to_gguf.py exists."""
        from dojo.quant import gguf as gguf_module

        monkeypatch.setenv("DOJO_AUTO_DOWNLOAD_SCRIPTS", "0")
        monkeypatch.setenv("LLAMA_CPP_PATH", "")

        fake_home = tmp_path / "fakehome"
        llama_dir = fake_home / "llama.cpp"
        llama_dir.mkdir(parents=True)
        script = llama_dir / "convert_hf_to_gguf.py"
        script.write_text("# fake\n", encoding="utf-8")

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with patch("dojo.quant.gguf.Path.home", return_value=fake_home):
                result = gguf_module._find_convert_script()

        assert result is not None
        assert result == script


# ---------------------------------------------------------------------------
# 3 & 4. Stub mode completes without error and returns .gguf path
# ---------------------------------------------------------------------------

class TestStubMode:
    def test_stub_mode_completes_without_error(self, tmp_path):
        """export_gguf with DOJO_GGUF_STUB=1 must complete without raising."""
        from dojo.quant import gguf as gguf_module

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with patch.object(gguf_module, "_find_convert_script", return_value=None):
                result = gguf_module.export_gguf(
                    model_path=tmp_path / "model",
                    output_dir=tmp_path / "out",
                    quant_type="Q4_K_M",
                )

        assert result is not None

    def test_stub_mode_returns_gguf_path(self, tmp_path):
        """Stub mode must return a path whose suffix is .gguf."""
        from dojo.quant import gguf as gguf_module

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with patch.object(gguf_module, "_find_convert_script", return_value=None):
                result = gguf_module.export_gguf(
                    model_path=tmp_path / "model",
                    output_dir=tmp_path / "out",
                    quant_type="Q5_K_M",
                )

        assert str(result).endswith(".gguf"), f"Expected .gguf suffix, got {result}"

    def test_stub_mode_file_exists(self, tmp_path):
        """Stub mode must actually write a file to disk."""
        from dojo.quant import gguf as gguf_module

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with patch.object(gguf_module, "_find_convert_script", return_value=None):
                result = gguf_module.export_gguf(
                    model_path=tmp_path / "model",
                    output_dir=tmp_path / "out",
                )

        assert result.exists(), f"Stub output file not found: {result}"


# ---------------------------------------------------------------------------
# 5. export_gguf emits QUANTIZATION_COMPLETE evidence
# ---------------------------------------------------------------------------

class TestQuantizationEvidence:
    def test_emits_quantization_complete_evidence(self, tmp_path):
        """export_gguf must emit a quant.complete evidence event."""
        from dojo.quant import gguf as gguf_module

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with patch.object(gguf_module, "_find_convert_script", return_value=None):
                gguf_module.export_gguf(
                    model_path=tmp_path / "model",
                    output_dir=tmp_path / "out",
                )

        types = [
            e["evidence"].get("type")
            for e in EvidenceLogger._events
            if isinstance(e.get("evidence"), dict)
        ]
        assert "quant.complete" in types, (
            f"Expected 'quant.complete' in evidence, got: {types}"
        )

    # -----------------------------------------------------------------------
    # 6. Evidence has output_path field
    # -----------------------------------------------------------------------

    def test_evidence_has_output_path(self, tmp_path):
        """quant.complete evidence data must include 'output_path' field."""
        from dojo.quant import gguf as gguf_module

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with patch.object(gguf_module, "_find_convert_script", return_value=None):
                gguf_module.export_gguf(
                    model_path=tmp_path / "model",
                    output_dir=tmp_path / "out",
                )

        data = _get_complete_evidence_data()
        assert data is not None, "No quant.complete evidence found"
        assert "output_path" in data, (
            f"Evidence data missing 'output_path': {data}"
        )

    def test_evidence_output_path_ends_with_gguf(self, tmp_path):
        """The output_path in evidence must end with .gguf."""
        from dojo.quant import gguf as gguf_module

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with patch.object(gguf_module, "_find_convert_script", return_value=None):
                gguf_module.export_gguf(
                    model_path=tmp_path / "model",
                    output_dir=tmp_path / "out",
                )

        data = _get_complete_evidence_data()
        assert data is not None
        assert str(data["output_path"]).endswith(".gguf"), (
            f"output_path should end with .gguf, got: {data['output_path']}"
        )

    # -----------------------------------------------------------------------
    # 7. Evidence has quant_type field
    # -----------------------------------------------------------------------

    def test_evidence_has_quant_type(self, tmp_path):
        """quant.complete evidence data must include 'quant_type' field."""
        from dojo.quant import gguf as gguf_module

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with patch.object(gguf_module, "_find_convert_script", return_value=None):
                gguf_module.export_gguf(
                    model_path=tmp_path / "model",
                    output_dir=tmp_path / "out",
                    quant_type="Q4_K_M",
                )

        data = _get_complete_evidence_data()
        assert data is not None
        assert "quant_type" in data, f"Evidence data missing 'quant_type': {data}"
        assert data["quant_type"] == "Q4_K_M"


# ---------------------------------------------------------------------------
# 8–10. Passport ID sidecar
# ---------------------------------------------------------------------------

class TestPassportSidecar:
    def test_sidecar_created_when_passport_id_provided(self, tmp_path):
        """A sidecar file must be created alongside the .gguf when passport_id is given."""
        from dojo.quant import gguf as gguf_module

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with patch.object(gguf_module, "_find_convert_script", return_value=None):
                result = gguf_module.export_gguf(
                    model_path=tmp_path / "model",
                    output_dir=tmp_path / "out",
                    passport_id="test-passport-123",
                )

        # Look for any sidecar alongside the .gguf
        sidecar_candidates = [
            result.with_suffix(".meta.json"),
            result.with_suffix(".json"),
            result.parent / (result.name + ".json"),
        ]
        found = [s for s in sidecar_candidates if s.exists()]
        assert found, (
            f"No sidecar file found alongside {result}. "
            f"Checked: {sidecar_candidates}"
        )

    def test_sidecar_contains_passport_id(self, tmp_path):
        """The sidecar JSON must contain the passport_id value."""
        from dojo.quant import gguf as gguf_module

        passport = "sovereign-passport-abc-456"

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with patch.object(gguf_module, "_find_convert_script", return_value=None):
                result = gguf_module.export_gguf(
                    model_path=tmp_path / "model",
                    output_dir=tmp_path / "out",
                    passport_id=passport,
                )

        # Try all sidecar paths
        sidecar_candidates = [
            result.with_suffix(".meta.json"),
            result.with_suffix(".json"),
            result.parent / (result.name + ".json"),
        ]
        sidecar = next((s for s in sidecar_candidates if s.exists()), None)
        assert sidecar is not None, "No sidecar file found"

        content = json.loads(sidecar.read_text(encoding="utf-8"))
        # Accept either key format: "dojo.passport_id" or "passport_id"
        passport_value = content.get("dojo.passport_id") or content.get("passport_id")
        assert passport_value == passport, (
            f"Expected passport_id={passport!r}, got {passport_value!r} in {content}"
        )

    def test_sidecar_contains_provenance_record(self, tmp_path):
        """The sidecar JSON must include some provenance field (sha256 or model_path)."""
        from dojo.quant import gguf as gguf_module

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with patch.object(gguf_module, "_find_convert_script", return_value=None):
                result = gguf_module.export_gguf(
                    model_path=tmp_path / "model",
                    output_dir=tmp_path / "out",
                    passport_id="provenance-test",
                )

        sidecar_candidates = [
            result.with_suffix(".meta.json"),
            result.with_suffix(".json"),
            result.parent / (result.name + ".json"),
        ]
        sidecar = next((s for s in sidecar_candidates if s.exists()), None)
        assert sidecar is not None, "No sidecar found"

        content = json.loads(sidecar.read_text(encoding="utf-8"))
        # Must contain at least one provenance field
        provenance_keys = {"dojo.gguf_sha256", "sha256", "model_path", "dojo.model_path"}
        found_keys = set(content.keys()) & provenance_keys
        assert found_keys, (
            f"Sidecar missing provenance field (one of {provenance_keys}), got keys: {set(content.keys())}"
        )

    def test_no_sidecar_when_no_passport_id(self, tmp_path):
        """No .meta.json sidecar should be written when passport_id is empty."""
        from dojo.quant import gguf as gguf_module

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with patch.object(gguf_module, "_find_convert_script", return_value=None):
                result = gguf_module.export_gguf(
                    model_path=tmp_path / "model",
                    output_dir=tmp_path / "out",
                    passport_id="",
                )

        meta = result.with_suffix(".meta.json")
        assert not meta.exists(), (
            f"Sidecar {meta} should NOT exist when passport_id is empty"
        )


# ---------------------------------------------------------------------------
# 11. SHA256 sidecar or evidence has sha256 field
# ---------------------------------------------------------------------------

class TestSha256:
    def test_evidence_has_file_hash(self, tmp_path):
        """quant.complete evidence must include 'file_hash' (SHA256) of output."""
        from dojo.quant import gguf as gguf_module

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with patch.object(gguf_module, "_find_convert_script", return_value=None):
                gguf_module.export_gguf(
                    model_path=tmp_path / "model",
                    output_dir=tmp_path / "out",
                )

        data = _get_complete_evidence_data()
        assert data is not None
        # Accept either key: "file_hash" or "sha256"
        sha_value = data.get("file_hash") or data.get("sha256")
        assert sha_value is not None, (
            f"Evidence missing SHA256 field (file_hash or sha256): {data}"
        )
        assert len(sha_value) == 64, (
            f"SHA256 hex digest should be 64 chars, got {len(sha_value)}: {sha_value!r}"
        )

    def test_passport_sidecar_contains_sha256(self, tmp_path):
        """When passport_id is provided, the sidecar must contain a sha256 field."""
        from dojo.quant import gguf as gguf_module

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with patch.object(gguf_module, "_find_convert_script", return_value=None):
                result = gguf_module.export_gguf(
                    model_path=tmp_path / "model",
                    output_dir=tmp_path / "out",
                    passport_id="sha256-test-passport",
                )

        sidecar_candidates = [
            result.with_suffix(".meta.json"),
            result.with_suffix(".json"),
        ]
        sidecar = next((s for s in sidecar_candidates if s.exists()), None)
        assert sidecar is not None, "No sidecar found"

        content = json.loads(sidecar.read_text(encoding="utf-8"))
        sha = content.get("dojo.gguf_sha256") or content.get("sha256") or content.get("file_hash")
        assert sha is not None, f"Sidecar missing sha256 field: {content}"
        assert len(sha) == 64, f"SHA256 should be 64 chars, got {len(sha)}"


# ---------------------------------------------------------------------------
# 12. export_gguf with DOJO_GGUF_STUB=0 and no convert script raises QuantizationError
# ---------------------------------------------------------------------------

class TestRealModeErrors:
    def test_raises_when_stub_off_and_no_script(self, monkeypatch, tmp_path):
        """With DOJO_GGUF_STUB=0 and no convert script, QuantizationError must be raised."""
        from dojo.quant import gguf as gguf_module

        monkeypatch.setenv("DOJO_GGUF_STUB", "0")

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with patch.object(gguf_module, "_find_convert_script", return_value=None):
                with pytest.raises(QuantizationError):
                    gguf_module.export_gguf(
                        model_path=tmp_path / "model",
                        output_dir=tmp_path / "out",
                    )

    # -----------------------------------------------------------------------
    # 13. With convert script, subprocess is called
    # -----------------------------------------------------------------------

    def test_calls_subprocess_when_convert_script_found(self, monkeypatch, tmp_path):
        """With a convert script present, export_gguf must invoke subprocess.run."""
        from dojo.quant import gguf as gguf_module

        monkeypatch.setenv("DOJO_GGUF_STUB", "0")
        monkeypatch.setenv("DOJO_REAL_GPU", "1")

        fake_script = tmp_path / "convert_hf_to_gguf.py"
        fake_script.write_text("# fake\n", encoding="utf-8")

        # Fake subprocess result: success, writes the f16 file
        f16_output = tmp_path / "out" / "model-f16.gguf"

        def fake_subprocess_run(cmd, **kwargs):
            # Simulate the convert script creating the f16 file
            f16_output.parent.mkdir(parents=True, exist_ok=True)
            f16_output.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 16)
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            return result

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with patch.object(gguf_module, "_find_convert_script", return_value=fake_script):
                with patch("dojo.quant.gguf.subprocess.run", side_effect=fake_subprocess_run) as mock_run:
                    with patch("dojo.quant.gguf.shutil.which", return_value=None):
                        try:
                            gguf_module.export_gguf(
                                model_path=tmp_path / "model",
                                output_dir=tmp_path / "out",
                            )
                        except Exception:
                            pass  # may fail due to file rename — we only care about subprocess call

        assert mock_run.called, "subprocess.run should have been called with the convert script"
        # Verify the convert script path appears in the call args
        first_call_cmd = mock_run.call_args_list[0][0][0]
        assert str(fake_script) in first_call_cmd, (
            f"Expected fake script in subprocess command, got: {first_call_cmd}"
        )


# ---------------------------------------------------------------------------
# 14. output_dir is created if it doesn't exist
# ---------------------------------------------------------------------------

class TestOutputDirCreation:
    def test_output_dir_created_if_missing(self, tmp_path):
        """export_gguf must create the output_dir even if it doesn't exist beforehand."""
        from dojo.quant import gguf as gguf_module

        out_dir = tmp_path / "nested" / "output" / "dir"
        assert not out_dir.exists(), "Test precondition: out_dir should not exist yet"

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with patch.object(gguf_module, "_find_convert_script", return_value=None):
                gguf_module.export_gguf(
                    model_path=tmp_path / "model",
                    output_dir=out_dir,
                )

        assert out_dir.exists(), f"output_dir {out_dir} was not created"

    def test_output_dir_exists_no_error(self, tmp_path):
        """export_gguf must not error when output_dir already exists."""
        from dojo.quant import gguf as gguf_module

        out_dir = tmp_path / "existing_out"
        out_dir.mkdir(parents=True)

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with patch.object(gguf_module, "_find_convert_script", return_value=None):
                result = gguf_module.export_gguf(
                    model_path=tmp_path / "model",
                    output_dir=out_dir,
                )

        assert result.exists()

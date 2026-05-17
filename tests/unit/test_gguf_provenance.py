"""Tests for d24 — GGUF provenance chain, quant-type validation, round-trip integrity.

Covers:
1.  export_gguf in stub mode creates output .gguf file
2.  Sidecar file exists alongside .gguf
3.  Sidecar is valid JSON
4.  Sidecar has dojo.passport_id field when passport_id provided
5.  Sidecar has dojo.gguf_sha256 (hash) field
6.  Sidecar has expected top-level keys only
7.  SHA256 in sidecar matches actual file hash (round-trip integrity)
8.  Unsupported quant_type raises QuantizationError
9.  QuantizationError message lists at least one valid quant type
10. QUANTIZATION_COMPLETE evidence has file_hash field
11. QUANTIZATION_COMPLETE evidence has output_path field
12. QUANTIZATION_COMPLETE evidence has quant_type field
13. export_gguf twice with same args produces consistent sidecar schema
14. Passport provenance: sidecar gets merkle_root + issued_at from profile.passport
15. Sidecar written without passport_id does NOT exist (no crash)
16. export_gguf returns a Path ending in .gguf
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import QuantizationError
from dojo.quant.gguf import VALID_QUANT_TYPES, export_gguf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture(autouse=True)
def stub_env(monkeypatch):
    """Guarantee stub mode (no real GPU / llama.cpp required)."""
    monkeypatch.setenv("DOJO_REAL_GPU", "0")
    monkeypatch.setenv("DOJO_GGUF_STUB", "1")


@pytest.fixture()
def model_dir(tmp_path):
    d = tmp_path / "mymodel"
    d.mkdir()
    return d


@pytest.fixture()
def out_dir(tmp_path):
    return tmp_path / "out"


def _export(model_dir, out_dir, **kwargs):
    """Call export_gguf with llama_cpp absent (pure stub)."""
    with patch.dict(sys.modules, {"llama_cpp": None}):
        return export_gguf(model_dir, out_dir, **kwargs)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Test 1: output .gguf file is created
# ---------------------------------------------------------------------------

class TestGgufOutputFileCreated:
    def test_gguf_file_exists(self, model_dir, out_dir):
        result = _export(model_dir, out_dir, passport_id="pp-001")
        assert result.exists(), f"Expected .gguf file at {result}"

    def test_output_path_ends_with_gguf(self, model_dir, out_dir):
        result = _export(model_dir, out_dir, passport_id="pp-001")
        assert result.suffix == ".gguf"


# ---------------------------------------------------------------------------
# Test 2: sidecar file exists alongside .gguf
# ---------------------------------------------------------------------------

class TestSidecarExists:
    def test_sidecar_file_exists(self, model_dir, out_dir):
        result = _export(model_dir, out_dir, passport_id="pp-002")
        sidecar = result.with_suffix(".meta.json")
        assert sidecar.exists(), f"Expected sidecar at {sidecar}"


# ---------------------------------------------------------------------------
# Test 3: sidecar is valid JSON
# ---------------------------------------------------------------------------

class TestSidecarIsValidJson:
    def test_sidecar_parses_as_json(self, model_dir, out_dir):
        result = _export(model_dir, out_dir, passport_id="pp-003")
        sidecar = result.with_suffix(".meta.json")
        content = sidecar.read_text(encoding="utf-8")
        parsed = json.loads(content)  # must not raise
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Test 4: sidecar has passport_id field
# ---------------------------------------------------------------------------

class TestSidecarHasPassportId:
    def test_sidecar_passport_id_matches(self, model_dir, out_dir):
        pid = "pp-constitutional-789"
        result = _export(model_dir, out_dir, passport_id=pid)
        meta = json.loads(result.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert meta.get("dojo.passport_id") == pid

    def test_sidecar_passport_id_is_string(self, model_dir, out_dir):
        result = _export(model_dir, out_dir, passport_id="pp-str-check")
        meta = json.loads(result.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert isinstance(meta["dojo.passport_id"], str)


# ---------------------------------------------------------------------------
# Test 5 & 6: sidecar has hash field and expected schema keys
# ---------------------------------------------------------------------------

class TestSidecarSchemaKeys:
    def test_sidecar_has_sha256_field(self, model_dir, out_dir):
        result = _export(model_dir, out_dir, passport_id="pp-hash")
        meta = json.loads(result.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert "dojo.gguf_sha256" in meta

    def test_sha256_is_64_hex_chars(self, model_dir, out_dir):
        result = _export(model_dir, out_dir, passport_id="pp-hex")
        meta = json.loads(result.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert len(meta["dojo.gguf_sha256"]) == 64

    def test_sidecar_minimum_required_keys_present(self, model_dir, out_dir):
        result = _export(model_dir, out_dir, passport_id="pp-keys")
        meta = json.loads(result.with_suffix(".meta.json").read_text(encoding="utf-8"))
        for key in ("dojo.passport_id", "dojo.gguf_sha256"):
            assert key in meta, f"Missing required sidecar key: {key}"


# ---------------------------------------------------------------------------
# Test 7: SHA256 in sidecar matches actual file hash (round-trip integrity)
# ---------------------------------------------------------------------------

class TestRoundTripIntegrity:
    def test_sidecar_sha256_matches_file(self, model_dir, out_dir):
        result = _export(model_dir, out_dir, passport_id="pp-roundtrip")
        meta = json.loads(result.with_suffix(".meta.json").read_text(encoding="utf-8"))
        actual_hash = _sha256(result)
        assert meta["dojo.gguf_sha256"] == actual_hash, (
            f"Sidecar SHA256 {meta['dojo.gguf_sha256']!r} "
            f"does not match file SHA256 {actual_hash!r}"
        )


# ---------------------------------------------------------------------------
# Test 8: unsupported quant_type raises QuantizationError
# ---------------------------------------------------------------------------

class TestInvalidQuantTypeRaises:
    def test_invalid_quant_type_raises(self, model_dir, out_dir):
        with pytest.raises(QuantizationError):
            _export(model_dir, out_dir, quant_type="INVALID_Q99")

    def test_another_invalid_quant_type_raises(self, model_dir, out_dir):
        with pytest.raises(QuantizationError):
            _export(model_dir, out_dir, quant_type="Q99_ZZZ")


# ---------------------------------------------------------------------------
# Test 9: error message lists at least one valid quant type
# ---------------------------------------------------------------------------

class TestQuantizationErrorMessage:
    def test_error_message_lists_valid_types(self, model_dir, out_dir):
        with pytest.raises(QuantizationError) as exc_info:
            _export(model_dir, out_dir, quant_type="INVALID_Q99")
        msg = str(exc_info.value)
        # At least one known valid type must appear in the message
        assert any(vt in msg for vt in VALID_QUANT_TYPES), (
            f"QuantizationError message should list valid types, got: {msg!r}"
        )

    def test_error_message_includes_invalid_type(self, model_dir, out_dir):
        with pytest.raises(QuantizationError) as exc_info:
            _export(model_dir, out_dir, quant_type="BAD_TYPE_X")
        assert "BAD_TYPE_X" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests 10-12: QUANTIZATION_COMPLETE evidence fields
# ---------------------------------------------------------------------------

class TestQuantizationCompleteEvidence:
    def _get_complete_event(self):
        events = EvidenceLogger.get_events()
        return next((e for e in events if e.get("type") == "quant.complete"), None)

    def test_complete_evidence_has_file_hash(self, model_dir, out_dir):
        _export(model_dir, out_dir, passport_id="pp-ev1")
        ev = self._get_complete_event()
        assert ev is not None
        assert "file_hash" in ev.get("data", {}), "Missing file_hash in QUANTIZATION_COMPLETE"

    def test_complete_evidence_has_output_path(self, model_dir, out_dir):
        _export(model_dir, out_dir, passport_id="pp-ev2")
        ev = self._get_complete_event()
        assert ev is not None
        assert "output_path" in ev.get("data", {}), "Missing output_path in QUANTIZATION_COMPLETE"

    def test_complete_evidence_has_quant_type(self, model_dir, out_dir):
        _export(model_dir, out_dir, quant_type="Q4_K_M", passport_id="pp-ev3")
        ev = self._get_complete_event()
        assert ev is not None
        data = ev.get("data", {})
        assert "quant_type" in data, "Missing quant_type in QUANTIZATION_COMPLETE"
        assert data["quant_type"] == "Q4_K_M"


# ---------------------------------------------------------------------------
# Test 13: consistent sidecar schema across two calls
# ---------------------------------------------------------------------------

class TestConsistentSidecarSchema:
    def test_two_exports_produce_same_schema(self, tmp_path):
        model_a = tmp_path / "ma"
        model_a.mkdir()
        model_b = tmp_path / "mb"
        model_b.mkdir()
        out_a = tmp_path / "oa"
        out_b = tmp_path / "ob"

        res_a = _export(model_a, out_a, passport_id="pp-same")
        res_b = _export(model_b, out_b, passport_id="pp-same")

        meta_a = set(json.loads(res_a.with_suffix(".meta.json").read_text(encoding="utf-8")).keys())
        meta_b = set(json.loads(res_b.with_suffix(".meta.json").read_text(encoding="utf-8")).keys())

        assert meta_a == meta_b, (
            f"Sidecar schema differs between exports: {meta_a} vs {meta_b}"
        )


# ---------------------------------------------------------------------------
# Test 14: passport provenance chain — merkle_root + issued_at from profile
# ---------------------------------------------------------------------------

class TestPassportProvenanceChain:
    def test_sidecar_gets_merkle_root_from_profile(self, model_dir, out_dir):
        from dojo.core.types import HALProfile

        fake_passport = MagicMock()
        fake_passport.merkle_root = "abc123merkle"
        fake_passport.issued_at = "2026-01-01T00:00:00+00:00"

        profile = MagicMock(spec=HALProfile)
        profile.passport = fake_passport
        profile.tier = MagicMock()
        profile.tier.value = "edge"

        result = _export(model_dir, out_dir, passport_id="pp-prov", profile=profile)
        meta = json.loads(result.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert meta.get("dojo.merkle_root") == "abc123merkle"

    def test_sidecar_gets_issued_at_from_profile(self, model_dir, out_dir):
        from dojo.core.types import HALProfile

        fake_passport = MagicMock()
        fake_passport.merkle_root = "rootXYZ"
        fake_passport.issued_at = "2026-05-17T12:00:00+00:00"

        profile = MagicMock(spec=HALProfile)
        profile.passport = fake_passport
        profile.tier = MagicMock()
        profile.tier.value = "edge"

        result = _export(model_dir, out_dir, passport_id="pp-date", profile=profile)
        meta = json.loads(result.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert meta.get("dojo.issued_at") == "2026-05-17T12:00:00+00:00"

    def test_no_crash_when_profile_has_no_passport(self, model_dir, out_dir):
        from dojo.core.types import HALProfile

        profile = MagicMock(spec=HALProfile)
        profile.passport = None
        profile.tier = MagicMock()
        profile.tier.value = "edge"

        # Must not raise
        result = _export(model_dir, out_dir, passport_id="pp-nopassport", profile=profile)
        assert result.exists()


# ---------------------------------------------------------------------------
# Test 15: no sidecar without passport_id (regression guard)
# ---------------------------------------------------------------------------

class TestNoSidecarWithoutPassportId:
    def test_no_sidecar_without_passport_id(self, model_dir, out_dir):
        result = _export(model_dir, out_dir)  # no passport_id
        sidecar = result.with_suffix(".meta.json")
        assert not sidecar.exists(), "Sidecar must NOT be written when passport_id is absent"

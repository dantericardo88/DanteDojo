"""Tests for d24: GGUF binary-format header validator."""
from __future__ import annotations

from pathlib import Path

import pytest

from dojo.quant.gguf_validator import (
    GGUF_MAGIC,
    GgufHeader,
    GgufValidationError,
    build_test_header,
    parse_gguf_header,
    validate_gguf_buffer,
    validate_gguf_file,
)


class TestBuildHeader:
    def test_header_starts_with_magic(self):
        h = build_test_header()
        assert h[:4] == GGUF_MAGIC

    def test_header_is_24_bytes(self):
        assert len(build_test_header()) == 24

    def test_version_encoded(self):
        h = build_test_header(version=3)
        # bytes 4..8 are little-endian uint32
        assert int.from_bytes(h[4:8], "little") == 3

    def test_tensor_count_encoded(self):
        h = build_test_header(tensor_count=42)
        assert int.from_bytes(h[8:16], "little") == 42


class TestParse:
    def test_parses_valid_header(self):
        h = parse_gguf_header(build_test_header(version=3, tensor_count=5))
        assert h.magic == GGUF_MAGIC
        assert h.version == 3
        assert h.tensor_count == 5

    def test_short_buffer_raises(self):
        with pytest.raises(GgufValidationError, match="too short"):
            parse_gguf_header(b"GGUF")

    def test_returns_header_dataclass(self):
        h = parse_gguf_header(build_test_header())
        assert isinstance(h, GgufHeader)

    def test_valid_magic_flag(self):
        h = parse_gguf_header(build_test_header())
        assert h.is_valid_magic

    def test_supported_version_flag(self):
        h = parse_gguf_header(build_test_header(version=3))
        assert h.is_supported_version


class TestValidateBuffer:
    def test_valid_buffer_passes(self):
        h = validate_gguf_buffer(build_test_header())
        assert h.is_valid_magic

    def test_wrong_magic_rejected(self):
        bad = b"WRNG" + build_test_header()[4:]
        with pytest.raises(GgufValidationError, match="bad GGUF magic"):
            validate_gguf_buffer(bad)

    def test_zero_version_rejected(self):
        bad = build_test_header(version=0)
        with pytest.raises(GgufValidationError, match="unsupported"):
            validate_gguf_buffer(bad)

    def test_future_version_rejected(self):
        bad = build_test_header(version=99)
        with pytest.raises(GgufValidationError, match="unsupported"):
            validate_gguf_buffer(bad)

    def test_explicit_version_check_passes(self):
        h = validate_gguf_buffer(build_test_header(version=3), expected_version=3)
        assert h.version == 3

    def test_explicit_version_mismatch_fails(self):
        with pytest.raises(GgufValidationError, match="version mismatch"):
            validate_gguf_buffer(build_test_header(version=2), expected_version=3)

    def test_expected_min_tensors_passes(self):
        h = validate_gguf_buffer(build_test_header(tensor_count=10), expected_min_tensors=5)
        assert h.tensor_count == 10

    def test_expected_min_tensors_fails(self):
        with pytest.raises(GgufValidationError, match="too_few_tensors"):
            validate_gguf_buffer(build_test_header(tensor_count=2), expected_min_tensors=5)


class TestValidateFile:
    def test_validates_file_on_disk(self, tmp_path: Path):
        p = tmp_path / "test.gguf"
        p.write_bytes(build_test_header(version=3, tensor_count=4))
        h = validate_gguf_file(p)
        assert h.version == 3
        assert h.tensor_count == 4

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(GgufValidationError, match="not found"):
            validate_gguf_file(tmp_path / "missing.gguf")

    def test_truncated_file_raises(self, tmp_path: Path):
        p = tmp_path / "short.gguf"
        p.write_bytes(b"GGUF")
        with pytest.raises(GgufValidationError, match="too short"):
            validate_gguf_file(p)


class TestEvidence:
    def test_evidence_emitted_on_validate(self):
        from dojo.core.evidence import EvidenceLogger
        EvidenceLogger.clear()
        validate_gguf_buffer(build_test_header(version=2, tensor_count=7))
        events = EvidenceLogger.get_events()
        assert any(
            e.get("type") == "quant.gguf_validate" and e["data"].get("version") == 2
            for e in events
        )

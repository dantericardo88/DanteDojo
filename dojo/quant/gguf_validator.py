"""Pure-Python GGUF binary-format header validator (d24).

The GGUF format starts with:
  - 4 bytes: magic "GGUF" (0x47, 0x47, 0x55, 0x46)
  - 4 bytes: format version (uint32, little-endian)
  - 8 bytes: tensor_count (uint64)
  - 8 bytes: metadata_kv_count (uint64)

We don't ship a full GGUF parser (the llama.cpp project is the source of
truth) but we can validate the header bytes round-trip the writer's
declared values. This lets quality-gate tests prove that an export
produced a well-formed file without needing llama-cpp installed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import DojoError


GGUF_MAGIC: bytes = b"GGUF"
GGUF_MIN_VERSION: int = 1
GGUF_MAX_KNOWN_VERSION: int = 3   # llama.cpp current as of writing


class GgufValidationError(DojoError):
    """Raised when a GGUF file or buffer fails validation."""


@dataclass(frozen=True)
class GgufHeader:
    """Parsed header from the first 24 bytes of a GGUF file."""

    magic: bytes
    version: int
    tensor_count: int
    metadata_kv_count: int

    @property
    def is_valid_magic(self) -> bool:
        return self.magic == GGUF_MAGIC

    @property
    def is_supported_version(self) -> bool:
        return GGUF_MIN_VERSION <= self.version <= GGUF_MAX_KNOWN_VERSION


_HEADER_STRUCT = struct.Struct("<4sIQQ")
_HEADER_BYTES = _HEADER_STRUCT.size   # 4 + 4 + 8 + 8 = 24


def parse_gguf_header(buf: bytes) -> GgufHeader:
    """Parse the 24-byte GGUF header from ``buf``.

    Raises:
        GgufValidationError: if ``buf`` is shorter than 24 bytes.
    """
    if len(buf) < _HEADER_BYTES:
        raise GgufValidationError(
            f"buffer too short: need {_HEADER_BYTES} bytes, got {len(buf)}",
            reason="short_buffer",
        )

    magic, version, tensor_count, kv_count = _HEADER_STRUCT.unpack_from(buf, 0)
    return GgufHeader(
        magic=magic,
        version=int(version),
        tensor_count=int(tensor_count),
        metadata_kv_count=int(kv_count),
    )


def validate_gguf_buffer(
    buf: bytes,
    expected_version: Optional[int] = None,
    expected_min_tensors: int = 0,
) -> GgufHeader:
    """Validate a GGUF buffer's header.

    Args:
        buf:                  Bytes containing at least the 24-byte GGUF header.
        expected_version:     If set, require this exact version.
        expected_min_tensors: Minimum tensor_count (default 0 = no check).

    Returns:
        The parsed GgufHeader on success.

    Raises:
        GgufValidationError: with a specific reason on any failure.
    """
    header = parse_gguf_header(buf)
    if not header.is_valid_magic:
        raise GgufValidationError(
            f"bad GGUF magic: {header.magic!r} (expected b'GGUF')",
            reason="bad_magic",
        )
    if not header.is_supported_version:
        raise GgufValidationError(
            f"unsupported GGUF version {header.version} "
            f"(supported {GGUF_MIN_VERSION}..{GGUF_MAX_KNOWN_VERSION})",
            reason="bad_version",
        )
    if expected_version is not None and header.version != expected_version:
        raise GgufValidationError(
            f"version mismatch: header={header.version}, expected={expected_version}",
            reason="version_mismatch",
        )
    if expected_min_tensors > 0 and header.tensor_count < expected_min_tensors:
        raise GgufValidationError(
            f"tensor_count {header.tensor_count} < expected_min {expected_min_tensors}",
            reason="too_few_tensors",
        )

    EvidenceLogger.get_instance().emit(
        {
            "type": "quant.gguf_validate",
            "data": {
                "version": header.version,
                "tensor_count": header.tensor_count,
                "metadata_kv_count": header.metadata_kv_count,
            },
        },
        organ="quant",
    )

    return header


def validate_gguf_file(
    path: Path | str,
    expected_version: Optional[int] = None,
    expected_min_tensors: int = 0,
) -> GgufHeader:
    """Validate a GGUF file on disk by reading and parsing its header."""
    p = Path(path)
    if not p.exists():
        raise GgufValidationError(
            f"GGUF file not found: {p}",
            reason="file_not_found",
        )
    with p.open("rb") as f:
        head = f.read(_HEADER_BYTES)
    return validate_gguf_buffer(
        head,
        expected_version=expected_version,
        expected_min_tensors=expected_min_tensors,
    )


def build_test_header(
    version: int = 3,
    tensor_count: int = 1,
    metadata_kv_count: int = 1,
) -> bytes:
    """Synthesise a well-formed GGUF header — useful for tests."""
    return _HEADER_STRUCT.pack(GGUF_MAGIC, version, tensor_count, metadata_kv_count)


__all__ = [
    "GGUF_MAGIC",
    "GGUF_MIN_VERSION",
    "GGUF_MAX_KNOWN_VERSION",
    "GgufHeader",
    "GgufValidationError",
    "parse_gguf_header",
    "validate_gguf_buffer",
    "validate_gguf_file",
    "build_test_header",
]

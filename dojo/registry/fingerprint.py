"""Deterministic model fingerprinting for the DanteDojo registry.

All fingerprints are SHA-256 based and stable across Python versions
because we use sorted-key JSON serialization for artifact dicts and
raw byte reads for weight files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dojo.core.types import ModelArtifact

# Maximum bytes to read from a weights file for fast fingerprinting.
_WEIGHT_READ_LIMIT = 1024 * 1024  # 1 MB


def fingerprint_artifact(artifact: ModelArtifact) -> str:
    """Return a deterministic SHA-256 hex digest for a ModelArtifact.

    The hash is computed over the JSON serialization of artifact.to_dict()
    with keys sorted alphabetically. This means the fingerprint is stable
    as long as the artifact fields do not change.

    Args:
        artifact: The ModelArtifact to fingerprint.

    Returns:
        64-character lowercase hex string (SHA-256).
    """
    payload = json.dumps(artifact.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_weights(weights_path: Path) -> str:
    """Return a SHA-256 hex digest of the first 1 MB of a weights file.

    This is a fast approximation — it does NOT guarantee full file integrity
    but provides a stable identifier for the weights header, which is
    sufficient for the registry's deduplication needs.

    Args:
        weights_path: Path to the weights file (e.g. a .safetensors or .bin file).

    Returns:
        64-character lowercase hex string (SHA-256).

    Raises:
        FileNotFoundError: If weights_path does not exist.
        PermissionError: If the file cannot be read.
    """
    weights_path = Path(weights_path).expanduser().resolve()
    h = hashlib.sha256()
    with weights_path.open("rb") as fh:
        chunk = fh.read(_WEIGHT_READ_LIMIT)
        h.update(chunk)
    return h.hexdigest()


__all__ = [
    "fingerprint_artifact",
    "fingerprint_weights",
]

"""DanteDojo promotion gate — Evidence sealer.

Transplanted concept from STEAK's certification/ultimate/evidence_sealer.

Computes a Merkle root over the three gate receipts, model artifact ID,
and issued_at timestamp. This canonical seal is used for offline verification.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dojo.core.types import ConstitutionalPassport


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_str(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _merkle_root(leaves: list[str]) -> str:
    """Compute a Merkle root over a list of hex-string leaf hashes."""
    if not leaves:
        return hashlib.sha256(b"").hexdigest()

    nodes = list(leaves)
    while len(nodes) > 1:
        next_level: list[str] = []
        for i in range(0, len(nodes), 2):
            left = nodes[i]
            right = nodes[i + 1] if i + 1 < len(nodes) else left
            combined = hashlib.sha256((left + right).encode()).hexdigest()
            next_level.append(combined)
        nodes = next_level
    return nodes[0]


def seal_passport(passport: "ConstitutionalPassport") -> str:
    """Compute and return a Merkle root over the passport's constituent parts.

    Leaves (in order):
    1. SHA-256 of harvest_check_receipt bytes
    2. SHA-256 of dojo_check_receipt bytes
    3. SHA-256 of forge_check_receipt bytes
    4. SHA-256 of model_artifact_id string
    5. SHA-256 of issued_at ISO string

    Returns the Merkle root hex string.
    """
    leaves = [
        _hash_bytes(passport.harvest_check_receipt),
        _hash_bytes(passport.dojo_check_receipt),
        _hash_bytes(passport.forge_check_receipt),
        _hash_str(passport.model_artifact_id),
        _hash_str(passport.issued_at.isoformat()),
    ]
    return _merkle_root(leaves)


def verify_seal(passport: "ConstitutionalPassport") -> bool:
    """Recompute the seal and verify it matches passport.merkle_root."""
    expected = seal_passport(passport)
    return expected == passport.merkle_root

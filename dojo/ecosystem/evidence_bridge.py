"""DanteDojo ecosystem — Cross-organ evidence bridge.

Creates and verifies cross-organ evidence links by hashing the combination
of local and remote Merkle roots with the remote organ identifier.
"""

from __future__ import annotations

import hashlib

from dojo.core.evidence import EvidenceLogger


def cross_link_evidence(
    local_merkle_root: str,
    remote_organ: str,
    remote_merkle_root: str,
) -> str:
    """Create a cross-organ evidence link.

    Computes SHA-256 of (local_merkle_root + remote_organ + remote_merkle_root).
    Emits an evidence event with type "evidence.cross_link".

    Args:
        local_merkle_root: Merkle root of the local evidence chain.
        remote_organ: Identifier of the remote organ (e.g. "harvest", "forge").
        remote_merkle_root: Merkle root of the remote organ's evidence chain.

    Returns:
        Hex string of the cross-link SHA-256 hash.
    """
    payload = local_merkle_root + remote_organ + remote_merkle_root
    link_hash = hashlib.sha256(payload.encode()).hexdigest()

    logger = EvidenceLogger.get_instance()
    logger.emit(
        {
            "type": "evidence.cross_link",
            "data": {
                "local_merkle_root": local_merkle_root,
                "remote_organ": remote_organ,
                "remote_merkle_root": remote_merkle_root,
                "link_hash": link_hash,
            },
        },
        organ="ecosystem",
    )

    return link_hash


def verify_cross_link(
    link_hash: str,
    local_merkle_root: str,
    remote_organ: str,
    remote_merkle_root: str,
) -> bool:
    """Verify a cross-link hash.

    Recomputes the hash from the inputs and compares it to link_hash.

    Args:
        link_hash: The expected cross-link hash (hex string).
        local_merkle_root: Merkle root of the local evidence chain.
        remote_organ: Identifier of the remote organ.
        remote_merkle_root: Merkle root of the remote organ's evidence chain.

    Returns:
        True if the recomputed hash matches link_hash.
    """
    payload = local_merkle_root + remote_organ + remote_merkle_root
    expected = hashlib.sha256(payload.encode()).hexdigest()
    return expected == link_hash

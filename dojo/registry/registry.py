"""One-door model registry entry points for DanteDojo.

All external code interacts with the registry exclusively through
register(), lookup(), list_artifacts(), delete_artifact(), and
registry_integrity_check(). Internal store operations are not exported.
"""

from __future__ import annotations

import fnmatch
from typing import Dict, List, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import RegistryError
from dojo.core.one_door import one_door
from dojo.core.types import ModelArtifact
from dojo.pack.license import check_license_allows_training
from dojo.pack.schema import SpecializationPack
from dojo.registry.fingerprint import fingerprint_artifact
from dojo.registry.store import LocalRegistryStore


def _default_store() -> LocalRegistryStore:
    """Return a default LocalRegistryStore using the standard ~/.dante path."""
    return LocalRegistryStore()


@one_door("registry.register")
def register(
    artifact: ModelArtifact,
    store: Optional[LocalRegistryStore] = None,
    pack: Optional[SpecializationPack] = None,
) -> str:
    """Register a model artifact in the local registry.

    Computes the artifact fingerprint for integrity verification and persists
    the manifest via LocalRegistryStore.

    Args:
        artifact: The ModelArtifact to register.
        store: Optional custom store backend. Defaults to LocalRegistryStore().
        pack: Optional SpecializationPack used to train this artifact.
            If provided, license is checked before registration proceeds.

    Returns:
        The artifact_id of the registered artifact.

    Raises:
        PackLicenseRejected: If pack.license.allows_derived_ft is False.
        RegistryError: If the artifact cannot be persisted.
    """
    if pack is not None:
        check_license_allows_training(pack)

    backend = store if store is not None else _default_store()

    # Compute fingerprint for record-keeping (stored as side-channel evidence)
    fp = fingerprint_artifact(artifact)

    EvidenceLogger.get_instance().emit(
        {
            "type": "registry.register",
            "data": {
                "artifact_id": artifact.artifact_id,
                "fingerprint": fp,
            },
        },
        organ="registry",
    )

    backend.save(artifact)
    return artifact.artifact_id


@one_door("registry.lookup")
def lookup(
    artifact_id: str,
    store: Optional[LocalRegistryStore] = None,
) -> ModelArtifact:
    """Look up a model artifact by ID from the local registry.

    Args:
        artifact_id: The artifact ID to retrieve.
        store: Optional custom store backend. Defaults to LocalRegistryStore().

    Returns:
        The reconstructed ModelArtifact.

    Raises:
        RegistryError: If the artifact is not found or is corrupt.
    """
    backend = store if store is not None else _default_store()
    return backend.load(artifact_id)


@one_door("registry.list_artifacts")
def list_artifacts(
    pattern: str = "*",
    store: Optional[LocalRegistryStore] = None,
) -> List[Dict[str, str]]:
    """List all registered artifacts matching a glob pattern.

    Args:
        pattern: Glob pattern matched against artifact_id (default "*" = all).
        store: Optional custom store backend. Defaults to LocalRegistryStore().

    Returns:
        List of dicts with keys: artifact_id, model_name, version,
        registered_at.  Sorted by artifact_id.

    Raises:
        RegistryError: If the registry directory cannot be read.
    """
    backend = store if store is not None else _default_store()
    entries = backend.list_artifact_details()

    if pattern != "*":
        entries = [e for e in entries if fnmatch.fnmatch(e["artifact_id"], pattern)]

    logger = EvidenceLogger.get_instance()
    logger.emit(
        {
            "type": EvidenceType.EVIDENCE_MERKLE_ROOT.value,
            "data": {
                "operation": "registry.list_artifacts",
                "pattern": pattern,
                "count": len(entries),
            },
        },
        organ="registry",
    )
    return entries


@one_door("registry.delete_artifact")
def delete_artifact(
    artifact_id: str,
    store: Optional[LocalRegistryStore] = None,
) -> bool:
    """Delete an artifact from the registry by ID.

    Args:
        artifact_id: The artifact ID to delete.
        store: Optional custom store backend. Defaults to LocalRegistryStore().

    Returns:
        True on successful deletion.

    Raises:
        RegistryError: If the artifact is not found or cannot be deleted.
    """
    backend = store if store is not None else _default_store()
    backend.delete(artifact_id)

    EvidenceLogger.get_instance().emit(
        {
            "type": EvidenceType.EVIDENCE_MERKLE_ROOT.value,
            "data": {
                "operation": "registry.delete_artifact",
                "artifact_id": artifact_id,
            },
        },
        organ="registry",
    )
    return True


@one_door("registry.integrity_check")
def registry_integrity_check(
    store: Optional[LocalRegistryStore] = None,
) -> Dict[str, int]:
    """Scan all registry artifacts and verify manifest consistency.

    For each artifact manifest on disk, checks that the required fields are
    present and that the stored artifact_id matches the filename.  Reports
    healthy, corrupt, and missing counts.

    Args:
        store: Optional custom store backend. Defaults to LocalRegistryStore().

    Returns:
        Dict with keys: total, healthy, corrupt, missing.

    Raises:
        RegistryError: Only if the registry directory itself cannot be listed.
    """
    backend = store if store is not None else _default_store()
    result = backend.integrity_check()

    EvidenceLogger.get_instance().emit(
        {
            "type": EvidenceType.EVIDENCE_MERKLE_ROOT.value,
            "data": {
                "operation": "registry.integrity_check",
                **result,
            },
        },
        organ="registry",
    )
    return result


__all__ = [
    "register",
    "lookup",
    "list_artifacts",
    "delete_artifact",
    "registry_integrity_check",
]

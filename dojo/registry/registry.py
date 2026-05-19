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


@one_door("registry.lookup_version")
def lookup_version(
    model_name: str,
    version: Optional[str] = None,
    store: Optional[LocalRegistryStore] = None,
) -> ModelArtifact:
    """Look up a model artifact by logical name and optional version tag.

    Resolves ``dojo://model-name:version`` style URIs.  The *model_name* is
    matched against the prefix of each ``artifact_id`` (everything before the
    last ``/``-separated segment).  If *version* is ``None`` or ``"latest"``,
    the artifact with the lexicographically greatest version segment is
    returned (e.g. ``v3`` > ``v2`` > ``v1``).

    Args:
        model_name: Logical model name (e.g. ``"my-model"``).  May include a
            ``dojo://`` prefix and an optional ``:version`` suffix — both are
            stripped before matching.
        version: Specific version string to retrieve (e.g. ``"v2"``).
            ``None`` or ``"latest"`` returns the most recent version.
        store: Optional custom store backend.

    Returns:
        The matching ModelArtifact.

    Raises:
        RegistryError: If no artifact matching name (and version) is found.
    """
    # Strip dojo:// scheme and parse optional :version suffix
    name = model_name
    if name.startswith("dojo://"):
        name = name[len("dojo://"):]
    if ":" in name and version is None:
        name, version = name.rsplit(":", 1)

    backend = store if store is not None else _default_store()
    all_ids = backend.list_artifacts()

    # Match artifacts whose name prefix equals model_name
    matches: list[dict] = []
    for art_id in all_ids:
        # artifact_id may be "model-name/v1" or "dojo://model-name/v1" etc.
        stripped = art_id
        if stripped.startswith("dojo://"):
            stripped = stripped[len("dojo://"):]
        # Split off version segment (last "/"-delimited token)
        if "/" in stripped:
            art_name, art_ver = stripped.rsplit("/", 1)
        else:
            art_name, art_ver = stripped, ""

        if art_name == name:
            matches.append({"artifact_id": art_id, "version": art_ver})

    if not matches:
        raise RegistryError(
            f"No artifacts found for model '{model_name}'",
            reason=f"Searched {len(all_ids)} registry entries; name prefix '{name}' not matched",
        )

    if version is None or version == "latest":
        # Return lexicographically greatest version
        target = max(matches, key=lambda m: m["version"])
    else:
        candidates = [m for m in matches if m["version"] == version]
        if not candidates:
            available = ", ".join(m["version"] for m in matches)
            raise RegistryError(
                f"Version '{version}' of model '{name}' not found",
                reason=f"Available versions: {available}",
            )
        target = candidates[0]

    logger = EvidenceLogger.get_instance()
    logger.emit(
        {
            "type": "registry.lookup_version",
            "data": {
                "model_name": name,
                "requested_version": version,
                "resolved_artifact_id": target["artifact_id"],
                "resolved_version": target["version"],
            },
        },
        organ="registry",
    )

    return backend.load(target["artifact_id"])


@one_door("registry.list_versions")
def list_versions(
    model_name: str,
    store: Optional[LocalRegistryStore] = None,
) -> list:
    """List all registered versions of a named model.

    Args:
        model_name: Logical model name (``dojo://`` prefix is stripped).

    Returns:
        Sorted list of version strings for the given model name.
        Empty list if no matching artifacts exist.
    """
    name = model_name
    if name.startswith("dojo://"):
        name = name[len("dojo://"):]
    if ":" in name:
        name = name.rsplit(":", 1)[0]

    backend = store if store is not None else _default_store()
    all_ids = backend.list_artifacts()

    versions: list[str] = []
    for art_id in all_ids:
        stripped = art_id
        if stripped.startswith("dojo://"):
            stripped = stripped[len("dojo://"):]
        if "/" in stripped:
            art_name, art_ver = stripped.rsplit("/", 1)
        else:
            art_name, art_ver = stripped, ""
        if art_name == name and art_ver:
            versions.append(art_ver)

    return sorted(versions)


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
    "lookup_version",
    "list_versions",
    "list_artifacts",
    "delete_artifact",
    "registry_integrity_check",
]

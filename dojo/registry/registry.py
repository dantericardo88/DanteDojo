"""One-door model registry entry points for DanteDojo.

All external code interacts with the registry exclusively through
register() and lookup(). Internal store operations are not exported.
"""

from __future__ import annotations

from typing import Optional

from dojo.core.one_door import one_door
from dojo.core.types import ModelArtifact
from dojo.registry.fingerprint import fingerprint_artifact
from dojo.registry.store import LocalRegistryStore


def _default_store() -> LocalRegistryStore:
    """Return a default LocalRegistryStore using the standard ~/.dante path."""
    return LocalRegistryStore()


@one_door("registry.register")
def register(
    artifact: ModelArtifact,
    store: Optional[LocalRegistryStore] = None,
) -> str:
    """Register a model artifact in the local registry.

    Computes the artifact fingerprint for integrity verification and persists
    the manifest via LocalRegistryStore.

    Args:
        artifact: The ModelArtifact to register.
        store: Optional custom store backend. Defaults to LocalRegistryStore().

    Returns:
        The artifact_id of the registered artifact.

    Raises:
        RegistryError: If the artifact cannot be persisted.
    """
    backend = store if store is not None else _default_store()

    # Compute fingerprint for record-keeping (stored as side-channel evidence)
    fp = fingerprint_artifact(artifact)

    from dojo.core.evidence import EvidenceLogger
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


__all__ = [
    "register",
    "lookup",
]

"""DanteDojo model registry subsystem — Phase 4."""

from __future__ import annotations

from dojo.registry.fingerprint import fingerprint_artifact, fingerprint_weights
from dojo.registry.registry import (
    delete_artifact,
    list_artifacts,
    lookup,
    register,
    registry_integrity_check,
)
from dojo.registry.store import LocalRegistryStore

__all__ = [
    "LocalRegistryStore",
    "delete_artifact",
    "fingerprint_artifact",
    "fingerprint_weights",
    "list_artifacts",
    "lookup",
    "register",
    "registry_integrity_check",
]

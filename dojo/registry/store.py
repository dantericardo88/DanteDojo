"""Local filesystem registry backend for the DanteDojo model registry.

Artifacts are stored as JSON manifests at:
    <registry_dir>/<artifact_id_safe>.json

Where artifact_id_safe is the artifact_id with "/" and ":" replaced by "_"
to ensure valid filenames on all platforms.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import RegistryError
from dojo.core.types import ModelArtifact, QuantizationSpec


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _artifact_to_manifest(artifact: ModelArtifact) -> Dict[str, Any]:
    """Serialize a ModelArtifact to a JSON-compatible manifest dict."""
    d = artifact.to_dict()
    # Store extra fields needed for reconstruction
    d["_schema_version"] = "1"
    d["_artifact_class"] = "ModelArtifact"
    # Persist quantization details fully
    if artifact.quantization is not None:
        d["quantization"] = artifact.quantization.to_dict()
    return d


def _manifest_to_artifact(manifest: Dict[str, Any]) -> ModelArtifact:
    """Reconstruct a ModelArtifact from its JSON manifest dict.

    Raises:
        RegistryError: If the manifest is missing required fields.
    """
    required = {"artifact_id", "base_model", "training_receipt"}
    missing = required - set(manifest.keys())
    if missing:
        raise RegistryError(
            f"Manifest is missing required fields: {sorted(missing)}",
            reason="Corrupt or incompatible registry manifest",
        )

    quant_dict = manifest.get("quantization")
    quant: Optional[QuantizationSpec] = None
    if quant_dict is not None:
        quant = QuantizationSpec(
            method=quant_dict["method"],
            bits=int(quant_dict["bits"]),
            group_size=quant_dict.get("group_size"),
            calibration_dataset=quant_dict.get("calibration_dataset"),
        )

    lora_raw = manifest.get("lora_weights")
    lora_weights: Optional[Path] = Path(lora_raw) if lora_raw else None

    created_at_raw = manifest.get("created_at")
    if created_at_raw:
        try:
            created_at = datetime.fromisoformat(created_at_raw)
        except Exception:
            created_at = datetime.now(timezone.utc)
    else:
        created_at = datetime.now(timezone.utc)

    return ModelArtifact(
        artifact_id=manifest["artifact_id"],
        base_model=manifest["base_model"],
        lora_weights=lora_weights,
        quantization=quant,
        training_receipt=manifest["training_receipt"],
        passport=None,  # passports are not stored in the registry manifest
        created_at=created_at,
    )


def _safe_filename(artifact_id: str) -> str:
    """Convert an artifact_id to a safe filesystem filename (no slashes or colons)."""
    return re.sub(r"[/:?*<>|\"\\]", "_", artifact_id)


# ---------------------------------------------------------------------------
# LocalRegistryStore
# ---------------------------------------------------------------------------

class LocalRegistryStore:
    """Filesystem-backed model registry that persists ModelArtifact manifests as JSON.

    Each artifact is stored as a single JSON file:
        <registry_dir>/<safe_id>.json

    Args:
        registry_dir: Root directory for storing artifact manifests.
            Defaults to ~/.dante/registry/models/.
    """

    _DEFAULT_DIR = Path("~/.dante/registry/models")

    def __init__(self, registry_dir: Optional[Path] = None) -> None:
        self._dir = Path(registry_dir or self._DEFAULT_DIR).expanduser().resolve()

    def _ensure_dir(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

    def _artifact_path(self, artifact_id: str) -> Path:
        return self._dir / f"{_safe_filename(artifact_id)}.json"

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def save(self, artifact: ModelArtifact) -> None:
        """Write artifact manifest JSON to the registry directory.

        Emits evidence on successful save.

        Args:
            artifact: The ModelArtifact to persist.

        Raises:
            RegistryError: If the file cannot be written.
        """
        self._ensure_dir()
        path = self._artifact_path(artifact.artifact_id)

        try:
            manifest = _artifact_to_manifest(artifact)
            with path.open("w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
        except (OSError, TypeError, ValueError) as exc:
            raise RegistryError(
                f"Failed to save artifact '{artifact.artifact_id}' to registry",
                reason=str(exc),
            ) from exc

        EvidenceLogger.get_instance().emit(
            {
                "type": "registry.artifact.saved",
                "data": {
                    "artifact_id": artifact.artifact_id,
                    "path": str(path),
                },
            },
            organ="registry",
        )

    def load(self, artifact_id: str) -> ModelArtifact:
        """Load a ModelArtifact by its ID from the registry.

        Args:
            artifact_id: The artifact ID to load.

        Returns:
            The reconstructed ModelArtifact.

        Raises:
            RegistryError: If the artifact is not found or the manifest is corrupt.
        """
        path = self._artifact_path(artifact_id)

        if not path.exists():
            raise RegistryError(
                f"Artifact '{artifact_id}' not found in registry",
                reason=f"Expected path: {path}",
            )

        try:
            with path.open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(
                f"Failed to read manifest for artifact '{artifact_id}'",
                reason=str(exc),
            ) from exc

        return _manifest_to_artifact(manifest)

    def list_artifacts(self) -> List[str]:
        """Return all artifact IDs currently in the registry.

        Returns:
            Sorted list of artifact ID strings.
        """
        if not self._dir.exists():
            return []

        artifact_ids: List[str] = []
        for json_file in self._dir.glob("*.json"):
            try:
                with json_file.open("r", encoding="utf-8") as fh:
                    manifest = json.load(fh)
                artifact_id = manifest.get("artifact_id")
                if artifact_id:
                    artifact_ids.append(artifact_id)
            except Exception:
                # Skip corrupt manifests silently
                pass

        return sorted(artifact_ids)

    def delete(self, artifact_id: str) -> None:
        """Remove an artifact manifest from the registry.

        Args:
            artifact_id: The artifact ID to delete.

        Raises:
            RegistryError: If the artifact does not exist.
        """
        path = self._artifact_path(artifact_id)

        if not path.exists():
            raise RegistryError(
                f"Cannot delete artifact '{artifact_id}': not found in registry",
                reason=f"Expected path: {path}",
            )

        try:
            path.unlink()
        except OSError as exc:
            raise RegistryError(
                f"Failed to delete artifact '{artifact_id}' from registry",
                reason=str(exc),
            ) from exc

        EvidenceLogger.get_instance().emit(
            {
                "type": "registry.artifact.deleted",
                "data": {"artifact_id": artifact_id},
            },
            organ="registry",
        )

    def exists(self, artifact_id: str) -> bool:
        """Return True if the artifact is present in the registry.

        Args:
            artifact_id: The artifact ID to check.
        """
        return self._artifact_path(artifact_id).exists()

    def list_artifact_details(self) -> List[Dict[str, Any]]:
        """Return structured metadata for every artifact in the registry.

        Each entry contains:
            artifact_id  — the canonical artifact ID string
            model_name   — base_model field from the manifest
            version      — last path component of artifact_id (after "/") or ""
            registered_at — ISO-8601 created_at from the manifest, or ""

        Corrupt manifests are silently skipped (consistent with list_artifacts).

        Returns:
            List of dicts sorted by artifact_id.
        """
        if not self._dir.exists():
            return []

        entries: List[Dict[str, Any]] = []
        for json_file in self._dir.glob("*.json"):
            try:
                with json_file.open("r", encoding="utf-8") as fh:
                    manifest = json.load(fh)
                artifact_id = manifest.get("artifact_id", "")
                if not artifact_id:
                    continue
                base_model: str = manifest.get("base_model", "")
                # Extract version as the last "/" segment of the artifact_id.
                version = artifact_id.rsplit("/", 1)[-1] if "/" in artifact_id else ""
                registered_at: str = str(manifest.get("created_at", ""))
                entries.append(
                    {
                        "artifact_id": artifact_id,
                        "model_name": base_model,
                        "version": version,
                        "registered_at": registered_at,
                    }
                )
            except Exception:
                pass  # Skip corrupt manifests

        return sorted(entries, key=lambda e: e["artifact_id"])

    def integrity_check(self) -> Dict[str, int]:
        """Scan all manifests and verify structural consistency.

        For each ``.json`` file in the registry directory:
        - **healthy**: manifest loads cleanly and artifact_id is non-empty.
        - **corrupt**: file exists but cannot be parsed or is missing required fields.
        - **missing**: artifact_id present but the expected manifest path does not
          match the actual file (i.e. the stored artifact_id maps to a different
          filename than where it was found).

        Returns:
            Dict with keys ``total``, ``healthy``, ``corrupt``, ``missing``.
        """
        if not self._dir.exists():
            return {"total": 0, "healthy": 0, "corrupt": 0, "missing": 0}

        total = healthy = corrupt = missing = 0

        for json_file in self._dir.glob("*.json"):
            total += 1
            try:
                with json_file.open("r", encoding="utf-8") as fh:
                    manifest = json.load(fh)
                artifact_id = manifest.get("artifact_id", "")
                if not artifact_id:
                    corrupt += 1
                    continue
                # Verify the canonical path for this artifact_id matches the file.
                expected_path = self._artifact_path(artifact_id)
                if expected_path.resolve() != json_file.resolve():
                    missing += 1
                else:
                    healthy += 1
            except Exception:
                corrupt += 1

        return {
            "total": total,
            "healthy": healthy,
            "corrupt": corrupt,
            "missing": missing,
        }


__all__ = [
    "LocalRegistryStore",
]

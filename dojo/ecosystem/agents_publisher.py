"""DanteDojo ecosystem — DanteAgents manifest publisher.

Writes model manifests to a registry directory that DanteAgents polls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from dojo.core.evidence import EvidenceLogger

if TYPE_CHECKING:
    from dojo.core.types import ConstitutionalPassport, ModelArtifact

_DEFAULT_REGISTRY_DIR = Path("~/.dante/registry/models")


class AgentsPublisher:
    """Publishes model manifests to the DanteAgents polling directory.

    Args:
        registry_dir: Path to the directory where manifests are written.
                      DanteAgents polls this directory for new models.
                      Defaults to ~/.dante/registry/models.
    """

    def __init__(self, registry_dir: Path = _DEFAULT_REGISTRY_DIR) -> None:
        self._registry_dir = Path(registry_dir).expanduser().resolve()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def publish_model(
        self,
        artifact: "ModelArtifact",
        passport: "ConstitutionalPassport",
    ) -> Path:
        """Write {artifact_id}.manifest.json to registry_dir.

        DanteAgents polls this directory for new certified models.

        Args:
            artifact: The model artifact to publish.
            passport: The constitutional passport authorizing deployment.

        Returns:
            Path to the manifest file written.

        Raises:
            RegistryError: On any filesystem failure.
        """
        from dojo.core.exceptions import RegistryError  # noqa: PLC0415

        logger = EvidenceLogger.get_instance()

        try:
            self._registry_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RegistryError(
                f"Failed to create registry directory '{self._registry_dir}': {exc}",
                reason="mkdir_failed",
            ) from exc

        # Sanitize artifact_id for use as a filename
        safe_id = artifact.artifact_id.replace("://", "_").replace("/", "_").replace(":", "_")
        manifest_path = self._registry_dir / f"{safe_id}.manifest.json"

        manifest = {
            "artifact": artifact.to_dict(),
            "passport": passport.to_dict(),
            "registry_dir": str(self._registry_dir),
        }

        try:
            with manifest_path.open("w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
        except OSError as exc:
            raise RegistryError(
                f"Failed to write manifest to '{manifest_path}': {exc}",
                reason="write_failed",
            ) from exc

        logger.emit(
            {
                "type": "agents.model.published",
                "data": {
                    "artifact_id": artifact.artifact_id,
                    "passport_id": passport.passport_id,
                    "manifest_path": str(manifest_path),
                },
            },
            organ="ecosystem",
        )

        return manifest_path

    def list_published(self) -> list[str]:
        """Return list of published artifact IDs.

        Reads all *.manifest.json files from registry_dir and returns
        the artifact_id from each.
        """
        if not self._registry_dir.exists():
            return []

        artifact_ids: list[str] = []
        for manifest_file in sorted(self._registry_dir.glob("*.manifest.json")):
            try:
                with manifest_file.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                artifact_id = data.get("artifact", {}).get("artifact_id", "")
                if artifact_id:
                    artifact_ids.append(artifact_id)
            except (OSError, json.JSONDecodeError):
                # Skip corrupt manifests — don't fail the whole listing
                continue

        return artifact_ids

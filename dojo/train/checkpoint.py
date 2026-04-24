"""DanteDojo checkpoint manager — save, load, list, and prune checkpoints.

Checkpoints are stored as JSON manifest files on disk.  The manager enforces
a maximum checkpoint count (oldest-first pruning) and emits evidence on every
save and prune operation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType


class CheckpointManager:
    """Manages fine-tuning checkpoints in *checkpoint_dir*.

    At most *max_checkpoints* checkpoints are kept; older ones are pruned
    automatically after each save.
    """

    def __init__(self, checkpoint_dir: Path, max_checkpoints: int = 3) -> None:
        self._dir = checkpoint_dir
        self._max = max_checkpoints
        self._logger = EvidenceLogger.get_instance()
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, step: int, metadata: dict) -> Path:  # type: ignore[type-arg]
        """Save a checkpoint manifest JSON for *step*.

        The manifest file is named ``checkpoint_{step:08d}.json`` so that
        lexicographic and numeric orderings agree.  Prunes old checkpoints
        after saving.

        Returns the path of the written manifest.
        """
        manifest_path = self._dir / f"checkpoint_{step:08d}.json"
        manifest = {"step": step, **metadata}
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")

        self._logger.emit(
            {
                "type": EvidenceType.TRAINING_CHECKPOINT.value,
                "data": {
                    "step": step,
                    "path": str(manifest_path),
                    "metadata_keys": list(metadata.keys()),
                },
            },
            organ="train",
        )

        self.prune()
        return manifest_path

    def load_latest(self) -> Optional[tuple[int, dict]]:  # type: ignore[type-arg]
        """Return ``(step, metadata)`` for the most recent checkpoint, or ``None``."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        step, path = checkpoints[-1]
        raw = json.loads(path.read_text(encoding="utf-8"))
        step_from_file = raw.pop("step", step)
        return int(step_from_file), raw

    def list_checkpoints(self) -> list[tuple[int, Path]]:
        """Return checkpoints sorted by step number (ascending)."""
        result: list[tuple[int, Path]] = []
        for p in self._dir.glob("checkpoint_*.json"):
            try:
                step = int(p.stem.split("_", 1)[1])
                result.append((step, p))
            except (ValueError, IndexError):
                continue
        result.sort(key=lambda t: t[0])
        return result

    def prune(self) -> None:
        """Remove the oldest checkpoints when more than *max_checkpoints* exist."""
        checkpoints = self.list_checkpoints()
        excess = checkpoints[: max(0, len(checkpoints) - self._max)]
        for step, path in excess:
            path.unlink(missing_ok=True)
            self._logger.emit(
                {
                    "type": "train.checkpoint.pruned",
                    "data": {"step": step, "path": str(path)},
                },
                organ="train",
            )


__all__ = ["CheckpointManager"]

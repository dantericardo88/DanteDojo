"""DanteDojo gradient accumulator — idempotent idle-training recovery.

Tracks which micro-batches have been processed within the current optimizer
step so that if idle training is interrupted mid-accumulation, recovery can
resume exactly from the right position without double-counting gradients.
"""

from __future__ import annotations

import json
from pathlib import Path


class GradientAccumulator:
    """Track micro-batch accumulation state for crash-safe idle training.

    When idle training is interrupted (thermal pause, battery disconnect,
    foreground activity), the accumulator state is persisted to JSON.  On
    restart, ``load_state()`` restores the exact count so the training loop
    knows how many micro-batches still need processing before taking an
    optimizer step.

    Usage::

        accum = GradientAccumulator(accum_steps=8)
        for batch in dataloader:
            should_step = accum.record_microbatch()
            loss = model(batch) / accum_steps
            loss.backward()
            if should_step:
                optimizer.step()
                optimizer.zero_grad()
                accum.optimizer_step_taken()
    """

    def __init__(self, accum_steps: int) -> None:
        if accum_steps < 1:
            raise ValueError(f"accum_steps must be >= 1, got {accum_steps}")
        self._accum_steps = accum_steps
        self._accumulated: int = 0
        self._current_step: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_step(self) -> int:
        """Number of completed optimizer steps (each step = accum_steps micro-batches)."""
        return self._current_step

    @property
    def accumulated(self) -> int:
        """Number of micro-batches accumulated in the current (in-progress) step."""
        return self._accumulated

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_microbatch(self) -> bool:
        """Record one processed micro-batch.

        Returns ``True`` when the caller should take an optimizer step
        (i.e., when ``accumulated == accum_steps`` after this micro-batch).
        """
        self._accumulated += 1
        return self._accumulated >= self._accum_steps

    def optimizer_step_taken(self) -> None:
        """Confirm that the optimizer step was taken and reset the accumulation counter."""
        self._current_step += 1
        self._accumulated = 0

    def restore_state(self, current_step: int, accumulated: int = 0) -> None:
        """Restore counters from a checkpoint manifest."""
        if current_step < 0:
            raise ValueError(f"current_step must be >= 0, got {current_step}")
        if not 0 <= accumulated < self._accum_steps:
            raise ValueError(
                f"accumulated must be in [0, {self._accum_steps}), got {accumulated}"
            )
        self._current_step = current_step
        self._accumulated = accumulated

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: Path) -> None:
        """Persist accumulator state to JSON at *path* for crash recovery."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "accum_steps": self._accum_steps,
            "accumulated": self._accumulated,
            "current_step": self._current_step,
        }
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def load_state(self, path: Path) -> None:
        """Restore accumulator state from JSON at *path*.

        Raises ``FileNotFoundError`` if the file does not exist.
        Raises ``ValueError`` if the stored ``accum_steps`` does not match.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        stored_steps = int(raw["accum_steps"])
        if stored_steps != self._accum_steps:
            raise ValueError(
                f"Stored accum_steps={stored_steps} does not match "
                f"current accum_steps={self._accum_steps}"
            )
        self._accumulated = int(raw["accumulated"])
        self._current_step = int(raw["current_step"])


__all__ = ["GradientAccumulator"]

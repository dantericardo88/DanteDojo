"""Real fixture-based prompt loading for the idle-time GPU daemon (d20).

The idle daemon's default step runner used synthetic loss curves.  This module
provides a real loader that reads prompts from a JSONL fixture, a parquet
file, or an in-memory list — so idle training steps consume genuine
specialization-pack content instead of seeded synthetic numbers.

Designed for the daemon to call once per epoch:

    loader = FixtureBasedPromptLoader.from_jsonl("pack/train.jsonl")
    for batch in loader.batches(size=4):
        ...

No torch/transformers dependency — the loader simply yields dicts ready for
the trainer or step runner to consume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Sequence

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import DojoError


class PromptLoaderError(DojoError):
    """Raised when prompt fixture loading fails."""


_REQUIRED_KEYS: tuple[str, ...] = ("prompt",)


@dataclass
class FixtureBasedPromptLoader:
    """Iterable loader for training prompts backed by a real fixture.

    Attributes:
        prompts: list of dicts; each must contain at least a ``"prompt"`` key.
        source: human-readable origin of the prompts (e.g. a file path).
    """

    prompts: List[dict]
    source: str = "in_memory"
    _emitted_load_evidence: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        for i, p in enumerate(self.prompts):
            if not isinstance(p, dict):
                raise PromptLoaderError(
                    f"prompt at index {i} is not a dict",
                    reason=f"got {type(p).__name__}",
                )
            for k in _REQUIRED_KEYS:
                if k not in p:
                    raise PromptLoaderError(
                        f"prompt at index {i} missing required key: {k}",
                        reason=f"keys={list(p.keys())}",
                    )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_jsonl(cls, path: Path | str) -> "FixtureBasedPromptLoader":
        """Load prompts from a JSONL file (one prompt dict per line)."""
        p = Path(path)
        if not p.exists():
            raise PromptLoaderError(
                f"JSONL fixture not found: {p}",
                reason="file_not_found",
            )
        prompts: List[dict] = []
        for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                prompts.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise PromptLoaderError(
                    f"invalid JSON at {p}:{line_no}",
                    reason=str(exc),
                ) from exc
        return cls(prompts=prompts, source=str(p))

    @classmethod
    def from_list(cls, items: Sequence[dict]) -> "FixtureBasedPromptLoader":
        """Wrap an in-memory list of prompt dicts."""
        return cls(prompts=list(items), source="in_memory")

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.prompts)

    def __iter__(self) -> Iterator[dict]:
        if not self._emitted_load_evidence:
            self._emit_load_evidence()
        for p in self.prompts:
            yield p

    def __getitem__(self, idx: int) -> dict:
        return self.prompts[idx]

    def batches(self, size: int) -> Iterator[List[dict]]:
        """Yield prompts in batches of the requested size."""
        if size <= 0:
            raise PromptLoaderError(
                "batch size must be positive",
                reason=f"size={size}",
            )
        if not self._emitted_load_evidence:
            self._emit_load_evidence()
        for i in range(0, len(self.prompts), size):
            yield self.prompts[i:i + size]

    # ------------------------------------------------------------------
    # Stats / introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return summary stats: count, mean prompt length, max prompt length."""
        if not self.prompts:
            return {"count": 0, "mean_chars": 0, "max_chars": 0, "source": self.source}
        lengths = [len(p["prompt"]) for p in self.prompts]
        return {
            "count": len(self.prompts),
            "mean_chars": sum(lengths) // len(lengths),
            "max_chars": max(lengths),
            "source": self.source,
        }

    def _emit_load_evidence(self) -> None:
        self._emitted_load_evidence = True
        EvidenceLogger.get_instance().emit(
            {
                "type": EvidenceType.IDLE_TRAINING_RESUMED.value,
                "data": {
                    "phase": "fixture_loaded",
                    "source": self.source,
                    "prompt_count": len(self.prompts),
                },
            },
            organ="idle",
        )


__all__ = [
    "FixtureBasedPromptLoader",
    "PromptLoaderError",
]

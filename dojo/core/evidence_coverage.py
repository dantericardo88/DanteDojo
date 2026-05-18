"""Evidence-emission coverage audit (d04).

The inspector flagged d04 with "not all helpers log".  This module records
which operations have actually emitted evidence in a given execution scope
and produces a coverage report.  Tests can then assert that every critical
operation surfaced an event.

Two layers:
  - ``EvidenceCoverage`` — context manager that snapshots ``EvidenceLogger``
    before / after a code block and reports which event types fired.
  - ``assert_operations_emitted`` — convenience that raises if any expected
    event type was missing.

Pure introspection — does not modify how evidence is emitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import DojoError


class EvidenceCoverageError(DojoError):
    """Raised when expected evidence emissions are missing."""


@dataclass
class EvidenceCoverage:
    """Context manager that records evidence emissions inside a block.

    Usage:

        with EvidenceCoverage() as cov:
            do_some_work()

        if not cov.contains("train.start"):
            ...

    Or as a coverage gate:

        with EvidenceCoverage() as cov:
            do_some_work()
        cov.assert_emitted(["train.start", "train.checkpoint", "train.complete"])
    """

    _starting_count: int = field(init=False, default=0)
    _emitted: List[Dict] = field(init=False, default_factory=list)

    def __enter__(self) -> "EvidenceCoverage":
        events = EvidenceLogger.get_events()
        self._starting_count = len(events)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        all_events = EvidenceLogger.get_events()
        self._emitted = all_events[self._starting_count:]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def event_types(self) -> Set[str]:
        """Set of distinct event ``type`` strings emitted inside the block."""
        return {e.get("type", "") for e in self._emitted if e.get("type")}

    def count_of(self, event_type: str) -> int:
        return sum(1 for e in self._emitted if e.get("type") == event_type)

    def contains(self, event_type: str) -> bool:
        return self.count_of(event_type) > 0

    def total_emitted(self) -> int:
        return len(self._emitted)

    def missing(self, expected: Iterable[str]) -> List[str]:
        """Return expected event types that were *not* emitted."""
        emitted = self.event_types()
        return [e for e in expected if e not in emitted]

    def report(self) -> Dict[str, object]:
        """Structured coverage report — emitted counts per event type."""
        return {
            "total_emitted": self.total_emitted(),
            "distinct_types": len(self.event_types()),
            "counts": {t: self.count_of(t) for t in sorted(self.event_types())},
        }

    def assert_emitted(self, expected: Iterable[str]) -> None:
        """Raise if any expected event type is missing.

        Raises:
            EvidenceCoverageError: with the list of missing event types.
        """
        missing = self.missing(expected)
        if missing:
            raise EvidenceCoverageError(
                f"missing required evidence emissions: {missing}",
                reason=f"missing_count={len(missing)}",
            )


def assert_operations_emitted(
    operation: callable,  # type: ignore[valid-type]
    expected_types: Iterable[str],
    *args,
    **kwargs,
):
    """Run ``operation`` and assert it emits all of ``expected_types``.

    Returns the operation's return value on success.
    """
    with EvidenceCoverage() as cov:
        result = operation(*args, **kwargs)
    cov.assert_emitted(expected_types)
    return result


__all__ = [
    "EvidenceCoverage",
    "EvidenceCoverageError",
    "assert_operations_emitted",
]

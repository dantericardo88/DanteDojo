"""Judge drift detector for DanteDojo evaluation.

Tracks agreement between a judge LLM and gold labels over a rolling window.
If agreement falls below MIN_AGREEMENT_THRESHOLD the model is considered
"drifted" and JudgeDriftError is raised to block further evaluation.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import List, NamedTuple, Optional

from dojo.core.exceptions import JudgeDriftError


class _AgreementRecord(NamedTuple):
    """Single agreement check record with a UTC timestamp."""

    matched: bool
    recorded_at: datetime


class JudgeDriftDetector:
    """Sliding-window judge-vs-gold agreement tracker.

    Thread-safe. All records older than window_days are automatically pruned
    on each read to keep memory bounded.

    Args:
        judge_family: The model family string (e.g. "claude-sonnet-4-6").
        window_days: Number of days to retain records in the sliding window.
    """

    MIN_AGREEMENT_THRESHOLD: float = 0.80  # 80% agreement with gold required

    def __init__(self, judge_family: str, window_days: int = 30) -> None:
        self._judge_family = judge_family
        self._window_days = window_days
        self._records: List[_AgreementRecord] = []
        self._lock = threading.Lock()

    def record_agreement(self, judge_output: str, gold_label: str) -> bool:
        """Record whether the judge matched the gold label.

        Comparison is case-insensitive and strips surrounding whitespace.

        Args:
            judge_output: The string output produced by the judge LLM.
            gold_label: The authoritative gold-standard label.

        Returns:
            True if judge_output matches gold_label (case-insensitive, stripped).
        """
        matched = judge_output.strip().lower() == gold_label.strip().lower()
        record = _AgreementRecord(matched=matched, recorded_at=datetime.now(timezone.utc))
        with self._lock:
            self._records.append(record)
        return matched

    def _active_records(self) -> List[_AgreementRecord]:
        """Return only records within the current window (thread-unsafe, call under lock)."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._window_days)
        return [r for r in self._records if r.recorded_at >= cutoff]

    def agreement_rate(self) -> float:
        """Return the fraction of agreements in the current sliding window.

        Returns:
            Float in [0.0, 1.0]. Returns 1.0 if no records exist yet.
        """
        with self._lock:
            active = self._active_records()

        if not active:
            return 1.0  # no data yet — optimistic default

        matched = sum(1 for r in active if r.matched)
        return matched / len(active)

    def check_drift(self) -> None:
        """Raise JudgeDriftError if agreement_rate < MIN_AGREEMENT_THRESHOLD.

        This is a blocking check — callers must not proceed with evaluation
        if this raises.

        Raises:
            JudgeDriftError: If the agreement rate is below threshold.
        """
        rate = self.agreement_rate()
        if rate < self.MIN_AGREEMENT_THRESHOLD:
            raise JudgeDriftError(
                f"Judge '{self._judge_family}' has drifted: "
                f"agreement_rate={rate:.3f} < threshold={self.MIN_AGREEMENT_THRESHOLD}",
                reason=(
                    f"Judge LLM outputs disagree with gold labels too frequently. "
                    f"Rate: {rate:.1%}. Minimum required: {self.MIN_AGREEMENT_THRESHOLD:.0%}."
                ),
            )

    def is_same_family(self, candidate_model: str, judge_family: str) -> bool:
        """Return True if candidate_model belongs to the same model family as judge_family.

        Used to enforce the frozen-judge rule: the candidate under evaluation
        must NOT be the same model family as the judge (evaluation bias).

        Family matching is done by checking if either string is a case-insensitive
        substring of the other, or if they share the same family prefix.

        Args:
            candidate_model: The model being evaluated (e.g. "claude-sonnet-4-6-tuned").
            judge_family: The judge model family string (e.g. "claude-sonnet-4-6").

        Returns:
            True if they appear to be the same family (potential violation).
        """
        c = candidate_model.strip().lower()
        j = judge_family.strip().lower()

        if not c or not j:
            return False

        # Direct match
        if c == j:
            return True

        # Substring containment — either direction
        if c in j or j in c:
            return True

        # Prefix match: split on common delimiters and compare first token
        def _prefix(s: str) -> str:
            for sep in ("-", "_", "/", ":"):
                if sep in s:
                    return s.split(sep)[0]
            return s

        return _prefix(c) == _prefix(j)


__all__ = [
    "JudgeDriftDetector",
]

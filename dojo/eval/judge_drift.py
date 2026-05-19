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


class MultiJudgeDriftDetector:
    """Pairwise agreement tracker for 3+ distinct named judge models.

    Each call to record_pairwise() records the output of every judge on the
    same prompt + gold label.  check_pairwise_drift() then computes agreement
    for every (judge_a, judge_b) pair and raises JudgeDriftError if any pair
    falls below min_agreement (default 80%).

    Design rationale: single-judge drift only catches drift relative to gold.
    Pairwise drift catches cases where one judge drifts away from the other
    two while all still technically pass the gold threshold — a stronger
    guarantee of judge stability.

    Args:
        judge_families: List of at least 2 judge model name strings.
        min_agreement: Pairwise agreement floor (default 0.80 = 80%).
    """

    def __init__(
        self,
        judge_families: list,
        min_agreement: float = 0.80,
    ) -> None:
        if len(judge_families) < 2:
            raise ValueError("MultiJudgeDriftDetector requires at least 2 judge families")
        if len(set(judge_families)) != len(judge_families):
            raise ValueError("MultiJudgeDriftDetector requires distinct judge families")
        self._families: list = list(judge_families)
        self._min_agreement = min_agreement
        # per-judge individual drift trackers (gold agreement)
        self._detectors: dict = {
            f: JudgeDriftDetector(f) for f in judge_families
        }
        # pairwise records: list of {"outputs": {family: output}, "gold": str}
        self._records: list = []
        self._lock = threading.Lock()

    @property
    def judge_families(self) -> list:
        return list(self._families)

    def record_pairwise(
        self,
        judge_outputs: dict,
        gold_label: str,
    ) -> None:
        """Record one evaluation round with outputs from all judges.

        Args:
            judge_outputs: Mapping of judge_family → output string.
                All families registered at construction must be present.
            gold_label: Authoritative gold-standard label for this prompt.
        """
        for family in self._families:
            output = judge_outputs.get(family, "")
            self._detectors[family].record_agreement(output, gold_label)

        with self._lock:
            self._records.append(
                {"outputs": dict(judge_outputs), "gold": gold_label}
            )

    def agreement_rates(self) -> dict:
        """Return per-judge gold-agreement rates and all pairwise rates.

        Returns:
            Dict with:
              "individual": {family: rate}
              "pairwise": {(family_a, family_b): rate} — all unordered pairs
        """
        individual = {
            f: self._detectors[f].agreement_rate() for f in self._families
        }
        pairwise: dict = {}
        for i in range(len(self._families)):
            for j in range(i + 1, len(self._families)):
                a, b = self._families[i], self._families[j]
                pairwise[(a, b)] = self._compute_pair_rate(a, b)
        return {"individual": individual, "pairwise": pairwise}

    def check_pairwise_drift(self) -> None:
        """Raise JudgeDriftError if any pairwise agreement < min_agreement.

        Checks every unordered (judge_a, judge_b) pair.  Also checks each
        judge's individual gold-agreement rate.

        Raises:
            JudgeDriftError: If any pair or individual judge is below threshold.
        """
        # Individual gold-agreement check
        for family, detector in self._detectors.items():
            detector.check_drift()

        # Pairwise inter-judge agreement check
        for i in range(len(self._families)):
            for j in range(i + 1, len(self._families)):
                a, b = self._families[i], self._families[j]
                rate = self._compute_pair_rate(a, b)
                if rate < self._min_agreement:
                    raise JudgeDriftError(
                        f"Pairwise judge drift: '{a}' vs '{b}' agreed "
                        f"{rate:.3f} ({rate:.1%}) — below {self._min_agreement:.0%} threshold",
                        reason=(
                            f"Inter-judge agreement between '{a}' and '{b}' "
                            f"is {rate:.1%}, minimum required: "
                            f"{self._min_agreement:.0%}."
                        ),
                    )

    def _compute_pair_rate(self, family_a: str, family_b: str) -> float:
        """Fraction of rounds where judge_a and judge_b gave the same output."""
        with self._lock:
            records = list(self._records)
        if not records:
            return 1.0  # optimistic default when no data yet
        agree = sum(
            1
            for r in records
            if (
                r["outputs"].get(family_a, "").strip().lower()
                == r["outputs"].get(family_b, "").strip().lower()
            )
        )
        return agree / len(records)


__all__ = [
    "JudgeDriftDetector",
    "MultiJudgeDriftDetector",
]

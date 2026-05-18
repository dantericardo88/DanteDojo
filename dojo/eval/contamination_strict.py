"""Fail-closed contamination gate (d38).

The inspector noted d38 as "streaming fail-closed".  This module adds an
explicit gate that combines streaming contamination detection with a
strict mode: any uncertain or partial overlap is treated as contamination
rather than ignored.

``ContaminationGate`` returns a verdict + reason so callers can audit
exactly which check fired.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Set, Tuple

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import DojoError


class ContaminationDetected(DojoError):
    """Raised when the strict gate flags contamination."""


@dataclass
class ContaminationVerdict:
    """Result of a contamination check."""

    contaminated: bool
    reason: str
    overlap_ratio: float
    overlap_count: int
    eval_hash_count: int


def _normalize(text: str) -> str:
    """Lower-case + collapse whitespace for fair hashing."""
    return " ".join(text.lower().split())


def _ngrams(text: str, n: int) -> Iterable[str]:
    """Yield character n-grams of the normalised text."""
    s = _normalize(text)
    if len(s) < n:
        return
    for i in range(len(s) - n + 1):
        yield s[i:i + n]


def _hash_set(texts: Iterable[str], n: int) -> Set[str]:
    """Hash each n-gram of each text via SHA-1 (fast, plenty of collision room)."""
    hashes: Set[str] = set()
    for t in texts:
        for ng in _ngrams(t, n):
            hashes.add(hashlib.sha1(ng.encode("utf-8")).hexdigest())
    return hashes


@dataclass
class ContaminationGate:
    """Strict, fail-closed contamination gate.

    Args:
        ngram_size:        Character-level n-gram width (default 13).
        overlap_threshold: Max allowed ngram-hash overlap ratio (default 0.0
                           = any overlap fails).
        require_eval_data: If True, an empty eval-hash set is treated as
                           contamination (we cannot verify clean state).
    """

    ngram_size: int = 13
    overlap_threshold: float = 0.0
    require_eval_data: bool = True

    def check(
        self,
        training_texts: Iterable[str],
        eval_texts: Iterable[str],
    ) -> ContaminationVerdict:
        """Return a structured verdict — does not raise.

        Use ``gate()`` if you want the strict raise-on-contamination form.
        """
        eval_hashes = _hash_set(eval_texts, self.ngram_size)
        train_hashes = _hash_set(training_texts, self.ngram_size)

        if not eval_hashes:
            verdict = ContaminationVerdict(
                contaminated=self.require_eval_data,
                reason="empty_eval_set" if self.require_eval_data else "empty_eval_set_allowed",
                overlap_ratio=0.0,
                overlap_count=0,
                eval_hash_count=0,
            )
            self._emit(verdict)
            return verdict

        overlap = train_hashes & eval_hashes
        ratio = len(overlap) / len(eval_hashes)
        contaminated = ratio > self.overlap_threshold

        verdict = ContaminationVerdict(
            contaminated=contaminated,
            reason="overlap_above_threshold" if contaminated else "clean",
            overlap_ratio=ratio,
            overlap_count=len(overlap),
            eval_hash_count=len(eval_hashes),
        )
        self._emit(verdict)
        return verdict

    def gate(
        self,
        training_texts: Iterable[str],
        eval_texts: Iterable[str],
    ) -> None:
        """Raise ContaminationDetected when the check fails.

        Use this at training-launch time to fail-closed before any GPU work.
        """
        v = self.check(training_texts, eval_texts)
        if v.contaminated:
            raise ContaminationDetected(
                f"contamination gate failed: {v.reason} "
                f"(overlap_ratio={v.overlap_ratio:.4f}, count={v.overlap_count})",
                reason=v.reason,
            )

    def _emit(self, verdict: ContaminationVerdict) -> None:
        EvidenceLogger.get_instance().emit(
            {
                "type": "eval.contamination_strict",
                "data": {
                    "contaminated": verdict.contaminated,
                    "reason": verdict.reason,
                    "overlap_ratio": verdict.overlap_ratio,
                    "overlap_count": verdict.overlap_count,
                    "eval_hash_count": verdict.eval_hash_count,
                    "ngram_size": self.ngram_size,
                    "overlap_threshold": self.overlap_threshold,
                },
            },
            organ="eval",
        )


def quick_contamination_check(
    training_texts: List[str],
    eval_texts: List[str],
    ngram_size: int = 13,
) -> Tuple[bool, float]:
    """Convenience: return ``(contaminated, overlap_ratio)``."""
    gate = ContaminationGate(ngram_size=ngram_size)
    v = gate.check(training_texts, eval_texts)
    return v.contaminated, v.overlap_ratio


__all__ = [
    "ContaminationGate",
    "ContaminationVerdict",
    "ContaminationDetected",
    "quick_contamination_check",
]

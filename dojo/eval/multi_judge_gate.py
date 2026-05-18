"""Multi-judge agreement gate with cross-judge drift detection (d39).

Wraps the existing ``JudgeConsensus`` with a per-judge agreement matrix so
the gate can distinguish four cases:

  - all judges PASS  → PASS, high agreement
  - all judges FAIL  → FAIL, high agreement (still rejected)
  - majority + tie   → result + warning (agreement below threshold)
  - drift detected   → flagged for human review (per-judge variance > limit)

This is the integration layer the d39 inspector flagged as missing: existing
``JudgeConsensus`` had multi-judge mechanics but only a single drift detector
upstream.  ``MultiJudgeGate.evaluate()`` produces a structured verdict with
all the agreement signals attached.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import DojoError


class MultiJudgeGateError(DojoError):
    """Raised when the multi-judge gate cannot produce a verdict."""


# Each judge is a callable: (response, criterion) -> score in [0.0, 10.0]
JudgeFn = Callable[[str, str], float]


@dataclass
class MultiJudgeGate:
    """Cross-judge agreement gate.

    Args:
        judges:           List of (name, fn) judge tuples.  At least 2 required
                          for meaningful agreement signals.
        pass_threshold:   Mean score per criterion >= this passes (default 5.0).
        agreement_min:    Required fraction of judges that must agree on PASS/FAIL
                          for the verdict to be "confident" (default 0.67 = 2/3).
        drift_score_std:  If per-judge stdev for any criterion exceeds this
                          threshold, flag the result as "drifted" (default 2.5).
    """

    judges: List[tuple]  # list of (name, JudgeFn)
    pass_threshold: float = 5.0
    agreement_min: float = 0.67
    drift_score_std: float = 2.5

    def __post_init__(self) -> None:
        if len(self.judges) < 2:
            raise MultiJudgeGateError(
                "MultiJudgeGate requires at least 2 judges for agreement signals",
                reason=f"got {len(self.judges)} judges",
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, response: str, criteria: List[str]) -> Dict[str, Any]:
        """Run all judges across all criteria and return a structured verdict.

        Returns a dict with:
          - ``per_criterion``: { criterion: {scores: [...], mean, stdev, passes} }
          - ``per_judge``:     { judge_name: {pass_count, fail_count} }
          - ``overall_pass``:  bool — all criteria mean >= pass_threshold
          - ``agreement``:     float in [0, 1] — fraction of judges agreeing
                               with the overall verdict
          - ``confident``:     bool — agreement >= agreement_min
          - ``drifted``:       bool — any criterion's stdev > drift_score_std
          - ``flagged``:       bool — not confident OR drifted (review needed)
        """
        if not criteria:
            raise MultiJudgeGateError(
                "criteria list must be non-empty",
                reason="got empty criteria",
            )

        per_criterion: Dict[str, Dict[str, Any]] = {}
        per_judge: Dict[str, Dict[str, int]] = {
            name: {"pass_count": 0, "fail_count": 0} for name, _ in self.judges
        }

        for c in criteria:
            scores = [float(fn(response, c)) for _name, fn in self.judges]
            mean_score = statistics.mean(scores)
            stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
            passes = mean_score >= self.pass_threshold

            per_criterion[c] = {
                "scores": scores,
                "mean": mean_score,
                "stdev": stdev,
                "passes": passes,
            }

            for (name, _fn), s in zip(self.judges, scores):
                if s >= self.pass_threshold:
                    per_judge[name]["pass_count"] += 1
                else:
                    per_judge[name]["fail_count"] += 1

        overall_pass = all(v["passes"] for v in per_criterion.values())

        # Per-judge overall verdict (pass iff judge passed every criterion)
        n_criteria = len(criteria)
        judge_verdicts: List[bool] = [
            per_judge[name]["pass_count"] == n_criteria for name, _ in self.judges
        ]
        agreeing = sum(1 for v in judge_verdicts if v == overall_pass)
        agreement = agreeing / len(self.judges)

        confident = agreement >= self.agreement_min
        drifted = any(v["stdev"] > self.drift_score_std for v in per_criterion.values())
        flagged = (not confident) or drifted

        verdict: Dict[str, Any] = {
            "per_criterion": per_criterion,
            "per_judge": per_judge,
            "overall_pass": overall_pass,
            "agreement": agreement,
            "confident": confident,
            "drifted": drifted,
            "flagged": flagged,
        }

        EvidenceLogger.get_instance().emit(
            {
                "type": "eval.multi_judge_gate",
                "data": {
                    "overall_pass": overall_pass,
                    "agreement": agreement,
                    "confident": confident,
                    "drifted": drifted,
                    "flagged": flagged,
                    "judge_count": len(self.judges),
                    "criterion_count": n_criteria,
                },
            },
            organ="eval",
        )

        return verdict

    def gate(self, response: str, criteria: List[str]) -> bool:
        """Convenience: return True iff overall_pass AND confident AND not drifted."""
        v = self.evaluate(response, criteria)
        return bool(v["overall_pass"] and v["confident"] and not v["drifted"])


__all__ = [
    "MultiJudgeGate",
    "MultiJudgeGateError",
]

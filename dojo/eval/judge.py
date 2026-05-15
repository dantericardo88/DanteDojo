"""Multi-judge consensus protocol for DanteDojo — d41.

Provides:
- JudgeVerdict: Pass/Fail string constants.
- JudgeConsensus: Multi-judge evaluation with tiebreaker protocol.
- evaluate_with_judges(): One-door entry for judge evaluation.

Design:
- Each judge scores each criterion on [0, 10].
- A criterion passes if its mean score >= pass_threshold (default: 5.0).
- A response PASSES if all criteria pass; FAILs if any criterion fails.
- With 2+ judges, if exactly half pass and half fail: a tiebreaker judge is invoked.
- Emits eval.judge_disagreement when any judge disagrees with the majority.
- Emits eval.judge_verdict on every evaluation with consensus fraction.
- Raises EvalJudgeRejection when all judges agree on FAIL.
- Single judge mode works normally (no tiebreaker needed).
- Empty criteria list → vacuous pass (all criteria pass trivially).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import EvalJudgeRejection


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class JudgeVerdict:
    PASS = "pass"
    FAIL = "fail"


# ---------------------------------------------------------------------------
# Judge callable type
# ---------------------------------------------------------------------------

# A Judge is any callable that takes (response: str, criterion: str) and
# returns a numeric score in [0, 10] (int or float accepted).
Judge = Callable[[str, str], Any]


# ---------------------------------------------------------------------------
# JudgeConsensus
# ---------------------------------------------------------------------------

@dataclass
class JudgeConsensus:
    """Multi-judge consensus evaluator.

    Args:
        judges: List of judge callables. Must be non-empty.
        pass_threshold: Minimum mean score per criterion to pass (default 5.0).
    """

    judges: List[Judge]
    pass_threshold: float = 5.0

    def __post_init__(self) -> None:
        if not self.judges:
            raise ValueError("JudgeConsensus requires at least one judge")

    def evaluate(self, response: str, criteria: List[str]) -> Dict[str, Any]:
        """Evaluate *response* against all *criteria* using all judges.

        Protocol:
        1. Each judge scores each criterion [0, 10].
        2. For each criterion: mean score >= pass_threshold → pass, else fail.
        3. Overall verdict: PASS only if all criteria pass.
        4. If there are at least 2 judges and exactly half agree (tie): the
           last judge in the list acts as tiebreaker for the whole response.
        5. Emit eval.judge_disagreement when any judge disagrees.
        6. Emit eval.judge_verdict with final decision.
        7. Raise EvalJudgeRejection if all judges agree on FAIL.

        Args:
            response: The model response string to evaluate.
            criteria: List of criterion names/descriptions.

        Returns:
            Dict with keys:
                - "verdict": JudgeVerdict.PASS or JudgeVerdict.FAIL
                - "consensus": fraction of judges that agreed with final verdict
                - "criteria": list of criterion result dicts
                - "judge_count": total number of judges used
                - "tiebreaker_invoked": bool
                - "judge_consensus": alias for "consensus"

        Raises:
            EvalJudgeRejection: if all judges unanimously vote FAIL.
        """
        logger = EvidenceLogger.get_instance()

        # --- Empty criteria: vacuous pass ---
        if not criteria:
            result: Dict[str, Any] = {
                "verdict": JudgeVerdict.PASS,
                "consensus": 1.0,
                "criteria": [],
                "judge_count": len(self.judges),
                "tiebreaker_invoked": False,
                "judge_consensus": 1.0,
            }
            logger.emit(
                {
                    "type": "eval.judge_verdict",
                    "data": {
                        "verdict": JudgeVerdict.PASS,
                        "consensus": 1.0,
                        "criteria": [],
                        "judge_count": len(self.judges),
                        "tiebreaker_invoked": False,
                    },
                },
                organ="eval",
            )
            return result

        n_judges = len(self.judges)

        # --- Collect per-judge verdicts ---
        # judge_verdicts[j] = JudgeVerdict for judge index j
        judge_verdicts: List[str] = []
        # judge_criteria_scores[j][c] = float score for judge j on criterion c
        judge_criteria_scores: List[Dict[str, float]] = []

        for judge in self.judges:
            scores: Dict[str, float] = {}
            for criterion in criteria:
                try:
                    raw = float(judge(response, criterion))
                    # Clamp to [0, 10]
                    scores[criterion] = max(0.0, min(10.0, raw))
                except Exception:
                    scores[criterion] = 0.0
            judge_criteria_scores.append(scores)

            # Judge verdict: PASS only if all criteria meet threshold
            verdict = JudgeVerdict.PASS
            for criterion in criteria:
                if scores[criterion] < self.pass_threshold:
                    verdict = JudgeVerdict.FAIL
                    break
            judge_verdicts.append(verdict)

        # --- Determine majority verdict ---
        n_pass = sum(1 for v in judge_verdicts if v == JudgeVerdict.PASS)
        n_fail = n_judges - n_pass
        tiebreaker_invoked = False

        if n_judges >= 2 and n_pass == n_fail:
            # Tie: invoke the last judge as tiebreaker
            tiebreaker_invoked = True
            tb_judge = self.judges[-1]
            tb_scores: Dict[str, float] = {}
            for criterion in criteria:
                try:
                    raw = float(tb_judge(response, criterion))
                    tb_scores[criterion] = max(0.0, min(10.0, raw))
                except Exception:
                    tb_scores[criterion] = 0.0

            tb_verdict = JudgeVerdict.PASS
            for criterion in criteria:
                if tb_scores[criterion] < self.pass_threshold:
                    tb_verdict = JudgeVerdict.FAIL
                    break
            final_verdict = tb_verdict
            # Recompute consensus using tiebreaker verdict as reference
            n_agree = sum(1 for v in judge_verdicts if v == final_verdict)
            # Include tiebreaker itself
            consensus = (n_agree + 1) / (n_judges + 1)
        else:
            final_verdict = JudgeVerdict.PASS if n_pass > n_fail else JudgeVerdict.FAIL
            n_agree = n_pass if final_verdict == JudgeVerdict.PASS else n_fail
            consensus = n_agree / n_judges

        # --- Detect disagreement (any judge differs from final verdict) ---
        disagreeing_judges = [
            i for i, v in enumerate(judge_verdicts) if v != final_verdict
        ]
        if disagreeing_judges:
            # Find the two sides
            judge_a_idx = next(
                (i for i, v in enumerate(judge_verdicts) if v == JudgeVerdict.PASS),
                None,
            )
            judge_b_idx = next(
                (i for i, v in enumerate(judge_verdicts) if v == JudgeVerdict.FAIL),
                None,
            )
            logger.emit(
                {
                    "type": "eval.judge_disagreement",
                    "data": {
                        "judge_a_verdict": JudgeVerdict.PASS if judge_a_idx is not None else None,
                        "judge_b_verdict": JudgeVerdict.FAIL if judge_b_idx is not None else None,
                        "criteria": criteria,
                        "tiebreaker_invoked": tiebreaker_invoked,
                        "n_pass": n_pass,
                        "n_fail": n_fail,
                    },
                },
                organ="eval",
            )

        # --- Build criteria result dicts ---
        criteria_results: List[Dict[str, Any]] = []
        for criterion in criteria:
            scores_for_criterion = [
                judge_criteria_scores[j][criterion] for j in range(n_judges)
            ]
            mean_score = sum(scores_for_criterion) / n_judges
            criteria_results.append(
                {
                    "criterion": criterion,
                    "mean_score": mean_score,
                    "scores": scores_for_criterion,
                    "passed": mean_score >= self.pass_threshold,
                }
            )

        # --- Emit verdict evidence ---
        logger.emit(
            {
                "type": "eval.judge_verdict",
                "data": {
                    "verdict": final_verdict,
                    "consensus": consensus,
                    "criteria": criteria,
                    "judge_count": n_judges,
                    "tiebreaker_invoked": tiebreaker_invoked,
                },
            },
            organ="eval",
        )

        # --- Raise on unanimous FAIL ---
        if final_verdict == JudgeVerdict.FAIL and n_pass == 0 and not tiebreaker_invoked:
            raise EvalJudgeRejection(
                "All judges unanimously rejected the response",
                reason=(
                    f"All {n_judges} judge(s) returned FAIL for criteria: "
                    + ", ".join(criteria)
                ),
            )
        if tiebreaker_invoked and final_verdict == JudgeVerdict.FAIL:
            # Tiebreaker cast the deciding FAIL vote — treat as rejection
            raise EvalJudgeRejection(
                "Tiebreaker judge rejected the response",
                reason=(
                    f"Tiebreaker (last judge) returned FAIL after {n_judges} judge(s) tied"
                ),
            )

        return {
            "verdict": final_verdict,
            "consensus": consensus,
            "criteria": criteria,
            "judge_count": n_judges,
            "tiebreaker_invoked": tiebreaker_invoked,
            "judge_consensus": consensus,
            "criteria_results": criteria_results,
        }


# ---------------------------------------------------------------------------
# One-door entry
# ---------------------------------------------------------------------------

def evaluate_with_judges(
    response: str,
    criteria: List[str],
    judges: List[Judge],
    *,
    pass_threshold: float = 5.0,
) -> Dict[str, Any]:
    """One-door entry for judge-based evaluation.

    Args:
        response: The model response to evaluate.
        criteria: List of criterion strings.
        judges: List of judge callables — each takes (response, criterion) → float.
        pass_threshold: Score threshold for a criterion to pass.

    Returns:
        Result dict from JudgeConsensus.evaluate().

    Raises:
        ValueError: If judges list is empty.
        EvalJudgeRejection: If all judges unanimously reject the response.
    """
    if not judges:
        raise ValueError("evaluate_with_judges requires at least one judge")
    consensus = JudgeConsensus(judges=judges, pass_threshold=pass_threshold)
    return consensus.evaluate(response, criteria)


__all__ = [
    "JudgeVerdict",
    "JudgeConsensus",
    "evaluate_with_judges",
]

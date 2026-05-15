"""Tests for d41 — Eval Judge Pipeline (multi-judge consensus protocol).

Covers all 16 required scenarios:
1.  Single judge pass → {"verdict": "pass"}
2.  Single judge fail → {"verdict": "fail"} and raises EvalJudgeRejection
3.  Two judges both pass → consensus verdict pass
4.  Two judges both fail → raises EvalJudgeRejection
5.  Two judges disagree → tiebreaker is invoked
6.  Tiebreaker agrees with judge A (pass) → A's verdict wins
7.  Tiebreaker agrees with judge B (fail) → B's verdict wins
8.  eval.judge_disagreement evidence emitted when judges disagree
9.  eval.judge_verdict evidence emitted on every evaluation
10. Evidence includes consensus score (fraction of judges agreeing)
11. Evidence includes criteria list
12. Receipt has judge_consensus field
13. Zero judges raises ValueError
14. Empty criteria list still evaluates (vacuous pass)
15. Score threshold: all criteria >= threshold → pass; any below → fail
16. Multi-criteria with mixed scores uses mean/majority correctly
"""

from __future__ import annotations

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import EvalJudgeRejection
from dojo.eval.judge import JudgeConsensus, JudgeVerdict, evaluate_with_judges


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ---------------------------------------------------------------------------
# Judge factories
# ---------------------------------------------------------------------------

def _always_pass(score: float = 8.0):
    """Return a judge that always gives *score* (>= 5.0 → pass)."""
    def judge(_response: str, _criterion: str) -> float:
        return score
    return judge


def _always_fail(score: float = 2.0):
    """Return a judge that always gives *score* (< 5.0 → fail)."""
    def judge(_response: str, _criterion: str) -> float:
        return score
    return judge


def _score_map(mapping: dict):
    """Return a judge whose score depends on criterion name."""
    def judge(_response: str, criterion: str) -> float:
        return mapping.get(criterion, 5.0)
    return judge


# ---------------------------------------------------------------------------
# Test 1: Single judge pass
# ---------------------------------------------------------------------------

class TestSingleJudge:
    def test_single_judge_pass(self):
        """Single judge that passes → verdict = 'pass'."""
        result = evaluate_with_judges(
            response="good answer",
            criteria=["accuracy"],
            judges=[_always_pass()],
        )
        assert result["verdict"] == JudgeVerdict.PASS

    # Test 2: Single judge fail
    def test_single_judge_fail_raises(self):
        """Single judge that fails → EvalJudgeRejection raised."""
        with pytest.raises(EvalJudgeRejection):
            evaluate_with_judges(
                response="bad answer",
                criteria=["accuracy"],
                judges=[_always_fail()],
            )

    def test_single_judge_fail_verdict_before_raise(self):
        """The eval.judge_verdict evidence shows 'fail' before exception."""
        with pytest.raises(EvalJudgeRejection):
            evaluate_with_judges(
                response="bad answer",
                criteria=["accuracy"],
                judges=[_always_fail()],
            )
        events = EvidenceLogger.get_events()
        verdict_events = [e for e in events if e["type"] == "eval.judge_verdict"]
        assert any(e["data"]["verdict"] == JudgeVerdict.FAIL for e in verdict_events)


# ---------------------------------------------------------------------------
# Tests 3 & 4: Two judges unanimous
# ---------------------------------------------------------------------------

class TestTwoJudgesUnanimous:
    def test_two_judges_both_pass(self):
        """Two judges both pass → consensus verdict pass."""
        result = evaluate_with_judges(
            response="great answer",
            criteria=["accuracy"],
            judges=[_always_pass(), _always_pass()],
        )
        assert result["verdict"] == JudgeVerdict.PASS
        assert result["judge_count"] == 2

    def test_two_judges_both_fail_raises(self):
        """Two judges both fail → EvalJudgeRejection raised."""
        with pytest.raises(EvalJudgeRejection):
            evaluate_with_judges(
                response="terrible answer",
                criteria=["accuracy"],
                judges=[_always_fail(), _always_fail()],
            )

    def test_two_judges_both_pass_no_disagreement_evidence(self):
        """Two judges agree → no eval.judge_disagreement evidence emitted."""
        evaluate_with_judges(
            response="great answer",
            criteria=["accuracy"],
            judges=[_always_pass(), _always_pass()],
        )
        events = EvidenceLogger.get_events()
        disagreement_events = [e for e in events if e["type"] == "eval.judge_disagreement"]
        assert len(disagreement_events) == 0


# ---------------------------------------------------------------------------
# Tests 5, 6, 7: Two judges disagree → tiebreaker
# ---------------------------------------------------------------------------

class TestTiebreakerProtocol:
    def test_two_judges_disagree_tiebreaker_invoked(self):
        """Two judges disagree (1 pass, 1 fail) → tiebreaker_invoked = True."""
        # Last judge = tiebreaker; make it pass
        result = evaluate_with_judges(
            response="ok answer",
            criteria=["accuracy"],
            judges=[_always_fail(), _always_pass()],  # fail, pass → tie; tiebreaker = last = pass
        )
        assert result["tiebreaker_invoked"] is True

    def test_tiebreaker_agrees_with_pass_judge(self):
        """Tiebreaker (last judge) is a passer → final verdict = pass."""
        # judges[0]=fail, judges[1]=pass (also acts as tiebreaker)
        result = evaluate_with_judges(
            response="ok answer",
            criteria=["accuracy"],
            judges=[_always_fail(), _always_pass()],
        )
        assert result["verdict"] == JudgeVerdict.PASS

    def test_tiebreaker_agrees_with_fail_judge(self):
        """Tiebreaker (last judge) is a failer → final verdict = fail."""
        # judges[0]=pass, judges[1]=fail (also acts as tiebreaker)
        with pytest.raises(EvalJudgeRejection):
            evaluate_with_judges(
                response="ok answer",
                criteria=["accuracy"],
                judges=[_always_pass(), _always_fail()],
            )


# ---------------------------------------------------------------------------
# Test 8: eval.judge_disagreement evidence emitted on disagreement
# ---------------------------------------------------------------------------

class TestDisagreementEvidence:
    def test_disagreement_evidence_emitted(self):
        """eval.judge_disagreement evidence emitted when judges disagree."""
        try:
            evaluate_with_judges(
                response="mediocre",
                criteria=["accuracy"],
                judges=[_always_fail(), _always_pass()],
            )
        except EvalJudgeRejection:
            pass

        events = EvidenceLogger.get_events()
        disagreement_events = [e for e in events if e["type"] == "eval.judge_disagreement"]
        assert len(disagreement_events) >= 1

    def test_disagreement_evidence_has_judge_verdicts(self):
        """eval.judge_disagreement has judge_a_verdict and judge_b_verdict fields."""
        try:
            evaluate_with_judges(
                response="mediocre",
                criteria=["accuracy"],
                judges=[_always_fail(), _always_pass()],
            )
        except EvalJudgeRejection:
            pass

        events = EvidenceLogger.get_events()
        disagreement_events = [e for e in events if e["type"] == "eval.judge_disagreement"]
        assert len(disagreement_events) >= 1
        d = disagreement_events[0]["data"]
        assert "judge_a_verdict" in d
        assert "judge_b_verdict" in d

    def test_disagreement_evidence_has_criteria(self):
        """eval.judge_disagreement evidence includes criteria list."""
        try:
            evaluate_with_judges(
                response="mediocre",
                criteria=["accuracy", "relevance"],
                judges=[_always_fail(), _always_pass()],
            )
        except EvalJudgeRejection:
            pass

        events = EvidenceLogger.get_events()
        disagreement_events = [e for e in events if e["type"] == "eval.judge_disagreement"]
        assert len(disagreement_events) >= 1
        criteria_in_evidence = disagreement_events[0]["data"]["criteria"]
        assert "accuracy" in criteria_in_evidence
        assert "relevance" in criteria_in_evidence


# ---------------------------------------------------------------------------
# Test 9: eval.judge_verdict emitted on every evaluation
# ---------------------------------------------------------------------------

class TestVerdictEvidence:
    def test_verdict_evidence_emitted_on_pass(self):
        """eval.judge_verdict evidence emitted when verdict is pass."""
        evaluate_with_judges(
            response="great",
            criteria=["accuracy"],
            judges=[_always_pass()],
        )
        events = EvidenceLogger.get_events()
        verdict_events = [e for e in events if e["type"] == "eval.judge_verdict"]
        assert len(verdict_events) >= 1

    def test_verdict_evidence_emitted_on_fail(self):
        """eval.judge_verdict evidence emitted even when EvalJudgeRejection raised."""
        with pytest.raises(EvalJudgeRejection):
            evaluate_with_judges(
                response="bad",
                criteria=["accuracy"],
                judges=[_always_fail()],
            )
        events = EvidenceLogger.get_events()
        verdict_events = [e for e in events if e["type"] == "eval.judge_verdict"]
        assert len(verdict_events) >= 1

    def test_verdict_evidence_emitted_for_empty_criteria(self):
        """eval.judge_verdict evidence emitted even for empty criteria."""
        evaluate_with_judges(
            response="any",
            criteria=[],
            judges=[_always_pass()],
        )
        events = EvidenceLogger.get_events()
        verdict_events = [e for e in events if e["type"] == "eval.judge_verdict"]
        assert len(verdict_events) >= 1


# ---------------------------------------------------------------------------
# Test 10: Evidence includes consensus score
# ---------------------------------------------------------------------------

class TestConsensusScore:
    def test_consensus_score_in_evidence(self):
        """eval.judge_verdict evidence has consensus field as a fraction."""
        evaluate_with_judges(
            response="good",
            criteria=["accuracy"],
            judges=[_always_pass(), _always_pass()],
        )
        events = EvidenceLogger.get_events()
        verdict_events = [e for e in events if e["type"] == "eval.judge_verdict"]
        assert len(verdict_events) >= 1
        consensus = verdict_events[-1]["data"]["consensus"]
        assert isinstance(consensus, float)
        assert 0.0 <= consensus <= 1.0

    def test_full_consensus_is_1_0(self):
        """Two judges both pass → consensus = 1.0."""
        evaluate_with_judges(
            response="good",
            criteria=["accuracy"],
            judges=[_always_pass(), _always_pass()],
        )
        events = EvidenceLogger.get_events()
        verdict_events = [e for e in events if e["type"] == "eval.judge_verdict"]
        assert verdict_events[-1]["data"]["consensus"] == 1.0

    def test_result_has_judge_consensus_field(self):
        """Result dict contains judge_consensus field (alias for consensus)."""
        result = evaluate_with_judges(
            response="good",
            criteria=["accuracy"],
            judges=[_always_pass(), _always_pass()],
        )
        assert "judge_consensus" in result
        assert result["judge_consensus"] == result["consensus"]


# ---------------------------------------------------------------------------
# Test 11: Evidence includes criteria list
# ---------------------------------------------------------------------------

class TestCriteriaInEvidence:
    def test_criteria_in_verdict_evidence(self):
        """eval.judge_verdict evidence includes criteria list."""
        evaluate_with_judges(
            response="answer",
            criteria=["accuracy", "fluency", "relevance"],
            judges=[_always_pass()],
        )
        events = EvidenceLogger.get_events()
        verdict_events = [e for e in events if e["type"] == "eval.judge_verdict"]
        assert len(verdict_events) >= 1
        criteria_in_evidence = verdict_events[-1]["data"]["criteria"]
        for c in ["accuracy", "fluency", "relevance"]:
            assert c in criteria_in_evidence


# ---------------------------------------------------------------------------
# Test 12: Receipt has judge_consensus field
# ---------------------------------------------------------------------------

class TestReceiptFields:
    def test_result_has_judge_consensus(self):
        """Result dict (the 'receipt') contains judge_consensus field."""
        result = evaluate_with_judges(
            response="good answer",
            criteria=["accuracy"],
            judges=[_always_pass()],
        )
        assert "judge_consensus" in result
        assert isinstance(result["judge_consensus"], float)

    def test_result_has_all_required_fields(self):
        """Result dict has verdict, consensus, criteria, judge_count."""
        result = evaluate_with_judges(
            response="answer",
            criteria=["accuracy"],
            judges=[_always_pass()],
        )
        for key in ("verdict", "consensus", "criteria", "judge_count", "judge_consensus"):
            assert key in result, f"Missing field: {key}"


# ---------------------------------------------------------------------------
# Test 13: Zero judges raises ValueError
# ---------------------------------------------------------------------------

class TestZeroJudges:
    def test_zero_judges_raises_value_error(self):
        """evaluate_with_judges([]) raises ValueError."""
        with pytest.raises(ValueError):
            evaluate_with_judges(
                response="answer",
                criteria=["accuracy"],
                judges=[],
            )

    def test_judge_consensus_zero_judges_raises(self):
        """JudgeConsensus([]) raises ValueError at construction."""
        with pytest.raises(ValueError):
            JudgeConsensus(judges=[])


# ---------------------------------------------------------------------------
# Test 14: Empty criteria list → vacuous pass
# ---------------------------------------------------------------------------

class TestEmptyCriteria:
    def test_empty_criteria_returns_pass(self):
        """Empty criteria list → vacuous pass, no exception."""
        result = evaluate_with_judges(
            response="any response",
            criteria=[],
            judges=[_always_fail()],  # even a failing judge
        )
        assert result["verdict"] == JudgeVerdict.PASS

    def test_empty_criteria_consensus_is_1_0(self):
        """Empty criteria → consensus = 1.0 (vacuous agreement)."""
        result = evaluate_with_judges(
            response="any response",
            criteria=[],
            judges=[_always_fail()],
        )
        assert result["consensus"] == 1.0


# ---------------------------------------------------------------------------
# Test 15: Score threshold
# ---------------------------------------------------------------------------

class TestScoreThreshold:
    def test_all_criteria_above_threshold_pass(self):
        """Scores >= threshold → pass."""
        result = evaluate_with_judges(
            response="answer",
            criteria=["accuracy", "fluency"],
            judges=[_always_pass(score=7.0)],
            pass_threshold=5.0,
        )
        assert result["verdict"] == JudgeVerdict.PASS

    def test_any_criterion_below_threshold_fail(self):
        """Any criterion score < threshold → fail."""
        # accuracy=8.0 (pass), fluency=3.0 (fail)
        with pytest.raises(EvalJudgeRejection):
            evaluate_with_judges(
                response="answer",
                criteria=["accuracy", "fluency"],
                judges=[_score_map({"accuracy": 8.0, "fluency": 3.0})],
                pass_threshold=5.0,
            )

    def test_custom_threshold_respected(self):
        """Score=6.0 passes threshold=5.0 but fails threshold=7.0."""
        result = evaluate_with_judges(
            response="answer",
            criteria=["accuracy"],
            judges=[_always_pass(score=6.0)],
            pass_threshold=5.0,
        )
        assert result["verdict"] == JudgeVerdict.PASS

        with pytest.raises(EvalJudgeRejection):
            evaluate_with_judges(
                response="answer",
                criteria=["accuracy"],
                judges=[_always_pass(score=6.0)],
                pass_threshold=7.0,
            )

    def test_exact_threshold_score_passes(self):
        """Score == threshold exactly → pass (>= comparison)."""
        result = evaluate_with_judges(
            response="answer",
            criteria=["accuracy"],
            judges=[_always_pass(score=5.0)],
            pass_threshold=5.0,
        )
        assert result["verdict"] == JudgeVerdict.PASS


# ---------------------------------------------------------------------------
# Test 16: Multi-criteria with mixed scores
# ---------------------------------------------------------------------------

class TestMultiCriteriaMixedScores:
    def test_multi_criteria_all_pass(self):
        """Three criteria all passing → overall pass."""
        result = evaluate_with_judges(
            response="answer",
            criteria=["accuracy", "fluency", "relevance"],
            judges=[_score_map({"accuracy": 8.0, "fluency": 7.0, "relevance": 9.0})],
        )
        assert result["verdict"] == JudgeVerdict.PASS

    def test_multi_criteria_one_fails(self):
        """Three criteria, one below threshold → overall fail."""
        with pytest.raises(EvalJudgeRejection):
            evaluate_with_judges(
                response="answer",
                criteria=["accuracy", "fluency", "relevance"],
                judges=[_score_map({"accuracy": 8.0, "fluency": 2.0, "relevance": 9.0})],
            )

    def test_multi_criteria_two_judges_disagree_on_criterion(self):
        """Two judges with different per-criterion scores trigger tiebreaker protocol.

        Judge A: accuracy=8.0 >= 5.0 → PASS
        Judge B: accuracy=2.0 < 5.0  → FAIL
        Tie → tiebreaker = last judge (B) → FAIL → EvalJudgeRejection raised.
        """
        with pytest.raises(EvalJudgeRejection):
            evaluate_with_judges(
                response="answer",
                criteria=["accuracy"],
                judges=[
                    _score_map({"accuracy": 8.0}),
                    _score_map({"accuracy": 2.0}),
                ],
                pass_threshold=5.0,
            )
        # Also verify that disagreement evidence was emitted
        events = EvidenceLogger.get_events()
        disagree_events = [e for e in events if e["type"] == "eval.judge_disagreement"]
        assert len(disagree_events) >= 1

    def test_criteria_results_in_return_value(self):
        """Result contains criteria_results list with per-criterion detail."""
        result = evaluate_with_judges(
            response="answer",
            criteria=["accuracy", "fluency"],
            judges=[_score_map({"accuracy": 8.0, "fluency": 7.0})],
        )
        assert "criteria_results" in result
        criteria_names = [c["criterion"] for c in result["criteria_results"]]
        assert "accuracy" in criteria_names
        assert "fluency" in criteria_names

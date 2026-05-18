"""Tests for d39: MultiJudgeGate cross-judge agreement + drift detection.

Verifies the 4 verdict states (all-pass, all-fail, majority-tie, drift),
agreement scoring, and evidence emission.
"""
from __future__ import annotations

import pytest

from dojo.eval.multi_judge_gate import MultiJudgeGate, MultiJudgeGateError


def _always(score: float):
    """Build a judge function that always returns ``score``."""
    return lambda _resp, _crit: score


def _per_criterion(score_map: dict):
    """Judge that returns a different score per criterion."""
    return lambda _resp, crit: score_map.get(crit, 5.0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_requires_at_least_two_judges(self):
        with pytest.raises(MultiJudgeGateError, match="at least 2"):
            MultiJudgeGate(judges=[("only", _always(7.0))])

    def test_two_judges_ok(self):
        g = MultiJudgeGate(judges=[
            ("j1", _always(7.0)),
            ("j2", _always(8.0)),
        ])
        assert g is not None


# ---------------------------------------------------------------------------
# All-pass case
# ---------------------------------------------------------------------------

class TestAllPass:
    def test_all_pass_verdict_overall_pass(self):
        g = MultiJudgeGate(judges=[
            ("j1", _always(8.0)),
            ("j2", _always(9.0)),
            ("j3", _always(7.5)),
        ])
        v = g.evaluate("response", ["c1", "c2"])
        assert v["overall_pass"] is True

    def test_all_pass_high_agreement(self):
        g = MultiJudgeGate(judges=[
            ("j1", _always(8.0)),
            ("j2", _always(9.0)),
        ])
        v = g.evaluate("response", ["c1"])
        assert v["agreement"] == 1.0

    def test_all_pass_is_confident(self):
        g = MultiJudgeGate(judges=[
            ("j1", _always(8.0)),
            ("j2", _always(9.0)),
        ])
        v = g.evaluate("response", ["c1"])
        assert v["confident"] is True

    def test_all_pass_not_flagged(self):
        g = MultiJudgeGate(judges=[
            ("j1", _always(8.0)),
            ("j2", _always(9.0)),
        ])
        v = g.evaluate("response", ["c1"])
        assert v["flagged"] is False


# ---------------------------------------------------------------------------
# All-fail case
# ---------------------------------------------------------------------------

class TestAllFail:
    def test_all_fail_verdict_overall_fail(self):
        g = MultiJudgeGate(judges=[
            ("j1", _always(2.0)),
            ("j2", _always(3.0)),
        ])
        v = g.evaluate("response", ["c1"])
        assert v["overall_pass"] is False

    def test_all_fail_high_agreement(self):
        g = MultiJudgeGate(judges=[
            ("j1", _always(2.0)),
            ("j2", _always(3.0)),
        ])
        v = g.evaluate("response", ["c1"])
        assert v["agreement"] == 1.0


# ---------------------------------------------------------------------------
# Mixed / disagreement case
# ---------------------------------------------------------------------------

class TestDisagreement:
    def test_split_two_judges_low_agreement(self):
        # j1 passes (7.0), j2 fails (3.0); mean=5.0 → exactly at threshold
        # j1's verdict is pass, j2's is fail
        g = MultiJudgeGate(judges=[
            ("j1", _always(7.0)),
            ("j2", _always(3.0)),
        ])
        v = g.evaluate("response", ["c1"])
        # overall_pass = (mean 5.0 >= threshold 5.0) → True
        # j1 verdict = True (matches), j2 verdict = False (disagrees)
        assert v["agreement"] == 0.5

    def test_split_two_judges_not_confident(self):
        g = MultiJudgeGate(judges=[
            ("j1", _always(8.0)),
            ("j2", _always(2.0)),
        ])
        v = g.evaluate("response", ["c1"])
        assert v["confident"] is False

    def test_split_two_judges_flagged(self):
        g = MultiJudgeGate(judges=[
            ("j1", _always(8.0)),
            ("j2", _always(2.0)),
        ])
        v = g.evaluate("response", ["c1"])
        assert v["flagged"] is True


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

class TestDrift:
    def test_low_variance_not_drifted(self):
        g = MultiJudgeGate(judges=[
            ("j1", _always(8.0)),
            ("j2", _always(8.5)),
            ("j3", _always(7.5)),
        ])
        v = g.evaluate("response", ["c1"])
        assert v["drifted"] is False

    def test_high_variance_flagged_as_drifted(self):
        g = MultiJudgeGate(judges=[
            ("j1", _always(10.0)),
            ("j2", _always(0.0)),
            ("j3", _always(5.0)),
        ], drift_score_std=2.5)
        v = g.evaluate("response", ["c1"])
        assert v["drifted"] is True

    def test_drift_makes_result_flagged(self):
        g = MultiJudgeGate(judges=[
            ("j1", _always(10.0)),
            ("j2", _always(0.0)),
        ], drift_score_std=2.0)
        v = g.evaluate("response", ["c1"])
        assert v["flagged"] is True


# ---------------------------------------------------------------------------
# Per-criterion details
# ---------------------------------------------------------------------------

class TestPerCriterion:
    def test_per_criterion_scores_recorded(self):
        g = MultiJudgeGate(judges=[
            ("j1", _per_criterion({"c1": 8.0, "c2": 2.0})),
            ("j2", _per_criterion({"c1": 7.0, "c2": 3.0})),
        ])
        v = g.evaluate("response", ["c1", "c2"])
        assert v["per_criterion"]["c1"]["mean"] == pytest.approx(7.5)
        assert v["per_criterion"]["c2"]["mean"] == pytest.approx(2.5)

    def test_per_criterion_passes_flag(self):
        g = MultiJudgeGate(judges=[
            ("j1", _per_criterion({"c1": 8.0, "c2": 2.0})),
            ("j2", _per_criterion({"c1": 7.0, "c2": 3.0})),
        ])
        v = g.evaluate("response", ["c1", "c2"])
        assert v["per_criterion"]["c1"]["passes"] is True
        assert v["per_criterion"]["c2"]["passes"] is False

    def test_overall_fail_when_one_criterion_fails(self):
        g = MultiJudgeGate(judges=[
            ("j1", _per_criterion({"c1": 9.0, "c2": 2.0})),
            ("j2", _per_criterion({"c1": 9.0, "c2": 2.0})),
        ])
        v = g.evaluate("response", ["c1", "c2"])
        assert v["overall_pass"] is False


# ---------------------------------------------------------------------------
# Error / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_criteria_raises(self):
        g = MultiJudgeGate(judges=[
            ("j1", _always(8.0)),
            ("j2", _always(8.0)),
        ])
        with pytest.raises(MultiJudgeGateError):
            g.evaluate("response", [])


# ---------------------------------------------------------------------------
# gate() convenience
# ---------------------------------------------------------------------------

class TestGateMethod:
    def test_gate_true_for_all_pass(self):
        g = MultiJudgeGate(judges=[
            ("j1", _always(8.0)),
            ("j2", _always(9.0)),
        ])
        assert g.gate("response", ["c1"]) is True

    def test_gate_false_for_failed_criterion(self):
        g = MultiJudgeGate(judges=[
            ("j1", _always(2.0)),
            ("j2", _always(3.0)),
        ])
        assert g.gate("response", ["c1"]) is False

    def test_gate_false_when_drifted(self):
        g = MultiJudgeGate(judges=[
            ("j1", _always(10.0)),
            ("j2", _always(0.0)),
        ], drift_score_std=1.0)
        # mean=5.0, mean >= threshold so overall_pass=True
        # but stdev=5.0 > 1.0 → drifted, so gate should be False
        assert g.gate("response", ["c1"]) is False


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

class TestEvidence:
    def test_evidence_emitted_on_evaluate(self):
        from dojo.core.evidence import EvidenceLogger
        g = MultiJudgeGate(judges=[
            ("j1", _always(8.0)),
            ("j2", _always(8.0)),
        ])
        EvidenceLogger.clear()
        g.evaluate("response", ["c1"])
        events = EvidenceLogger.get_events()
        gate_events = [e for e in events if e.get("type") == "eval.multi_judge_gate"]
        assert len(gate_events) == 1
        assert gate_events[0]["data"]["judge_count"] == 2

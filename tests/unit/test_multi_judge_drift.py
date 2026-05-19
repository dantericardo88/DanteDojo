"""Unit tests for d39 — MultiJudgeDriftDetector: 3-judge pairwise agreement gate.

Validates:
- Pairwise agreement ≥ 80% → no error
- Any pairwise pair < 80% → JudgeDriftError
- Individual gold-agreement < 80% also triggers JudgeDriftError
- agreement_rates() returns correct per-judge and pairwise values
- check_pairwise_drift() with 3 distinct models (GPT-4, Claude Haiku, Gemini Flash)
"""

from __future__ import annotations

import pytest

from dojo.core.exceptions import JudgeDriftError
from dojo.eval.judge_drift import MultiJudgeDriftDetector

JUDGES = ["gpt-4", "claude-haiku-4-5", "gemini-flash-2"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feed_rounds(
    detector: MultiJudgeDriftDetector,
    n: int,
    agree_pair: tuple[str, str] | None = None,
    disagree_from: int = 0,
) -> None:
    """Feed *n* rounds.  If *agree_pair* is set, those two judges agree but the
    third disagrees on rounds *disagree_from* onwards."""
    families = detector.judge_families
    for i in range(n):
        gold = "positive"
        outputs: dict[str, str] = {}
        for f in families:
            if agree_pair and i >= disagree_from and f not in agree_pair:
                outputs[f] = "negative"  # third judge disagrees
            else:
                outputs[f] = gold
        detector.record_pairwise(outputs, gold)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMultiJudgeDriftDetectorConstruction:
    def test_requires_at_least_two_families(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            MultiJudgeDriftDetector(["only-one"])

    def test_requires_distinct_families(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            MultiJudgeDriftDetector(["gpt-4", "gpt-4", "claude"])

    def test_three_families_ok(self) -> None:
        d = MultiJudgeDriftDetector(JUDGES)
        assert d.judge_families == JUDGES


class TestPairwiseAgreement:
    def test_perfect_agreement_no_error(self) -> None:
        d = MultiJudgeDriftDetector(JUDGES)
        for _ in range(10):
            d.record_pairwise(
                {"gpt-4": "yes", "claude-haiku-4-5": "yes", "gemini-flash-2": "yes"},
                "yes",
            )
        d.check_pairwise_drift()  # must not raise

    def test_high_agreement_above_threshold(self) -> None:
        d = MultiJudgeDriftDetector(JUDGES, min_agreement=0.80)
        # 9/10 rounds all agree → 90% pairwise
        for i in range(10):
            outputs = {j: "yes" if i < 9 else "no" for j in JUDGES}
            # keep one judge disagreeing only last round
            d.record_pairwise(outputs, "yes")
        d.check_pairwise_drift()  # 90% ≥ 80% → ok

    def test_low_pairwise_agreement_raises(self) -> None:
        """Two judges agree 100% but disagree with the third 60% → drift error."""
        d = MultiJudgeDriftDetector(JUDGES, min_agreement=0.80)
        pair = ("gpt-4", "claude-haiku-4-5")
        for i in range(10):
            gold = "yes"
            outputs: dict[str, str] = {
                "gpt-4": "yes",
                "claude-haiku-4-5": "yes",
                # gemini disagrees on 6/10 rounds → pairwise (gpt-4, gemini) = 40%
                "gemini-flash-2": "yes" if i >= 6 else "no",
            }
            d.record_pairwise(outputs, gold)
        with pytest.raises(JudgeDriftError) as exc_info:
            d.check_pairwise_drift()
        assert "gemini" in str(exc_info.value).lower() or "80%" in str(exc_info.value)

    def test_individual_gold_agreement_below_threshold_raises(self) -> None:
        """If one judge's gold-agreement drops below 80%, drift must be detected."""
        d = MultiJudgeDriftDetector(JUDGES, min_agreement=0.80)
        for i in range(10):
            gold = "yes"
            outputs = {
                "gpt-4": "yes",
                "claude-haiku-4-5": "yes",
                # gemini gets gold right only 3/10 times → 30% individual agreement
                "gemini-flash-2": "yes" if i < 3 else "no",
            }
            d.record_pairwise(outputs, gold)
        with pytest.raises(JudgeDriftError):
            d.check_pairwise_drift()

    def test_agreement_rates_returns_all_pairs(self) -> None:
        d = MultiJudgeDriftDetector(JUDGES)
        for _ in range(5):
            d.record_pairwise({j: "yes" for j in JUDGES}, "yes")
        rates = d.agreement_rates()
        assert "individual" in rates
        assert "pairwise" in rates
        pairs = rates["pairwise"]
        assert len(pairs) == 3  # C(3,2) = 3 pairs
        for rate in pairs.values():
            assert rate == 1.0

    def test_no_records_returns_optimistic_default(self) -> None:
        d = MultiJudgeDriftDetector(JUDGES)
        d.check_pairwise_drift()  # must not raise with no data

    def test_three_distinct_model_names(self) -> None:
        """Explicitly uses GPT-4, Claude Haiku, Gemini Flash as required by d39."""
        d = MultiJudgeDriftDetector(["gpt-4", "claude-haiku-4-5", "gemini-flash-2"])
        for _ in range(10):
            d.record_pairwise(
                {"gpt-4": "A", "claude-haiku-4-5": "A", "gemini-flash-2": "A"},
                "A",
            )
        d.check_pairwise_drift()
        rates = d.agreement_rates()
        for pair_rate in rates["pairwise"].values():
            assert pair_rate >= 0.80

    def test_exact_80_percent_passes(self) -> None:
        d = MultiJudgeDriftDetector(JUDGES, min_agreement=0.80)
        # 8 agree, 2 disagree on one pair → 80% exactly
        for i in range(10):
            outputs = {
                "gpt-4": "yes",
                "claude-haiku-4-5": "yes",
                "gemini-flash-2": "yes" if i < 8 else "no",
            }
            d.record_pairwise(outputs, "yes")
        d.check_pairwise_drift()  # 80% == threshold → must pass

    def test_just_below_80_percent_fails(self) -> None:
        d = MultiJudgeDriftDetector(JUDGES, min_agreement=0.80)
        # 7 agree, 3 disagree → 70% < 80%
        for i in range(10):
            outputs = {
                "gpt-4": "yes",
                "claude-haiku-4-5": "yes",
                "gemini-flash-2": "yes" if i < 7 else "no",
            }
            d.record_pairwise(outputs, "yes")
        with pytest.raises(JudgeDriftError):
            d.check_pairwise_drift()

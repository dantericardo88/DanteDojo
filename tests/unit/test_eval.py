"""Unit tests for dojo/eval/ — Phase 4 evaluation harness."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import EvalError, JudgeDriftError
from dojo.core.types import ModelArtifact
from dojo.eval.dante_suite import DANTE_EVAL_SPECS, DanteEvalTask, get_spec
from dojo.eval.harness import EvalReceipt, EvalSuite, run_eval
from dojo.eval.judge_drift import JudgeDriftDetector
from dojo.eval.statistics import bootstrap_ci, seed_variance, seeds_are_stable


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    """Reset the evidence logger before each test."""
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture
def simple_artifact() -> ModelArtifact:
    return ModelArtifact(
        artifact_id="dojo://test-model/v1",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        lora_weights=None,
        quantization=None,
        training_receipt="abc123",
        passport=None,
        created_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# bootstrap_ci tests
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_identical_scores_ci_equals_score(self):
        """5 identical scores → CI should be [score, score] (all resamples give same mean)."""
        scores = [0.75, 0.75, 0.75, 0.75, 0.75]
        mean, lower, upper = bootstrap_ci(scores, n_bootstrap=1000, seed=42)
        assert abs(mean - 0.75) < 1e-9
        assert abs(lower - 0.75) < 1e-9
        assert abs(upper - 0.75) < 1e-9

    def test_varied_scores_lower_less_than_mean_less_than_upper(self):
        """Varied scores → lower < mean < upper."""
        scores = [0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.55, 0.65]
        mean, lower, upper = bootstrap_ci(scores, n_bootstrap=2000, seed=42)
        assert lower <= mean <= upper
        # With varied data, CI should be non-trivial
        assert upper > lower

    def test_requires_at_least_two_samples(self):
        with pytest.raises(ValueError, match="at least 2"):
            bootstrap_ci([0.5])

    def test_deterministic_given_seed(self):
        scores = [0.5, 0.6, 0.7, 0.8, 0.55]
        result1 = bootstrap_ci(scores, seed=99)
        result2 = bootstrap_ci(scores, seed=99)
        assert result1 == result2

    def test_different_seeds_may_differ(self):
        scores = [0.5, 0.6, 0.7, 0.8, 0.55]
        r1 = bootstrap_ci(scores, seed=1)
        r2 = bootstrap_ci(scores, seed=2)
        # Means must be equal (deterministic), but bounds may differ
        assert abs(r1[0] - r2[0]) < 1e-9


# ---------------------------------------------------------------------------
# seed_variance tests
# ---------------------------------------------------------------------------

class TestSeedVariance:
    def test_all_same_std_zero(self):
        assert seed_variance([1.0, 1.0, 1.0]) == 0.0

    def test_single_element_std_zero(self):
        assert seed_variance([5.0]) == 0.0

    def test_empty_std_zero(self):
        assert seed_variance([]) == 0.0

    def test_known_variance(self):
        # std([0, 2]) = 1.0 (population std)
        assert abs(seed_variance([0.0, 2.0]) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# seeds_are_stable tests
# ---------------------------------------------------------------------------

class TestSeedsAreStable:
    def test_zero_std_is_stable(self):
        assert seeds_are_stable([1.0, 1.0, 1.0]) is True

    def test_high_std_is_unstable(self):
        # std of [0, 5, 10] ≈ 4.08 > 2.0
        assert seeds_are_stable([0.0, 5.0, 10.0], max_std=2.0) is False

    def test_custom_max_std(self):
        # std([0, 2]) = 1.0; passes with max_std=2 but fails with max_std=0.5
        assert seeds_are_stable([0.0, 2.0], max_std=2.0) is True
        assert seeds_are_stable([0.0, 2.0], max_std=0.5) is False

    def test_std_exactly_at_threshold_is_stable(self):
        # std([0, 2]) = 1.0; max_std=1.0 → stable (<=)
        assert seeds_are_stable([0.0, 2.0], max_std=1.0) is True


# ---------------------------------------------------------------------------
# run_eval tests
# ---------------------------------------------------------------------------

class TestRunEval:
    def test_returns_eval_receipt(self, simple_artifact):
        suite = EvalSuite(suite_type="dante", tasks=("task_a", "task_b"), seeds=3)
        receipt = run_eval(simple_artifact, suite, seeds=3)
        assert isinstance(receipt, EvalReceipt)

    def test_merkle_hash_non_empty(self, simple_artifact):
        suite = EvalSuite(suite_type="dante", tasks=("task_x",), seeds=3)
        receipt = run_eval(simple_artifact, suite, seeds=3)
        assert receipt.merkle_hash and len(receipt.merkle_hash) == 64

    def test_emits_eval_start_evidence(self, simple_artifact):
        suite = EvalSuite(suite_type="dante", tasks=("task_y",), seeds=2)
        run_eval(simple_artifact, suite, seeds=2)
        events = EvidenceLogger.get_events()
        types = [e.get("type", "") for e in events]
        assert EvidenceType.EVAL_START.value in types

    def test_emits_eval_complete_evidence(self, simple_artifact):
        suite = EvalSuite(suite_type="dante", tasks=("task_z",), seeds=2)
        run_eval(simple_artifact, suite, seeds=2)
        events = EvidenceLogger.get_events()
        types = [e.get("type", "") for e in events]
        assert EvidenceType.EVAL_COMPLETE.value in types

    def test_receipt_to_dict_is_json_serializable(self, simple_artifact):
        suite = EvalSuite(suite_type="dante", tasks=("task_a",), seeds=2)
        receipt = run_eval(simple_artifact, suite, seeds=2)
        d = receipt.to_dict()
        # Must not raise
        serialized = json.dumps(d)
        assert serialized

    def test_receipt_contains_all_tasks(self, simple_artifact):
        suite = EvalSuite(suite_type="dante", tasks=("task_a", "task_b"), seeds=2)
        receipt = run_eval(simple_artifact, suite, seeds=2)
        assert "task_a" in receipt.mean_scores
        assert "task_b" in receipt.mean_scores

    def test_contamination_check_passed_when_no_hashes(self, simple_artifact):
        suite = EvalSuite(suite_type="dante", tasks=("task_c",), seeds=2)
        receipt = run_eval(simple_artifact, suite, seeds=2)
        assert receipt.contamination_check_passed is True

    def test_seed_scores_have_correct_count(self, simple_artifact):
        suite = EvalSuite(suite_type="dante", tasks=("task_d",), seeds=4)
        receipt = run_eval(simple_artifact, suite, seeds=4)
        assert len(receipt.seed_scores["task_d"]) == 4

    def test_ci_bounds_sensible(self, simple_artifact):
        suite = EvalSuite(suite_type="dante", tasks=("task_e",), seeds=5)
        receipt = run_eval(simple_artifact, suite, seeds=5)
        task = "task_e"
        assert receipt.ci_lower[task] <= receipt.mean_scores[task] <= receipt.ci_upper[task]


# ---------------------------------------------------------------------------
# DanteEvalSpec tests
# ---------------------------------------------------------------------------

class TestDanteEvalSpec:
    def test_get_spec_empanada_bookkeeping(self):
        spec = get_spec(DanteEvalTask.EMPANADA_BOOKKEEPING_F1)
        assert spec.passing_threshold == 0.75
        assert spec.metric == "f1"
        assert spec.judge_family is None

    def test_get_spec_b2b_email_quality(self):
        spec = get_spec(DanteEvalTask.B2B_EMAIL_QUALITY)
        assert spec.passing_threshold == 0.70
        assert spec.judge_family == "claude-sonnet-4-6"

    def test_all_five_tasks_registered(self):
        assert len(DANTE_EVAL_SPECS) == 5
        for task in DanteEvalTask:
            assert task in DANTE_EVAL_SPECS

    def test_specs_are_frozen(self):
        spec = get_spec(DanteEvalTask.OCR_CLEANUP_EXACT_MATCH)
        with pytest.raises((AttributeError, TypeError)):
            spec.passing_threshold = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# JudgeDriftDetector tests
# ---------------------------------------------------------------------------

class TestJudgeDriftDetector:
    def test_10_of_10_matches_agreement_rate_1(self):
        detector = JudgeDriftDetector("claude-sonnet-4-6", window_days=30)
        for _ in range(10):
            detector.record_agreement("positive", "positive")
        assert detector.agreement_rate() == 1.0

    def test_10_of_10_matches_check_drift_passes(self):
        detector = JudgeDriftDetector("claude-sonnet-4-6", window_days=30)
        for _ in range(10):
            detector.record_agreement("positive", "positive")
        # Should not raise
        detector.check_drift()

    def test_7_of_10_matches_raises_judge_drift_error(self):
        detector = JudgeDriftDetector("claude-sonnet-4-6", window_days=30)
        for _ in range(7):
            detector.record_agreement("positive", "positive")
        for _ in range(3):
            detector.record_agreement("positive", "negative")
        assert abs(detector.agreement_rate() - 0.7) < 1e-9
        with pytest.raises(JudgeDriftError):
            detector.check_drift()

    def test_record_agreement_returns_true_on_match(self):
        detector = JudgeDriftDetector("test-judge", window_days=30)
        assert detector.record_agreement("yes", "yes") is True
        assert detector.record_agreement("no", "yes") is False

    def test_case_insensitive_matching(self):
        detector = JudgeDriftDetector("test-judge", window_days=30)
        assert detector.record_agreement("YES", "yes") is True
        assert detector.record_agreement("  No  ", "no") is True

    def test_empty_detector_agreement_rate_is_1(self):
        """No records → optimistic default of 1.0."""
        detector = JudgeDriftDetector("test-judge", window_days=30)
        assert detector.agreement_rate() == 1.0

    def test_is_same_family_qwen_vs_qwen(self):
        detector = JudgeDriftDetector("qwen", window_days=30)
        assert detector.is_same_family("qwen", "qwen") is True

    def test_is_same_family_substring(self):
        detector = JudgeDriftDetector("qwen-72b", window_days=30)
        assert detector.is_same_family("qwen-7b-tuned", "qwen-72b") is True

    def test_is_same_family_different_families(self):
        detector = JudgeDriftDetector("claude", window_days=30)
        assert detector.is_same_family("qwen-7b", "claude-sonnet") is False

    def test_is_same_family_empty_strings(self):
        detector = JudgeDriftDetector("test", window_days=30)
        assert detector.is_same_family("", "qwen") is False
        assert detector.is_same_family("qwen", "") is False

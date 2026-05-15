"""Advanced unit tests for d39 — judge agreement gate.

Tests cover JudgeDriftDetector directly and its integration with run_eval().
The harness uses seed-score polarity agreement as a proxy for judge consistency:
  - scores[0] >= 0.5 → "positive" (reference gold label for that task)
  - subsequent seeds produce "votes" compared to the reference
  - agreement < 80% with >= 5 vote-pairs → JudgeDriftError
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import JudgeDriftError
from dojo.core.types import ModelArtifact
from dojo.eval.harness import EvalReceipt, EvalSuite, run_eval
from dojo.eval.judge_drift import JudgeDriftDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(artifact_id: str = "dojo://test-model/v1") -> ModelArtifact:
    return ModelArtifact(
        artifact_id=artifact_id,
        base_model="Qwen/Qwen2.5-7B-Instruct",
        lora_weights=None,
        quantization=None,
        training_receipt="abc123",
        passport=None,
    )


def _make_suite(
    tasks: tuple[str, ...] = ("task_a",),
    suite_type: str = "dante",
    seeds: int = 3,
) -> EvalSuite:
    return EvalSuite(suite_type=suite_type, tasks=tasks, seeds=seeds)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence() -> None:
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ---------------------------------------------------------------------------
# TestJudgeAgreementGate — 8 tests
# ---------------------------------------------------------------------------

class TestJudgeAgreementGate:

    def test_judge_agreement_above_threshold_passes(self):
        """High judge agreement (all seeds agree) → no JudgeDriftError raised."""
        detector = JudgeDriftDetector(judge_family="test-judge", window_days=1)

        # Record 10 pairs where judge always agrees with gold
        for _ in range(10):
            detector.record_agreement("positive", "positive")

        # Should not raise
        detector.check_drift()
        assert detector.agreement_rate() == 1.0

    def test_judge_agreement_below_threshold_raises(self):
        """Low agreement (below 80%) → JudgeDriftError raised."""
        detector = JudgeDriftDetector(judge_family="test-judge", window_days=1)

        # Record 10 pairs: 7 disagree, 3 agree → 30% agreement (< 80%)
        for _ in range(3):
            detector.record_agreement("positive", "positive")
        for _ in range(7):
            detector.record_agreement("negative", "positive")

        with pytest.raises(JudgeDriftError) as exc_info:
            detector.check_drift()

        assert "drifted" in str(exc_info.value).lower() or "agreement" in str(exc_info.value).lower()

    def test_judge_agreement_uses_multiple_seeds(self):
        """run_eval calls judge with multiple seeds for statistical robustness.

        The harness should call _run_single_seed (or _run_lm_eval_seed) once
        per seed. We verify by patching _run_single_seed and checking call count.
        """
        model = _make_model()
        num_seeds = 4
        suite = _make_suite(tasks=("task_a",), suite_type="dante", seeds=num_seeds)

        with patch("dojo.eval.harness._run_single_seed") as mock_seed:
            # Return consistent scores so no stability error fires
            mock_seed.return_value = {"task_a": 0.75}

            receipt = run_eval(model, suite, seeds=num_seeds)

        assert mock_seed.call_count == num_seeds

    def test_judge_agreement_emits_evidence(self):
        """'eval.judge_drift' evidence emitted when judge agreement gate trips."""
        # We need >=5 tasks x (seeds-1) = >=5 vote-pairs AND < 80% agreement.
        # Use 6 tasks, 2 seeds: 6 vote-pairs.  All seeds flip polarity → 0% agreement.
        tasks = tuple(f"task_{i}" for i in range(6))
        model = _make_model()
        suite = _make_suite(tasks=tasks, suite_type="dante", seeds=2)

        # Seed 0 for each task scores > 0.5 (positive reference)
        # Seed 1 for each task scores < 0.5 (negative vote) → 0% agreement
        call_counter = [0]

        def _alternating_seed(m: Any, s: Any, seed: int) -> Dict[str, float]:
            call_counter[0] += 1
            # seed=42 is first run (reference), seed=43 is second run (disagrees)
            if seed == 42:
                return {t: 0.80 for t in tasks}
            else:
                return {t: 0.20 for t in tasks}

        with patch("dojo.eval.harness._run_single_seed", side_effect=_alternating_seed):
            with pytest.raises(JudgeDriftError):
                run_eval(model, suite, seeds=2)

        events = EvidenceLogger.get_events()
        event_types = [e["type"] for e in events]
        assert EvidenceType.EVAL_JUDGE_DRIFT.value in event_types, (
            f"Expected {EvidenceType.EVAL_JUDGE_DRIFT.value!r} in {event_types}"
        )

    def test_judge_scores_averaged(self):
        """EvalReceipt.mean_scores[task] is the mean of per-seed scores."""
        model = _make_model()
        suite = _make_suite(tasks=("task_a",), suite_type="dante", seeds=3)

        seed_values = [0.60, 0.70, 0.80]
        expected_mean = sum(seed_values) / len(seed_values)

        call_counter = [0]

        def _seed_by_index(m: Any, s: Any, seed: int) -> Dict[str, float]:
            idx = seed - 42  # seeds are 42, 43, 44
            return {"task_a": seed_values[idx]}

        with patch("dojo.eval.harness._run_single_seed", side_effect=_seed_by_index):
            receipt = run_eval(model, suite, seeds=3)

        # Mean should equal the arithmetic mean of the three seed values
        assert abs(receipt.mean_scores["task_a"] - expected_mean) < 1e-9

    def test_judge_gate_integrates_with_eval_receipt(self):
        """Judge result (seed scores) is stored in EvalReceipt.seed_scores."""
        model = _make_model()
        suite = _make_suite(tasks=("task_a",), suite_type="dante", seeds=3)

        per_seed_scores = [0.55, 0.60, 0.65]

        call_counter = [0]

        def _seed_fn(m: Any, s: Any, seed: int) -> Dict[str, float]:
            idx = seed - 42
            return {"task_a": per_seed_scores[idx]}

        with patch("dojo.eval.harness._run_single_seed", side_effect=_seed_fn):
            receipt = run_eval(model, suite, seeds=3)

        assert isinstance(receipt, EvalReceipt)
        scores_in_receipt = receipt.seed_scores["task_a"]
        # All three seed scores should be present
        assert sorted(scores_in_receipt) == sorted(per_seed_scores)

    def test_judge_handles_draw_case(self):
        """Judges disagree exactly 50/50 → agreement_rate=0.5 → JudgeDriftError raised."""
        detector = JudgeDriftDetector(judge_family="test-judge", window_days=1)

        # 5 agree, 5 disagree → 50% agreement (below 80% threshold)
        for _ in range(5):
            detector.record_agreement("positive", "positive")
        for _ in range(5):
            detector.record_agreement("negative", "positive")

        assert abs(detector.agreement_rate() - 0.50) < 1e-9
        with pytest.raises(JudgeDriftError):
            detector.check_drift()

    def test_judge_rejects_on_single_judge_outlier(self):
        """One very low-scoring outlier judge pulls agreement below threshold.

        Scenario: run_eval with 6 tasks, 6 seeds.
        Seeds 1-5 all agree with seed 0 (positive polarity).
        But seeds 1-5 for task 0 ALL disagree → enough disagreements to
        bring the overall agreement rate below 80%.
        """
        # 6 tasks, 6 seeds → 6 * 5 = 30 vote-pairs needed
        # If 1 task contributes 5 disagreements out of 30 total: 25/30 = 83.3% — still passes.
        # To fail with one outlier task we need more disagreements.
        # Use 6 tasks with 6 seeds:
        #   - task_0: all seeds after seed-0 disagree (5 disagreements)
        #   - task_1..5: all seeds after seed-0 also disagree (5 each × 5 = 25)
        # That is 30/30 disagreements → 0% agreement.
        # To model "single outlier", we only flip ONE task across ALL seeds:
        # tasks 1..5 agree, task_0 disagrees for each subsequent seed.
        # 5 tasks × 5 disagreements = 25 votes total: 25 from agree + 0 from disagree? Let's
        # think more carefully.
        #
        # 6 tasks, 6 seeds:
        #   reference seed (seed=42): all tasks score 0.80 (positive)
        #   seeds 43-47: task_0 scores 0.20 (disagree), tasks 1-5 score 0.80 (agree)
        #   vote-pairs: 6 tasks × 5 extra seeds = 30 total
        #   agreements: 5 tasks × 5 seeds = 25 agreements
        #   disagreements: 1 task × 5 seeds = 5 disagreements
        #   rate = 25/30 ≈ 83.3% — above threshold!
        #
        # To have a single outlier that causes failure we need to model it differently.
        # The test description says "one very low-scoring judge pulls agreement below threshold".
        # Use 2 tasks, 6 seeds where task_0 disagrees in all 5 extra seeds:
        #   vote-pairs: 2 × 5 = 10
        #   agreements: 1 task × 5 = 5
        #   disagreements: 1 task × 5 = 5
        #   rate = 5/10 = 50% → below 80% → JudgeDriftError raised.
        tasks = ("task_a", "task_b")
        model = _make_model()
        suite = _make_suite(tasks=tasks, suite_type="dante", seeds=6)

        def _seed_fn(m: Any, s: Any, seed: int) -> Dict[str, float]:
            if seed == 42:
                # All tasks start positive
                return {t: 0.80 for t in tasks}
            else:
                # task_a is the outlier: always negative
                # task_b stays positive (agrees with reference)
                return {"task_a": 0.20, "task_b": 0.80}

        with patch("dojo.eval.harness._run_single_seed", side_effect=_seed_fn):
            with pytest.raises(JudgeDriftError):
                run_eval(model, suite, seeds=6)

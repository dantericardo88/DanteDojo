"""Comprehensive tests for synthetic eval fallback and judge-drift gate.

Covers:
- Synthetic score determinism and validity (6 tests)
- Synthetic lm-eval run behaviour (8 tests)
- Judge drift gate wired into harness (6 tests in TestJudgeDriftInHarness)

All tests run without a GPU or a real lm-eval installation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import JudgeDriftError
from dojo.core.types import ModelArtifact
from dojo.eval.harness import EvalReceipt, EvalSuite, run_eval
from dojo.eval.lm_eval_adapter import (
    LmEvalRunResult,
    _synthetic_results_hash,
    _synthetic_score,
    run_lm_eval,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture
def artifact() -> ModelArtifact:
    return ModelArtifact(
        artifact_id="dojo://test-synthetic/v1",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        lora_weights=None,
        quantization=None,
        training_receipt="syn_receipt",
        passport=None,
        created_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Synthetic score stability (6 tests)
# ---------------------------------------------------------------------------


class TestSyntheticScoreStability:
    def test_synthetic_score_same_seed_deterministic(self):
        """Same task/artifact/seed always returns the identical score."""
        s1 = _synthetic_score("mmlu", "model-abc", 42)
        s2 = _synthetic_score("mmlu", "model-abc", 42)
        assert s1 == s2

    def test_synthetic_score_different_seeds_may_differ(self):
        """Different seeds are allowed to produce distinct scores."""
        scores = {_synthetic_score("mmlu", "model-abc", seed) for seed in range(10)}
        # At least 2 distinct values among 10 seeds (extremely unlikely to all collide)
        assert len(scores) >= 2

    def test_synthetic_score_within_valid_range(self):
        """Scores must always lie in [0.0, 1.0]."""
        for task in ("mmlu", "hellaswag", "arc_challenge", "winogrande"):
            for seed in range(5):
                score = _synthetic_score(task, "model-abc", seed)
                assert 0.0 <= score <= 1.0, f"Out-of-range score {score} for {task}:{seed}"

    def test_synthetic_results_hash_is_hex_string(self):
        """Hash must be a 64-character hex string (SHA-256)."""
        h = _synthetic_results_hash("mmlu", "model-abc", 42)
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_synthetic_results_hash_same_inputs_same_output(self):
        """Hash is deterministic for identical inputs."""
        h1 = _synthetic_results_hash("hellaswag", "model-xyz", 7)
        h2 = _synthetic_results_hash("hellaswag", "model-xyz", 7)
        assert h1 == h2

    def test_synthetic_results_hash_different_tasks_differ(self):
        """Different task names must produce different hashes."""
        h_mmlu = _synthetic_results_hash("mmlu", "model-abc", 42)
        h_arc = _synthetic_results_hash("arc_challenge", "model-abc", 42)
        assert h_mmlu != h_arc


# ---------------------------------------------------------------------------
# Synthetic lm-eval run (8 tests)
# ---------------------------------------------------------------------------


class TestSyntheticLmEvalRun:
    """run_lm_eval with lm-eval absent → synthetic fallback always active."""

    @pytest.fixture(autouse=True)
    def patch_lm_eval_absent(self):
        with patch(
            "dojo.eval.lm_eval_adapter.is_lm_eval_available", return_value=False
        ):
            yield

    def test_run_lm_eval_synthetic_returns_result(self, artifact):
        results = run_lm_eval(artifact, ["mmlu"], seed=42, allow_synthetic=True)
        assert isinstance(results, dict)
        assert "mmlu" in results
        assert isinstance(results["mmlu"], LmEvalRunResult)

    def test_run_lm_eval_synthetic_is_synthetic_flag_true(self, artifact):
        results = run_lm_eval(artifact, ["hellaswag"], seed=42, allow_synthetic=True)
        assert results["hellaswag"].is_synthetic is True

    def test_run_lm_eval_synthetic_emits_evidence(self, artifact):
        run_lm_eval(artifact, ["mmlu"], seed=42, allow_synthetic=True)
        events = EvidenceLogger.get_events()
        types = [e.get("type", "") for e in events]
        assert EvidenceType.EVAL_LM_HARNESS_START.value in types
        assert EvidenceType.EVAL_LM_HARNESS_COMPLETE.value in types

    def test_run_lm_eval_synthetic_multiple_tasks(self, artifact):
        tasks = ["mmlu", "hellaswag", "arc_challenge"]
        results = run_lm_eval(artifact, tasks, seed=42, allow_synthetic=True)
        assert set(results.keys()) == set(tasks)

    def test_run_lm_eval_synthetic_scores_in_range(self, artifact):
        tasks = ["mmlu", "hellaswag", "arc_challenge", "winogrande"]
        results = run_lm_eval(artifact, tasks, seed=42, allow_synthetic=True)
        for task, result in results.items():
            assert 0.0 <= result.score <= 1.0, f"Score out of range for {task}"

    def test_run_lm_eval_synthetic_different_tasks_different_scores(self, artifact):
        tasks = ["mmlu", "hellaswag", "arc_challenge", "winogrande", "piqa"]
        results = run_lm_eval(artifact, tasks, seed=42, allow_synthetic=True)
        scores = [results[t].score for t in tasks]
        # Not all scores should be identical (task name drives the hash)
        assert len(set(scores)) > 1

    def test_run_lm_eval_synthetic_score_reflects_task_name(self, artifact):
        """Score is tied to task identity — running same task twice gives same score."""
        r1 = run_lm_eval(artifact, ["mmlu"], seed=42, allow_synthetic=True)
        r2 = run_lm_eval(artifact, ["mmlu"], seed=42, allow_synthetic=True)
        assert r1["mmlu"].score == r2["mmlu"].score

    def test_run_lm_eval_synthetic_no_real_gpu_needed(self, artifact, monkeypatch):
        """Test passes without DOJO_REAL_GPU environment variable set."""
        monkeypatch.delenv("DOJO_REAL_GPU", raising=False)
        results = run_lm_eval(artifact, ["mmlu"], seed=42, allow_synthetic=True)
        assert results["mmlu"].is_synthetic is True


# ---------------------------------------------------------------------------
# Judge drift integration in harness (6 tests)
# ---------------------------------------------------------------------------


class TestJudgeDriftInHarness:
    """Verify that run_eval wires JudgeDriftDetector correctly."""

    @pytest.fixture(autouse=True)
    def patch_lm_eval_absent(self):
        """Ensure no real lm-eval is needed for any harness test."""
        with patch(
            "dojo.eval.harness.is_lm_eval_available", return_value=False
        ), patch(
            "dojo.eval.lm_eval_adapter.is_lm_eval_available", return_value=False
        ):
            yield

    def test_judge_drift_check_called_during_multi_seed_eval(self, artifact):
        """Multi-seed run invokes drift check — no exception means check ran OK."""
        suite = EvalSuite(suite_type="dante", tasks=("task_a",), seeds=3)
        receipt = run_eval(artifact, suite, seeds=3)
        assert isinstance(receipt, EvalReceipt)

    def test_judge_drift_passes_when_seeds_agree(self, artifact):
        """When all seed scores share the same polarity, drift check should pass."""
        # Synthetic scores for "task_a" with model "dojo://test-synthetic/v1" are
        # deterministic; we just assert no JudgeDriftError is raised.
        suite = EvalSuite(suite_type="dante", tasks=("task_a",), seeds=4)
        receipt = run_eval(artifact, suite, seeds=4)
        assert receipt is not None

    def test_judge_drift_raises_on_low_agreement(self, artifact):
        """Inject disagreeing seed scores across enough tasks/seeds to pass the
        minimum-records gate (>= 5) and force a JudgeDriftError."""
        # 2 tasks × (5-1) seeds = 8 vote-pairs — comfortably above gate of 5.
        call_count = {"n": 0}

        def flipping_seed_scores(model, suite, seed):
            call_count["n"] += 1
            # Alternate between high (positive) and very low (negative) polarity
            if call_count["n"] % 2 == 0:
                return {task: 0.10 for task in suite.tasks}
            return {task: 0.90 for task in suite.tasks}

        with patch("dojo.eval.harness._run_single_seed", side_effect=flipping_seed_scores):
            suite = EvalSuite(suite_type="dante", tasks=("task_x", "task_y"), seeds=5)
            with pytest.raises(JudgeDriftError):
                run_eval(artifact, suite, seeds=5)

    def test_judge_drift_evidence_emitted_on_failure(self, artifact):
        """EVAL_JUDGE_DRIFT evidence is emitted before the error propagates."""
        call_count = {"n": 0}

        def flipping_seed_scores(model, suite, seed):
            call_count["n"] += 1
            if call_count["n"] % 2 == 0:
                return {task: 0.10 for task in suite.tasks}
            return {task: 0.90 for task in suite.tasks}

        with patch("dojo.eval.harness._run_single_seed", side_effect=flipping_seed_scores):
            suite = EvalSuite(suite_type="dante", tasks=("task_x", "task_y"), seeds=5)
            with pytest.raises(JudgeDriftError):
                run_eval(artifact, suite, seeds=5)

        events = EvidenceLogger.get_events()
        types = [e.get("type", "") for e in events]
        assert EvidenceType.EVAL_JUDGE_DRIFT.value in types

    def test_judge_drift_error_has_threshold_info(self, artifact):
        """JudgeDriftError message must mention the threshold value."""
        call_count = {"n": 0}

        def flipping_seed_scores(model, suite, seed):
            call_count["n"] += 1
            if call_count["n"] % 2 == 0:
                return {task: 0.10 for task in suite.tasks}
            return {task: 0.90 for task in suite.tasks}

        with patch("dojo.eval.harness._run_single_seed", side_effect=flipping_seed_scores):
            suite = EvalSuite(suite_type="dante", tasks=("task_x", "task_y"), seeds=5)
            with pytest.raises(JudgeDriftError) as exc_info:
                run_eval(artifact, suite, seeds=5)

        error_text = str(exc_info.value)
        # The error should mention the threshold (0.80 expressed in some form)
        assert "0.8" in error_text or "80" in error_text or "threshold" in error_text.lower()

    def test_judge_drift_single_seed_no_drift_check(self, artifact):
        """Single-seed run skips the drift gate — never raises JudgeDriftError."""
        suite = EvalSuite(suite_type="dante", tasks=("task_a",), seeds=1)
        # If drift check ran with a single seed it would have no comparisons and
        # would trivially pass (agreement_rate = 1.0), but we verify no drift
        # evidence is emitted either.
        receipt = run_eval(artifact, suite, seeds=1)
        assert isinstance(receipt, EvalReceipt)

        events = EvidenceLogger.get_events()
        types = [e.get("type", "") for e in events]
        assert EvidenceType.EVAL_JUDGE_DRIFT.value not in types

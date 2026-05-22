"""Tests verifying all 8 canonical benchmarks are covered by the synthetic eval fallback.

D41 criterion: All 8 canonical benchmarks covered synthetically with correct few-shot.

Benchmarks and required few-shot counts:
  mmlu (5-shot), hellaswag (10-shot), arc_easy (25-shot), arc_challenge (25-shot),
  winogrande (5-shot), gsm8k (8-shot), truthfulqa_mc1 (0-shot), math_hard (4-shot)

All tests run without GPU or a real lm-eval installation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.types import ModelArtifact
from dojo.eval.lm_eval_adapter import (
    LmEvalRunResult,
    run_lm_eval,
)
from dojo.eval.lm_eval_tasks import LM_EVAL_DEFAULT_SUITE, LM_EVAL_STANDARD_TASKS


# ---------------------------------------------------------------------------
# Canonical benchmark definitions (source of truth for this test module)
# ---------------------------------------------------------------------------

CANONICAL_BENCHMARKS = {
    "mmlu": 5,
    "hellaswag": 10,
    "arc_easy": 25,
    "arc_challenge": 25,
    "winogrande": 5,
    "gsm8k": 8,
    "truthfulqa_mc1": 0,
    "math_hard": 4,
}


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
        artifact_id="dojo://test-synthetic-coverage/v1",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        lora_weights=None,
        quantization=None,
        training_receipt="cov_receipt",
        passport=None,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture(autouse=True)
def patch_lm_eval_absent():
    """All tests in this module run without lm-eval installed."""
    with patch(
        "dojo.eval.lm_eval_adapter.is_lm_eval_available", return_value=False
    ), patch(
        "dojo.eval.harness.is_lm_eval_available", return_value=False
    ):
        yield


# ---------------------------------------------------------------------------
# Catalog coverage tests
# ---------------------------------------------------------------------------


class TestCatalogCoverage:
    """Verify the LM_EVAL_STANDARD_TASKS catalog and default suite contain all 8."""

    def test_all_8_benchmarks_in_standard_tasks_catalog(self):
        """All canonical benchmark names must be registered in LM_EVAL_STANDARD_TASKS."""
        for benchmark in CANONICAL_BENCHMARKS:
            assert benchmark in LM_EVAL_STANDARD_TASKS, (
                f"Benchmark '{benchmark}' missing from LM_EVAL_STANDARD_TASKS"
            )

    def test_correct_fewshot_counts_in_catalog(self):
        """Each canonical benchmark must have the correct few-shot count in the catalog."""
        for benchmark, expected_fewshot in CANONICAL_BENCHMARKS.items():
            task = LM_EVAL_STANDARD_TASKS[benchmark]
            assert task.default_num_fewshot == expected_fewshot, (
                f"Benchmark '{benchmark}': expected {expected_fewshot}-shot, "
                f"got {task.default_num_fewshot}-shot"
            )

    def test_all_8_benchmarks_in_default_suite(self):
        """LM_EVAL_DEFAULT_SUITE must contain all 8 canonical benchmarks."""
        for benchmark in CANONICAL_BENCHMARKS:
            assert benchmark in LM_EVAL_DEFAULT_SUITE, (
                f"Benchmark '{benchmark}' missing from LM_EVAL_DEFAULT_SUITE"
            )

    def test_default_suite_has_at_least_8_tasks(self):
        """The default suite should contain at least the 8 canonical benchmarks."""
        assert len(LM_EVAL_DEFAULT_SUITE) >= 8


# ---------------------------------------------------------------------------
# Synthetic fallback run tests — each benchmark individually
# ---------------------------------------------------------------------------


class TestSyntheticFallbackPerBenchmark:
    """Each canonical benchmark returns a valid synthetic result with correct few-shot."""

    @pytest.mark.parametrize("benchmark,expected_fewshot", CANONICAL_BENCHMARKS.items())
    def test_synthetic_result_returned(self, artifact, benchmark, expected_fewshot):
        """Synthetic fallback returns an LmEvalRunResult for each benchmark."""
        results = run_lm_eval(artifact, [benchmark], seed=42, allow_synthetic=True)
        assert benchmark in results
        assert isinstance(results[benchmark], LmEvalRunResult)

    @pytest.mark.parametrize("benchmark,expected_fewshot", CANONICAL_BENCHMARKS.items())
    def test_score_in_valid_range(self, artifact, benchmark, expected_fewshot):
        """Synthetic score must be in [0.0, 1.0]."""
        results = run_lm_eval(artifact, [benchmark], seed=42, allow_synthetic=True)
        score = results[benchmark].score
        assert 0.0 <= score <= 1.0, (
            f"Score {score} out of range [0.0, 1.0] for benchmark '{benchmark}'"
        )

    @pytest.mark.parametrize("benchmark,expected_fewshot", CANONICAL_BENCHMARKS.items())
    def test_correct_fewshot_count_in_result(self, artifact, benchmark, expected_fewshot):
        """num_fewshot in the synthetic result must match the canonical few-shot count."""
        results = run_lm_eval(artifact, [benchmark], seed=42, allow_synthetic=True)
        result = results[benchmark]
        assert result.num_fewshot == expected_fewshot, (
            f"Benchmark '{benchmark}': expected {expected_fewshot}-shot, "
            f"got {result.num_fewshot}-shot in result"
        )

    @pytest.mark.parametrize("benchmark,expected_fewshot", CANONICAL_BENCHMARKS.items())
    def test_is_synthetic_true(self, artifact, benchmark, expected_fewshot):
        """is_synthetic must be True for all synthetic fallback results."""
        results = run_lm_eval(artifact, [benchmark], seed=42, allow_synthetic=True)
        assert results[benchmark].is_synthetic is True, (
            f"is_synthetic should be True for benchmark '{benchmark}'"
        )

    @pytest.mark.parametrize("benchmark,expected_fewshot", CANONICAL_BENCHMARKS.items())
    def test_eval_synthetic_fallback_evidence_emitted(self, artifact, benchmark, expected_fewshot):
        """EVAL_SYNTHETIC_FALLBACK evidence must be emitted for each benchmark."""
        run_lm_eval(artifact, [benchmark], seed=42, allow_synthetic=True)
        events = EvidenceLogger.get_events()
        synthetic_events = [
            e for e in events
            if e.get("type") == EvidenceType.EVAL_SYNTHETIC_FALLBACK.value
        ]
        assert len(synthetic_events) >= 1, (
            f"No EVAL_SYNTHETIC_FALLBACK evidence emitted for benchmark '{benchmark}'"
        )
        # Check the evidence has correct task name and few_shot_count
        task_event = next(
            (e for e in synthetic_events if e.get("data", {}).get("task") == benchmark),
            None,
        )
        assert task_event is not None, (
            f"No EVAL_SYNTHETIC_FALLBACK evidence with task='{benchmark}'"
        )
        assert task_event["data"]["few_shot_count"] == expected_fewshot, (
            f"Evidence few_shot_count mismatch for '{benchmark}': "
            f"expected {expected_fewshot}, got {task_event['data']['few_shot_count']}"
        )
        assert task_event["data"]["is_synthetic"] is True


# ---------------------------------------------------------------------------
# All 8 benchmarks in a single run
# ---------------------------------------------------------------------------


class TestAllEightBenchmarksTogether:
    """Run all 8 benchmarks together and verify the complete result set."""

    def test_all_8_benchmarks_returned_in_single_run(self, artifact):
        """A single run requesting all 8 benchmarks returns results for each."""
        task_names = list(CANONICAL_BENCHMARKS.keys())
        results = run_lm_eval(artifact, task_names, seed=42, allow_synthetic=True)
        assert set(results.keys()) == set(CANONICAL_BENCHMARKS.keys()), (
            f"Missing tasks: {set(CANONICAL_BENCHMARKS.keys()) - set(results.keys())}"
        )

    def test_all_8_scores_in_valid_range(self, artifact):
        """All 8 synthetic scores must be in [0.0, 1.0]."""
        task_names = list(CANONICAL_BENCHMARKS.keys())
        results = run_lm_eval(artifact, task_names, seed=42, allow_synthetic=True)
        for benchmark, result in results.items():
            assert 0.0 <= result.score <= 1.0, (
                f"Score {result.score} out of range for '{benchmark}'"
            )

    def test_all_8_fewshot_counts_correct(self, artifact):
        """All 8 synthetic results carry the correct few-shot count."""
        task_names = list(CANONICAL_BENCHMARKS.keys())
        results = run_lm_eval(artifact, task_names, seed=42, allow_synthetic=True)
        for benchmark, expected_fewshot in CANONICAL_BENCHMARKS.items():
            actual = results[benchmark].num_fewshot
            assert actual == expected_fewshot, (
                f"'{benchmark}': expected {expected_fewshot}-shot, got {actual}-shot"
            )

    def test_all_8_is_synthetic_true(self, artifact):
        """All 8 results must declare is_synthetic=True."""
        task_names = list(CANONICAL_BENCHMARKS.keys())
        results = run_lm_eval(artifact, task_names, seed=42, allow_synthetic=True)
        for benchmark, result in results.items():
            assert result.is_synthetic is True, (
                f"is_synthetic should be True for '{benchmark}'"
            )

    def test_8_synthetic_fallback_evidence_events_emitted(self, artifact):
        """One EVAL_SYNTHETIC_FALLBACK evidence event must be emitted per benchmark."""
        task_names = list(CANONICAL_BENCHMARKS.keys())
        run_lm_eval(artifact, task_names, seed=42, allow_synthetic=True)
        events = EvidenceLogger.get_events()
        synthetic_events = [
            e for e in events
            if e.get("type") == EvidenceType.EVAL_SYNTHETIC_FALLBACK.value
        ]
        emitted_tasks = {e["data"]["task"] for e in synthetic_events}
        missing = set(CANONICAL_BENCHMARKS.keys()) - emitted_tasks
        assert not missing, (
            f"Missing EVAL_SYNTHETIC_FALLBACK evidence for: {missing}"
        )

    def test_evidence_fewshot_counts_all_correct(self, artifact):
        """EVAL_SYNTHETIC_FALLBACK evidence must carry the correct few_shot_count for each."""
        task_names = list(CANONICAL_BENCHMARKS.keys())
        run_lm_eval(artifact, task_names, seed=42, allow_synthetic=True)
        events = EvidenceLogger.get_events()
        synthetic_events = {
            e["data"]["task"]: e["data"]["few_shot_count"]
            for e in events
            if e.get("type") == EvidenceType.EVAL_SYNTHETIC_FALLBACK.value
        }
        for benchmark, expected_fewshot in CANONICAL_BENCHMARKS.items():
            assert benchmark in synthetic_events, (
                f"No evidence for benchmark '{benchmark}'"
            )
            assert synthetic_events[benchmark] == expected_fewshot, (
                f"Evidence few_shot_count for '{benchmark}': "
                f"expected {expected_fewshot}, got {synthetic_events[benchmark]}"
            )

    def test_scores_differ_across_benchmarks(self, artifact):
        """Synthetic scores should not all be identical (task name drives the hash)."""
        task_names = list(CANONICAL_BENCHMARKS.keys())
        results = run_lm_eval(artifact, task_names, seed=42, allow_synthetic=True)
        scores = [results[t].score for t in task_names]
        assert len(set(scores)) > 1, (
            "All 8 synthetic scores are identical — hash mixing is broken"
        )

    def test_default_suite_run_via_harness(self, artifact):
        """Running all canonical benchmarks through run_eval (harness) works end-to-end."""
        from dojo.eval.harness import EvalSuite, run_eval
        suite = EvalSuite(
            suite_type="lm_eval",
            tasks=tuple(CANONICAL_BENCHMARKS.keys()),
            seeds=2,
        )
        receipt = run_eval(artifact, suite, seeds=2)
        for benchmark in CANONICAL_BENCHMARKS:
            assert benchmark in receipt.mean_scores, (
                f"Benchmark '{benchmark}' missing from harness receipt"
            )
            assert 0.0 <= receipt.mean_scores[benchmark] <= 1.0

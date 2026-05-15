"""Integration tests for the real lm-eval harness path — d37.

All tests mock `lm_eval.evaluator.simple_evaluate` so CI can run without GPU.
They verify:
- Canonical task catalog completeness and correctness
- Per-task metric extraction (primary key, alt key fallback, missing key)
- Evidence emission (EVAL_LM_HARNESS_START, EVAL_LM_HARNESS_COMPLETE)
- run_eval receipt.source == "lm_eval_real" when real lm-eval is used
- run_eval receipt.source == "lm_eval_synthetic_fallback" when absent
- Merkle hash is derived from real (mocked) scores
- Contamination check runs before lm-eval call
- CLI --list-tasks and default suite injection
- TestRealLmEvalAPIBoundary: patches at lm_eval.evaluator.simple_evaluate to
  validate the real API call signature (num_fewshot batching, peft= model_args,
  error propagation, hash consistency) — not just _run_real_lm_eval.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import EvalContaminationError, EvalError
from dojo.core.types import ModelArtifact
from dojo.eval.harness import EvalReceipt, EvalSuite, run_eval
from dojo.eval.lm_eval_adapter import (
    LmEvalRunResult,
    LmEvalTaskSpec,
    _extract_score,
    _synthetic_score,
    is_lm_eval_available,
    run_lm_eval,
)
from dojo.eval.lm_eval_tasks import (
    LM_EVAL_DEFAULT_SUITE,
    LM_EVAL_STANDARD_TASKS,
    LmEvalTask,
    get_lm_eval_task,
    resolve_task_names,
    resolve_task_names_lenient,
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
        artifact_id="dojo://test-qwen/v1",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        lora_weights=None,
        quantization=None,
        training_receipt="test-receipt-abc",
        passport=None,
        created_at=datetime.now(timezone.utc),
    )


def _make_lm_eval_results(tasks: list[str], base_score: float = 0.72) -> Dict[str, Any]:
    """Build a fake lm_eval simple_evaluate result dict."""
    results: Dict[str, Any] = {}
    for task in tasks:
        results[task] = {
            "acc,none": base_score - 0.01,
            "acc_norm,none": base_score,
            "acc_stderr,none": 0.005,
        }
    return {"results": results, "config": {}}


# ---------------------------------------------------------------------------
# TestLmEvalTaskCatalog
# ---------------------------------------------------------------------------

class TestLmEvalTaskCatalog:
    def test_eight_standard_tasks_registered(self):
        assert len(LM_EVAL_STANDARD_TASKS) == 8

    def test_default_suite_is_subset_of_standard(self):
        for name in LM_EVAL_DEFAULT_SUITE:
            assert name in LM_EVAL_STANDARD_TASKS

    def test_default_suite_has_four_tasks(self):
        assert len(LM_EVAL_DEFAULT_SUITE) == 4

    def test_mmlu_config(self):
        task = LM_EVAL_STANDARD_TASKS["mmlu"]
        assert task.default_num_fewshot == 5
        assert task.primary_metric_key == "acc_norm,none"
        assert "acc,none" in task.alt_metric_keys

    def test_hellaswag_config(self):
        task = LM_EVAL_STANDARD_TASKS["hellaswag"]
        assert task.default_num_fewshot == 10
        assert task.primary_metric_key == "acc_norm,none"

    def test_arc_challenge_config(self):
        task = LM_EVAL_STANDARD_TASKS["arc_challenge"]
        assert task.default_num_fewshot == 25
        assert task.primary_metric_key == "acc_norm,none"

    def test_winogrande_config(self):
        task = LM_EVAL_STANDARD_TASKS["winogrande"]
        assert task.default_num_fewshot == 5
        assert task.primary_metric_key == "acc,none"

    def test_gsm8k_uses_exact_match(self):
        task = LM_EVAL_STANDARD_TASKS["gsm8k"]
        assert "exact_match" in task.primary_metric_key

    def test_get_lm_eval_task_known(self):
        task = get_lm_eval_task("mmlu")
        assert task.name == "mmlu"

    def test_get_lm_eval_task_unknown_raises(self):
        with pytest.raises(EvalError, match="Unknown lm-eval task"):
            get_lm_eval_task("not_a_real_task")

    def test_resolve_task_names_batch(self):
        tasks = resolve_task_names(["mmlu", "hellaswag"])
        assert len(tasks) == 2
        assert tasks[0].name == "mmlu"
        assert tasks[1].name == "hellaswag"

    def test_resolve_task_names_lenient_unknown(self):
        known, unknown = resolve_task_names_lenient(["mmlu", "custom_task_xyz"])
        assert len(known) == 2
        assert "custom_task_xyz" in unknown
        # Unknown task gets default acc,none metric
        custom = known[1]
        assert custom.primary_metric_key == "acc,none"

    def test_all_tasks_are_frozen(self):
        task = get_lm_eval_task("mmlu")
        with pytest.raises((AttributeError, TypeError)):
            task.name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestMetricExtraction
# ---------------------------------------------------------------------------

class TestMetricExtraction:
    def test_primary_key_used_when_present(self):
        results = {"acc_norm,none": 0.75, "acc,none": 0.70}
        score, key = _extract_score(results, "acc_norm,none", ("acc,none",))
        assert abs(score - 0.75) < 1e-9
        assert key == "acc_norm,none"

    def test_fallback_to_alt_key_when_primary_missing(self):
        results = {"acc,none": 0.70}
        score, key = _extract_score(results, "acc_norm,none", ("acc,none",))
        assert abs(score - 0.70) < 1e-9
        assert key == "acc,none"

    def test_fallback_to_first_numeric_when_both_missing(self):
        results = {"some_metric": 0.65, "stderr": 0.01}
        score, key = _extract_score(results, "acc_norm,none", ("acc,none",))
        assert abs(score - 0.65) < 1e-9
        assert key == "some_metric"

    def test_returns_zero_when_no_numeric_value(self):
        results = {"label": "text", "config": {}}
        score, key = _extract_score(results, "acc_norm,none", ())
        assert score == 0.0
        assert key == "missing"

    def test_bool_not_treated_as_numeric(self):
        results = {"flag": True, "acc,none": 0.80}
        score, key = _extract_score(results, "acc_norm,none", ("acc,none",))
        assert abs(score - 0.80) < 1e-9


# ---------------------------------------------------------------------------
# TestLmEvalAdapterMocked — mock evaluator.simple_evaluate
# ---------------------------------------------------------------------------

class TestLmEvalAdapterMocked:
    def test_real_path_uses_acc_norm_for_mmlu(self, artifact):
        """Mocked lm-eval: MMLU result has acc_norm,none → score extracted correctly."""
        fake_results = _make_lm_eval_results(["mmlu"])

        with patch("dojo.eval.lm_eval_adapter.is_lm_eval_available", return_value=True), \
             patch("dojo.eval.lm_eval_adapter._run_real_lm_eval") as mock_run:
            mock_run.return_value = (
                {
                    "mmlu": LmEvalRunResult(
                        task="mmlu", score=0.72,
                        metric_key="acc_norm,none", num_fewshot=5,
                        raw_results_hash="abc123", is_synthetic=False,
                    )
                },
                "abc123",
            )
            results = run_lm_eval(artifact, ["mmlu"])

        assert "mmlu" in results
        r = results["mmlu"]
        assert abs(r.score - 0.72) < 1e-9
        assert r.metric_key == "acc_norm,none"
        assert r.is_synthetic is False

    def test_real_path_emits_harness_start_evidence(self, artifact):
        with patch("dojo.eval.lm_eval_adapter.is_lm_eval_available", return_value=True), \
             patch("dojo.eval.lm_eval_adapter._run_real_lm_eval") as mock_run:
            mock_run.return_value = (
                {"hellaswag": LmEvalRunResult("hellaswag", 0.81, "acc_norm,none", 10, "hash1", False)},
                "hash1",
            )
            run_lm_eval(artifact, ["hellaswag"])

        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        assert EvidenceType.EVAL_LM_HARNESS_START.value in types

    def test_real_path_emits_harness_complete_evidence(self, artifact):
        with patch("dojo.eval.lm_eval_adapter.is_lm_eval_available", return_value=True), \
             patch("dojo.eval.lm_eval_adapter._run_real_lm_eval") as mock_run:
            mock_run.return_value = (
                {"arc_challenge": LmEvalRunResult("arc_challenge", 0.65, "acc_norm,none", 25, "hash2", False)},
                "hash2",
            )
            run_lm_eval(artifact, ["arc_challenge"])

        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        assert EvidenceType.EVAL_LM_HARNESS_COMPLETE.value in types

    def test_complete_evidence_carries_source_lm_eval_real(self, artifact):
        with patch("dojo.eval.lm_eval_adapter.is_lm_eval_available", return_value=True), \
             patch("dojo.eval.lm_eval_adapter._run_real_lm_eval") as mock_run:
            mock_run.return_value = (
                {"winogrande": LmEvalRunResult("winogrande", 0.77, "acc,none", 5, "hash3", False)},
                "hash3",
            )
            run_lm_eval(artifact, ["winogrande"])

        events = EvidenceLogger.get_events()
        complete_events = [e for e in events if e.get("type") == EvidenceType.EVAL_LM_HARNESS_COMPLETE.value]
        assert complete_events
        assert complete_events[-1]["data"]["source"] == "lm_eval_real"

    def test_synthetic_path_marks_is_synthetic_true(self, artifact):
        with patch("dojo.eval.lm_eval_adapter.is_lm_eval_available", return_value=False):
            results = run_lm_eval(artifact, ["mmlu"], allow_synthetic=True)
        assert results["mmlu"].is_synthetic is True
        assert results["mmlu"].metric_key == "synthetic"

    def test_synthetic_path_emits_complete_with_synthetic_fallback_source(self, artifact):
        with patch("dojo.eval.lm_eval_adapter.is_lm_eval_available", return_value=False):
            run_lm_eval(artifact, ["mmlu"])

        events = EvidenceLogger.get_events()
        complete_events = [e for e in events if e.get("type") == EvidenceType.EVAL_LM_HARNESS_COMPLETE.value]
        assert complete_events
        assert complete_events[-1]["data"]["source"] == "synthetic_fallback"

    def test_no_synthetic_raises_when_lm_eval_absent(self, artifact):
        with patch("dojo.eval.lm_eval_adapter.is_lm_eval_available", return_value=False):
            with pytest.raises(EvalError, match="lm-eval is not installed"):
                run_lm_eval(artifact, ["mmlu"], allow_synthetic=False)

    def test_harness_start_carries_per_task_fewshot(self, artifact):
        with patch("dojo.eval.lm_eval_adapter.is_lm_eval_available", return_value=False):
            run_lm_eval(artifact, ["mmlu", "arc_challenge"])

        events = EvidenceLogger.get_events()
        start_events = [e for e in events if e.get("type") == EvidenceType.EVAL_LM_HARNESS_START.value]
        assert start_events
        fewshot = start_events[0]["data"]["num_fewshot"]
        assert fewshot["mmlu"] == 5
        assert fewshot["arc_challenge"] == 25

    def test_raw_results_hash_is_sha256(self, artifact):
        with patch("dojo.eval.lm_eval_adapter.is_lm_eval_available", return_value=True), \
             patch("dojo.eval.lm_eval_adapter._run_real_lm_eval") as mock_run:
            expected_hash = hashlib.sha256(b"test").hexdigest()
            mock_run.return_value = (
                {"boolq": LmEvalRunResult("boolq", 0.85, "acc,none", 0, expected_hash, False)},
                expected_hash,
            )
            results = run_lm_eval(artifact, ["boolq"])

        assert len(results["boolq"].raw_results_hash) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# TestRunEvalWithRealHarness
# ---------------------------------------------------------------------------

class TestRunEvalWithRealHarness:
    def test_receipt_source_lm_eval_real_when_not_synthetic(self, artifact):
        suite = EvalSuite(suite_type="lm_eval", tasks=("mmlu", "hellaswag"), seeds=2)
        with patch("dojo.eval.harness.is_lm_eval_available", return_value=True), \
             patch("dojo.eval.harness._run_lm_eval_seed") as mock_seed:
            mock_seed.return_value = (
                {"mmlu": 0.72, "hellaswag": 0.81},
                {"mmlu": "acc_norm,none", "hellaswag": "acc_norm,none"},
                False,  # is_synthetic=False
            )
            receipt = run_eval(artifact, suite, seeds=2)

        assert receipt.source == "lm_eval_real"

    def test_receipt_source_synthetic_fallback_when_no_lm_eval(self, artifact):
        suite = EvalSuite(suite_type="lm_eval", tasks=("mmlu",), seeds=2)
        with patch("dojo.eval.harness.is_lm_eval_available", return_value=False), \
             patch("dojo.eval.harness._run_lm_eval_seed") as mock_seed:
            mock_seed.return_value = (
                {"mmlu": 0.55},
                {"mmlu": "synthetic"},
                True,  # is_synthetic=True
            )
            receipt = run_eval(artifact, suite, seeds=2)

        assert receipt.source == "lm_eval_synthetic_fallback"

    def test_receipt_merkle_hash_non_empty(self, artifact):
        suite = EvalSuite(suite_type="lm_eval", tasks=("arc_challenge",), seeds=2)
        with patch("dojo.eval.harness._run_lm_eval_seed") as mock_seed:
            mock_seed.return_value = ({"arc_challenge": 0.64}, {"arc_challenge": "acc_norm,none"}, True)
            receipt = run_eval(artifact, suite, seeds=2)

        assert receipt.merkle_hash and len(receipt.merkle_hash) == 64

    def test_contamination_check_runs_before_lm_eval(self, artifact):
        """Contamination error must raise before lm_eval is called."""
        suite = EvalSuite(suite_type="lm_eval", tasks=("mmlu",), seeds=1)
        task_hash = hashlib.sha256(b"mmlu").hexdigest()
        contamination_hashes = frozenset([task_hash])

        with patch("dojo.eval.harness._run_lm_eval_seed") as mock_seed:
            with pytest.raises(EvalContaminationError):
                run_eval(artifact, suite, seeds=1, contamination_hashes=contamination_hashes)
            mock_seed.assert_not_called()

    def test_default_suite_injected_when_no_tasks(self, artifact):
        """lm_eval suite with empty tasks → harness injects default 4 tasks."""
        suite = EvalSuite(suite_type="lm_eval", tasks=(), seeds=1)
        with patch("dojo.eval.harness._run_lm_eval_seed") as mock_seed:
            mock_seed.return_value = (
                {"mmlu": 0.72, "hellaswag": 0.81, "arc_challenge": 0.64, "winogrande": 0.77},
                {k: "acc_norm,none" for k in ("mmlu", "hellaswag", "arc_challenge", "winogrande")},
                True,
            )
            receipt = run_eval(artifact, suite, seeds=1)

        assert set(receipt.tasks) == set(LM_EVAL_DEFAULT_SUITE)

    def test_receipt_contains_metric_keys(self, artifact):
        suite = EvalSuite(suite_type="lm_eval", tasks=("mmlu",), seeds=2)
        with patch("dojo.eval.harness._run_lm_eval_seed") as mock_seed:
            mock_seed.return_value = (
                {"mmlu": 0.72},
                {"mmlu": "acc_norm,none"},
                False,
            )
            receipt = run_eval(artifact, suite, seeds=2)

        assert "mmlu" in receipt.metric_keys
        assert receipt.metric_keys["mmlu"] == "acc_norm,none"

    def test_eval_start_evidence_emitted(self, artifact):
        suite = EvalSuite(suite_type="lm_eval", tasks=("boolq",), seeds=1)
        with patch("dojo.eval.harness._run_lm_eval_seed") as mock_seed:
            mock_seed.return_value = ({"boolq": 0.85}, {"boolq": "acc,none"}, True)
            run_eval(artifact, suite, seeds=1)

        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        assert EvidenceType.EVAL_START.value in types

    def test_eval_complete_evidence_emitted(self, artifact):
        suite = EvalSuite(suite_type="lm_eval", tasks=("boolq",), seeds=1)
        with patch("dojo.eval.harness._run_lm_eval_seed") as mock_seed:
            mock_seed.return_value = ({"boolq": 0.85}, {"boolq": "acc,none"}, True)
            run_eval(artifact, suite, seeds=1)

        events = EvidenceLogger.get_events()
        types = [e.get("type") for e in events]
        assert EvidenceType.EVAL_COMPLETE.value in types

    def test_complete_evidence_carries_source(self, artifact):
        suite = EvalSuite(suite_type="lm_eval", tasks=("winogrande",), seeds=1)
        with patch("dojo.eval.harness._run_lm_eval_seed") as mock_seed:
            mock_seed.return_value = ({"winogrande": 0.77}, {"winogrande": "acc,none"}, False)
            run_eval(artifact, suite, seeds=1)

        events = EvidenceLogger.get_events()
        complete = [e for e in events if e.get("type") == EvidenceType.EVAL_COMPLETE.value]
        assert complete
        assert complete[-1]["data"]["source"] == "lm_eval_real"

    def test_receipt_to_dict_includes_source(self, artifact):
        suite = EvalSuite(suite_type="lm_eval", tasks=("mmlu",), seeds=1)
        with patch("dojo.eval.harness._run_lm_eval_seed") as mock_seed:
            mock_seed.return_value = ({"mmlu": 0.72}, {"mmlu": "acc_norm,none"}, True)
            receipt = run_eval(artifact, suite, seeds=1)

        d = receipt.to_dict()
        assert "source" in d
        assert "metric_keys" in d
        # Must be JSON-serializable
        json.dumps(d)

    def test_ci_bounds_sensible_with_mocked_scores(self, artifact):
        suite = EvalSuite(suite_type="lm_eval", tasks=("mmlu",), seeds=5)
        scores_sequence = [0.70, 0.71, 0.72, 0.73, 0.74]
        call_count = [0]

        def mock_seed_fn(model, suite_obj, seed, allow_synthetic):
            idx = call_count[0] % len(scores_sequence)
            call_count[0] += 1
            return ({"mmlu": scores_sequence[idx]}, {"mmlu": "acc_norm,none"}, False)

        with patch("dojo.eval.harness._run_lm_eval_seed", side_effect=mock_seed_fn):
            receipt = run_eval(artifact, suite, seeds=5)

        assert receipt.ci_lower["mmlu"] <= receipt.mean_scores["mmlu"] <= receipt.ci_upper["mmlu"]


# ---------------------------------------------------------------------------
# TestLmEvalCLI
# ---------------------------------------------------------------------------

class TestLmEvalCLI:
    def test_list_tasks_prints_catalog(self):
        from typer.testing import CliRunner
        from dojo.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["eval", "--list-tasks"])
        assert result.exit_code == 0
        assert "mmlu" in result.output
        assert "hellaswag" in result.output
        assert "arc_challenge" in result.output
        assert "winogrande" in result.output

    def test_list_tasks_shows_fewshot_counts(self):
        from typer.testing import CliRunner
        from dojo.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["eval", "--list-tasks"])
        assert "5" in result.output   # MMLU 5-shot
        assert "25" in result.output  # ARC 25-shot

    def test_list_tasks_does_not_require_artifact_id(self):
        from typer.testing import CliRunner
        from dojo.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["eval", "--list-tasks"])
        # Should exit 0 even with no artifact_id
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Helpers for lm_eval sys.modules injection
# ---------------------------------------------------------------------------

def _make_lm_eval_mock(simple_evaluate_return: Any) -> tuple[MagicMock, MagicMock]:
    """Build a mock lm_eval module tree injected via sys.modules.

    Returns (mock_lm_eval_module, mock_evaluator_module) so callers can
    inspect simple_evaluate call_args after running.
    """
    mock_evaluator = MagicMock()
    mock_evaluator.simple_evaluate.return_value = simple_evaluate_return
    mock_lm_eval = MagicMock()
    mock_lm_eval.evaluator = mock_evaluator
    return mock_lm_eval, mock_evaluator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def artifact_with_lora() -> ModelArtifact:
    return ModelArtifact(
        artifact_id="dojo://test-qwen-lora/v1",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        lora_weights=Path("/tmp/fake-lora-weights"),
        quantization=None,
        training_receipt="test-receipt-lora",
        passport=None,
        created_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# TestRealLmEvalAPIBoundary
# Patches at lm_eval.evaluator.simple_evaluate — validates the real call
# signature (num_fewshot batching, peft= model_args, error propagation).
# ---------------------------------------------------------------------------

class TestRealLmEvalAPIBoundary:
    """Validate _run_real_lm_eval calls simple_evaluate with correct arguments.

    Every test injects a mock lm_eval module via sys.modules so the deferred
    `from lm_eval import evaluator` inside _run_real_lm_eval resolves to our
    mock — without requiring lm_eval to be installed.
    """

    def _patch_lm_eval(self, simple_evaluate_return: Any) -> tuple[MagicMock, dict]:
        mock_lm_eval, mock_evaluator = _make_lm_eval_mock(simple_evaluate_return)
        modules_patch = {"lm_eval": mock_lm_eval, "lm_eval.evaluator": mock_evaluator}
        return mock_evaluator, modules_patch

    def test_mmlu_calls_simple_evaluate_with_num_fewshot_5(self, artifact):
        fake = {"results": {"mmlu": {"acc_norm,none": 0.72}}}
        mock_eval, mods = self._patch_lm_eval(fake)

        with patch.dict(sys.modules, mods), \
             patch("dojo.eval.lm_eval_adapter.is_lm_eval_available", return_value=True):
            run_lm_eval(artifact, ["mmlu"])

        mock_eval.simple_evaluate.assert_called_once()
        kwargs = mock_eval.simple_evaluate.call_args.kwargs
        assert kwargs["num_fewshot"] == 5
        assert kwargs["tasks"] == ["mmlu"]

    def test_arc_challenge_calls_simple_evaluate_with_num_fewshot_25(self, artifact):
        fake = {"results": {"arc_challenge": {"acc_norm,none": 0.64}}}
        mock_eval, mods = self._patch_lm_eval(fake)

        with patch.dict(sys.modules, mods), \
             patch("dojo.eval.lm_eval_adapter.is_lm_eval_available", return_value=True):
            run_lm_eval(artifact, ["arc_challenge"])

        kwargs = mock_eval.simple_evaluate.call_args.kwargs
        assert kwargs["num_fewshot"] == 25
        assert kwargs["tasks"] == ["arc_challenge"]

    def test_different_fewshot_tasks_issue_separate_calls(self, artifact):
        """MMLU=5-shot, ARC=25-shot → two separate simple_evaluate calls."""
        mock_evaluator = MagicMock()
        mock_evaluator.simple_evaluate.side_effect = [
            {"results": {"mmlu": {"acc_norm,none": 0.72}}},
            {"results": {"arc_challenge": {"acc_norm,none": 0.64}}},
        ]
        mock_lm_eval = MagicMock()
        mock_lm_eval.evaluator = mock_evaluator
        mods = {"lm_eval": mock_lm_eval, "lm_eval.evaluator": mock_evaluator}

        with patch.dict(sys.modules, mods), \
             patch("dojo.eval.lm_eval_adapter.is_lm_eval_available", return_value=True):
            run_lm_eval(artifact, ["mmlu", "arc_challenge"])

        assert mock_evaluator.simple_evaluate.call_count == 2
        fewshots = {c.kwargs["num_fewshot"] for c in mock_evaluator.simple_evaluate.call_args_list}
        assert fewshots == {5, 25}

    def test_same_fewshot_tasks_batched_into_one_call(self, artifact):
        """MMLU=5-shot, WinoGrande=5-shot → single batched simple_evaluate call."""
        fake = {"results": {
            "mmlu": {"acc_norm,none": 0.72},
            "winogrande": {"acc,none": 0.77},
        }}
        mock_eval, mods = self._patch_lm_eval(fake)

        with patch.dict(sys.modules, mods), \
             patch("dojo.eval.lm_eval_adapter.is_lm_eval_available", return_value=True):
            run_lm_eval(artifact, ["mmlu", "winogrande"])

        assert mock_eval.simple_evaluate.call_count == 1
        tasks_called = set(mock_eval.simple_evaluate.call_args.kwargs["tasks"])
        assert tasks_called == {"mmlu", "winogrande"}
        assert mock_eval.simple_evaluate.call_args.kwargs["num_fewshot"] == 5

    def test_lora_artifact_includes_peft_in_model_args(self, artifact_with_lora):
        fake = {"results": {"mmlu": {"acc_norm,none": 0.72}}}
        mock_eval, mods = self._patch_lm_eval(fake)

        with patch.dict(sys.modules, mods), \
             patch("dojo.eval.lm_eval_adapter.is_lm_eval_available", return_value=True):
            run_lm_eval(artifact_with_lora, ["mmlu"])

        model_args = mock_eval.simple_evaluate.call_args.kwargs["model_args"]
        assert "peft=" in model_args
        assert str(artifact_with_lora.lora_weights) in model_args

    def test_base_artifact_no_peft_in_model_args(self, artifact):
        fake = {"results": {"mmlu": {"acc_norm,none": 0.72}}}
        mock_eval, mods = self._patch_lm_eval(fake)

        with patch.dict(sys.modules, mods), \
             patch("dojo.eval.lm_eval_adapter.is_lm_eval_available", return_value=True):
            run_lm_eval(artifact, ["mmlu"])

        model_args = mock_eval.simple_evaluate.call_args.kwargs["model_args"]
        assert "peft=" not in model_args
        assert artifact.base_model in model_args

    def test_simple_evaluate_failure_raises_eval_error(self, artifact):
        mock_evaluator = MagicMock()
        mock_evaluator.simple_evaluate.side_effect = RuntimeError("GPU OOM")
        mock_lm_eval = MagicMock()
        mock_lm_eval.evaluator = mock_evaluator
        mods = {"lm_eval": mock_lm_eval, "lm_eval.evaluator": mock_evaluator}

        with patch.dict(sys.modules, mods), \
             patch("dojo.eval.lm_eval_adapter.is_lm_eval_available", return_value=True):
            with pytest.raises(EvalError, match="lm-eval evaluation failed"):
                run_lm_eval(artifact, ["mmlu"])

    def test_scores_extracted_with_correct_metric_key(self, artifact):
        fake = {"results": {"mmlu": {"acc_norm,none": 0.72, "acc,none": 0.68}}}
        mock_eval, mods = self._patch_lm_eval(fake)

        with patch.dict(sys.modules, mods), \
             patch("dojo.eval.lm_eval_adapter.is_lm_eval_available", return_value=True):
            results = run_lm_eval(artifact, ["mmlu"])

        r = results["mmlu"]
        assert abs(r.score - 0.72) < 1e-9
        assert r.metric_key == "acc_norm,none"
        assert r.is_synthetic is False

    def test_raw_results_hash_is_consistent_sha256_over_all_tasks(self, artifact):
        """Hash must be SHA-256(merged results JSON) — deterministic across calls."""
        task_data = {"acc_norm,none": 0.72, "acc,none": 0.68}
        fake = {"results": {"mmlu": task_data, "winogrande": {"acc,none": 0.77}}}

        # Two calls with identical inputs must produce the same hash
        hashes = []
        for _ in range(2):
            mock_eval, mods = self._patch_lm_eval(fake)
            with patch.dict(sys.modules, mods), \
                 patch("dojo.eval.lm_eval_adapter.is_lm_eval_available", return_value=True):
                results = run_lm_eval(artifact, ["mmlu", "winogrande"])
            hashes.append(results["mmlu"].raw_results_hash)

        assert len(hashes[0]) == 64  # SHA-256 hex length
        assert hashes[0] == hashes[1]  # deterministic

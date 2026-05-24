"""Advanced Merkle receipt and lm-eval harness tests — d37/d40.

Fills gaps not covered by test_lm_eval_integration.py and test_eval.py:
- Merkle hash is a valid 64-char hex string
- Merkle hash is reproducible: identical seed_scores → identical hash
- Merkle hash changes when scores change (sensitivity)
- Contamination check result is embedded in EvalReceipt
- Contamination check runs for lm_eval suite type too
- lm_eval tasks catalog contains expected canonical tasks
- LM_EVAL_DEFAULT_SUITE is a tuple of strings
- Missing task uses synthetic fallback with acc,none metric
- EvalReceipt.to_dict() preserves contamination_check_passed field
- Compute merkle hash directly from seed_scores for verification
- Different artifacts with same tasks produce different Merkle hashes
- EvalReceipt.evaluated_at is a valid ISO 8601 timestamp string
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import EvalContaminationError
from dojo.core.types import ModelArtifact
from dojo.eval.harness import EvalReceipt, EvalSuite, _compute_merkle_hash, run_eval
from dojo.eval.lm_eval_tasks import (
    LM_EVAL_DEFAULT_SUITE,
    LM_EVAL_STANDARD_TASKS,
    get_lm_eval_task,
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
        artifact_id="dojo://merkle-test-model/v1",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        lora_weights=None,
        quantization=None,
        training_receipt="merkle-test-receipt",
        passport=None,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def artifact_b() -> ModelArtifact:
    return ModelArtifact(
        artifact_id="dojo://merkle-test-model-b/v2",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        lora_weights=None,
        quantization=None,
        training_receipt="merkle-test-receipt-b",
        passport=None,
        created_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Merkle hash validity (d40)
# ---------------------------------------------------------------------------

class TestMerkleHashValidity:
    def test_merkle_hash_is_64_char_hex(self, artifact):
        """EvalReceipt.merkle_hash must be a 64-character lowercase hex string (SHA-256)."""
        suite = EvalSuite(suite_type="dante", tasks=("task_a",), seeds=3)
        receipt = run_eval(artifact, suite, seeds=3)
        assert len(receipt.merkle_hash) == 64
        assert all(c in "0123456789abcdef" for c in receipt.merkle_hash)

    def test_merkle_hash_non_empty_for_single_task(self, artifact):
        """Single task evaluation produces a non-empty Merkle hash."""
        suite = EvalSuite(suite_type="dante", tasks=("only_task",), seeds=2)
        receipt = run_eval(artifact, suite, seeds=2)
        assert receipt.merkle_hash != ""
        assert receipt.merkle_hash != "0" * 64  # not all zeros

    def test_compute_merkle_hash_directly(self):
        """_compute_merkle_hash produces SHA-256 hex of JSON-encoded seed scores."""
        seed_scores = {"task_a": [0.7, 0.72, 0.68], "task_b": [0.81, 0.80, 0.82]}
        result = _compute_merkle_hash(seed_scores)

        # Manually compute expected hash
        payload = json.dumps(seed_scores, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(payload.encode()).hexdigest()
        assert result == expected

    def test_compute_merkle_hash_is_64_chars(self):
        """_compute_merkle_hash always returns a 64-char hex string."""
        h = _compute_merkle_hash({"task": [0.5, 0.6]})
        assert len(h) == 64

    def test_compute_merkle_hash_single_task_single_seed(self):
        """Edge case: single task with one seed produces a valid hash."""
        h = _compute_merkle_hash({"t": [0.75]})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# Merkle reproducibility (d40)
# ---------------------------------------------------------------------------

class TestMerkleReproducibility:
    def test_same_seed_scores_produce_same_hash(self):
        """Identical seed_scores always produce the same Merkle hash."""
        scores = {"task_a": [0.70, 0.71, 0.72], "task_b": [0.80, 0.81, 0.82]}
        h1 = _compute_merkle_hash(scores)
        h2 = _compute_merkle_hash(scores)
        assert h1 == h2

    def test_different_scores_produce_different_hash(self):
        """Even a tiny difference in scores changes the hash."""
        scores_a = {"task_a": [0.70, 0.71, 0.72]}
        scores_b = {"task_a": [0.70, 0.71, 0.73]}  # last score differs by 0.01
        h1 = _compute_merkle_hash(scores_a)
        h2 = _compute_merkle_hash(scores_b)
        assert h1 != h2

    def test_different_task_names_produce_different_hash(self):
        """Same numeric scores with different task names produce different hashes."""
        scores_a = {"task_a": [0.70]}
        scores_b = {"task_b": [0.70]}  # different key
        assert _compute_merkle_hash(scores_a) != _compute_merkle_hash(scores_b)

    def test_run_eval_reproducible_with_same_artifact_and_suite(self, artifact):
        """Two run_eval calls with the same artifact + synthetic path yield same hash."""
        suite = EvalSuite(suite_type="dante", tasks=("repro_task",), seeds=3)
        # Both calls use the deterministic synthetic fallback
        receipt1 = run_eval(artifact, suite, seeds=3)
        EvidenceLogger.clear()
        receipt2 = run_eval(artifact, suite, seeds=3)
        # Synthetic scores are deterministic from (task, artifact_id, seed) → same hash
        assert receipt1.merkle_hash == receipt2.merkle_hash

    def test_different_artifact_produces_different_hash(self, artifact, artifact_b):
        """Different model artifacts produce different hashes for the same tasks."""
        suite = EvalSuite(suite_type="dante", tasks=("same_task",), seeds=3)
        receipt_a = run_eval(artifact, suite, seeds=3)
        EvidenceLogger.clear()
        receipt_b = run_eval(artifact_b, suite, seeds=3)
        # Different artifact IDs → different synthetic scores → different hashes
        assert receipt_a.merkle_hash != receipt_b.merkle_hash


# ---------------------------------------------------------------------------
# Contamination check embedded in receipt (d40)
# ---------------------------------------------------------------------------

class TestContaminationEmbeddedInReceipt:
    def test_contamination_check_passed_true_when_no_hashes(self, artifact):
        """contamination_check_passed=True when contamination_hashes is None."""
        suite = EvalSuite(suite_type="dante", tasks=("clean_task",), seeds=2)
        receipt = run_eval(artifact, suite, seeds=2, contamination_hashes=None)
        assert receipt.contamination_check_passed is True

    def test_contamination_check_passed_in_to_dict(self, artifact):
        """contamination_check_passed field survives serialization to dict."""
        suite = EvalSuite(suite_type="dante", tasks=("task_a",), seeds=2)
        receipt = run_eval(artifact, suite, seeds=2)
        d = receipt.to_dict()
        assert "contamination_check_passed" in d
        assert isinstance(d["contamination_check_passed"], bool)

    def test_contamination_raises_and_no_receipt_returned(self, artifact):
        """EvalContaminationError is raised when task hash matches contamination_hashes."""
        task_name = "contaminated_task"
        task_hash = hashlib.sha256(task_name.encode()).hexdigest()
        contamination_hashes = frozenset([task_hash])

        suite = EvalSuite(suite_type="dante", tasks=(task_name,), seeds=1)
        with pytest.raises(EvalContaminationError):
            run_eval(artifact, suite, seeds=1, contamination_hashes=contamination_hashes)

    def test_contamination_check_passed_false_would_be_in_partial_receipt(self):
        """EvalReceipt can be constructed with contamination_check_passed=False."""
        # This verifies the dataclass supports the False state
        receipt = EvalReceipt(
            model_artifact_id="test",
            suite_type="dante",
            tasks=("t",),
            seed_scores={"t": [0.5]},
            mean_scores={"t": 0.5},
            ci_lower={"t": 0.5},
            ci_upper={"t": 0.5},
            contamination_check_passed=False,  # explicitly False
            merkle_hash="a" * 64,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            source="dante_suite",
        )
        assert receipt.contamination_check_passed is False
        assert receipt.to_dict()["contamination_check_passed"] is False

    def test_contamination_check_for_lm_eval_suite_type(self, artifact):
        """Contamination check runs when suite_type='lm_eval' too."""
        suite = EvalSuite(suite_type="lm_eval", tasks=("mmlu",), seeds=1)
        task_hash = hashlib.sha256(b"mmlu").hexdigest()
        contamination_hashes = frozenset([task_hash])

        with patch("dojo.eval.harness._run_lm_eval_seed") as mock_seed:
            with pytest.raises(EvalContaminationError):
                run_eval(
                    artifact, suite, seeds=1,
                    contamination_hashes=contamination_hashes,
                )
            # Contamination check must run BEFORE the eval
            mock_seed.assert_not_called()


# ---------------------------------------------------------------------------
# lm-eval task catalog completeness (d37)
# ---------------------------------------------------------------------------

class TestLmEvalTaskCatalogCompleteness:
    def test_catalog_has_eleven_standard_tasks(self):
        """LM_EVAL_STANDARD_TASKS contains exactly 11 canonical tasks."""
        assert len(LM_EVAL_STANDARD_TASKS) == 11

    def test_all_expected_tasks_present(self):
        """All expected task names are registered in the catalog."""
        expected = {
            "mmlu", "hellaswag", "arc_easy", "arc_challenge", "winogrande",
            "truthfulqa_mc1", "truthfulqa_mc2", "boolq", "gsm8k",
            "math_hard", "lambada_openai",
        }
        assert set(LM_EVAL_STANDARD_TASKS.keys()) == expected

    def test_default_suite_is_tuple(self):
        """LM_EVAL_DEFAULT_SUITE is a tuple (not a list)."""
        assert isinstance(LM_EVAL_DEFAULT_SUITE, tuple)

    def test_default_suite_has_tasks(self):
        """Default suite contains at least 4 tasks (expanded from 4 → 8 in Wave 33)."""
        assert len(LM_EVAL_DEFAULT_SUITE) >= 4

    def test_default_suite_tasks_are_strings(self):
        """Every item in LM_EVAL_DEFAULT_SUITE is a plain string."""
        for name in LM_EVAL_DEFAULT_SUITE:
            assert isinstance(name, str), f"Expected str, got {type(name)} for {name!r}"

    def test_default_suite_is_subset_of_standard_tasks(self):
        """Every default suite task exists in the standard catalog."""
        for name in LM_EVAL_DEFAULT_SUITE:
            assert name in LM_EVAL_STANDARD_TASKS, f"{name!r} not in standard catalog"

    def test_all_tasks_have_primary_metric_key(self):
        """Every catalog task has a non-empty primary_metric_key."""
        for name, task in LM_EVAL_STANDARD_TASKS.items():
            assert task.primary_metric_key, f"Task {name!r} has empty primary_metric_key"

    def test_all_tasks_have_description(self):
        """Every catalog task has a non-empty description string."""
        for name, task in LM_EVAL_STANDARD_TASKS.items():
            assert task.description, f"Task {name!r} has empty description"

    def test_all_tasks_are_frozen_dataclasses(self):
        """LmEvalTask instances must be immutable (frozen=True)."""
        for name in LM_EVAL_STANDARD_TASKS:
            task = get_lm_eval_task(name)
            with pytest.raises((AttributeError, TypeError)):
                task.name = "mutated"  # type: ignore[misc]

    def test_lenient_resolution_returns_stub_for_unknown(self):
        """Unknown task names get a stub LmEvalTask with acc,none as primary metric."""
        known, unknown = resolve_task_names_lenient(["mmlu", "custom_xyz_task"])
        assert "custom_xyz_task" in unknown
        stub = known[1]
        assert stub.primary_metric_key == "acc,none"
        assert stub.default_num_fewshot == 0


# ---------------------------------------------------------------------------
# EvalReceipt structure and serialization (d40)
# ---------------------------------------------------------------------------

class TestEvalReceiptStructure:
    def test_evaluated_at_is_iso8601_string(self, artifact):
        """EvalReceipt.evaluated_at is a valid ISO 8601 datetime string."""
        suite = EvalSuite(suite_type="dante", tasks=("t1",), seeds=2)
        receipt = run_eval(artifact, suite, seeds=2)
        # datetime.fromisoformat raises ValueError on malformed strings
        parsed = datetime.fromisoformat(receipt.evaluated_at)
        assert parsed.tzinfo is not None  # must be timezone-aware

    def test_to_dict_contains_all_required_fields(self, artifact):
        """EvalReceipt.to_dict() contains all expected top-level keys."""
        suite = EvalSuite(suite_type="dante", tasks=("t2",), seeds=2)
        receipt = run_eval(artifact, suite, seeds=2)
        d = receipt.to_dict()
        required_keys = {
            "model_artifact_id", "suite_type", "tasks", "seed_scores",
            "mean_scores", "ci_lower", "ci_upper", "contamination_check_passed",
            "merkle_hash", "evaluated_at", "source",
        }
        for key in required_keys:
            assert key in d, f"Missing key {key!r} in EvalReceipt.to_dict()"

    def test_to_dict_tasks_is_list(self, artifact):
        """tasks field in to_dict() is a list, not a tuple."""
        suite = EvalSuite(suite_type="dante", tasks=("t3", "t4"), seeds=2)
        receipt = run_eval(artifact, suite, seeds=2)
        d = receipt.to_dict()
        assert isinstance(d["tasks"], list)

    def test_seed_scores_count_matches_seeds_param(self, artifact):
        """seed_scores dict contains exactly `seeds` entries per task."""
        n_seeds = 4
        suite = EvalSuite(suite_type="dante", tasks=("task_seed_check",), seeds=n_seeds)
        receipt = run_eval(artifact, suite, seeds=n_seeds)
        for task, scores in receipt.seed_scores.items():
            assert len(scores) == n_seeds, f"Expected {n_seeds} scores for {task}, got {len(scores)}"

    def test_ci_lower_leq_mean_leq_ci_upper(self, artifact):
        """For all tasks: ci_lower <= mean_scores <= ci_upper.

        Uses a mocked _run_lm_eval_seed to guarantee stable, same-polarity scores
        so the judge-drift gate in run_eval does not trigger during the CI check.
        """
        suite = EvalSuite(suite_type="lm_eval", tasks=("mmlu", "hellaswag"), seeds=5)
        # Provide scores that are all above 0.5 (same polarity) → no drift
        with patch("dojo.eval.harness._run_lm_eval_seed") as mock_seed:
            mock_seed.return_value = (
                {"mmlu": 0.72, "hellaswag": 0.80},
                {"mmlu": "acc_norm,none", "hellaswag": "acc_norm,none"},
                True,
            )
            receipt = run_eval(artifact, suite, seeds=5)
        for task in receipt.tasks:
            lower = receipt.ci_lower[task]
            mean = receipt.mean_scores[task]
            upper = receipt.ci_upper[task]
            assert lower <= mean <= upper, (
                f"CI bounds violated for {task}: {lower} <= {mean} <= {upper}"
            )

    def test_receipt_is_fully_json_serializable(self, artifact):
        """EvalReceipt.to_dict() round-trips through JSON without loss."""
        suite = EvalSuite(suite_type="dante", tasks=("t7",), seeds=3)
        receipt = run_eval(artifact, suite, seeds=3)
        original_dict = receipt.to_dict()
        json_str = json.dumps(original_dict)
        restored_dict = json.loads(json_str)
        assert restored_dict["merkle_hash"] == original_dict["merkle_hash"]
        assert restored_dict["contamination_check_passed"] == original_dict["contamination_check_passed"]
        assert restored_dict["evaluated_at"] == original_dict["evaluated_at"]

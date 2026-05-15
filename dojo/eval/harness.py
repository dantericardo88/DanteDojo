"""Unified evaluation runner for DanteDojo — Phase 4.

Single entry point: run_eval(). Handles lm-eval, livebench, and dante suites.

Source hierarchy (in order of preference):
  1. lm_eval_real       — lm-eval installed, real benchmark scores
  2. lm_eval_synthetic  — lm-eval absent, synthetic fallback (CI without GPU)
  3. dante_suite        — domain-specific tasks (Empanada workflows)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import EvalContaminationError, EvalError, JudgeDriftError
from dojo.core.one_door import one_door
from dojo.core.types import ModelArtifact
from dojo.eval.judge_drift import JudgeDriftDetector
from dojo.eval.lm_eval_adapter import LmEvalRunResult, LmEvalTaskSpec, is_lm_eval_available, run_lm_eval
from dojo.eval.statistics import bootstrap_ci, seeds_are_stable


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalSuite:
    """Descriptor for an evaluation suite run."""

    suite_type: str         # "lm_eval", "livebench", "dante"
    tasks: Tuple[str, ...]
    seeds: int = 5


@dataclass(frozen=True)
class EvalReceipt:
    """Immutable record of a completed evaluation run.

    The `source` field declares the origin of the scores:
    - "lm_eval_real"            — real lm-eval benchmarks (GPU required)
    - "lm_eval_synthetic_fallback" — synthetic scores (lm-eval not installed)
    - "dante_suite"             — domain-specific Dante tasks
    """

    model_artifact_id: str
    suite_type: str
    tasks: Tuple[str, ...]
    seed_scores: Dict[str, List[float]]
    mean_scores: Dict[str, float]
    ci_lower: Dict[str, float]
    ci_upper: Dict[str, float]
    contamination_check_passed: bool
    merkle_hash: str
    evaluated_at: str
    source: str = field(default="dante_suite")
    metric_keys: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_artifact_id": self.model_artifact_id,
            "suite_type": self.suite_type,
            "tasks": list(self.tasks),
            "seed_scores": {k: list(v) for k, v in self.seed_scores.items()},
            "mean_scores": dict(self.mean_scores),
            "ci_lower": dict(self.ci_lower),
            "ci_upper": dict(self.ci_upper),
            "contamination_check_passed": self.contamination_check_passed,
            "merkle_hash": self.merkle_hash,
            "evaluated_at": self.evaluated_at,
            "source": self.source,
            "metric_keys": dict(self.metric_keys),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_merkle_hash(seed_scores: Dict[str, List[float]]) -> str:
    payload = json.dumps(seed_scores, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _synthetic_scores_for_task(task: str, artifact_id: str, seed: int) -> float:
    key = f"harness:{task}:{artifact_id}:{seed}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF
    return 0.45 + raw * 0.35


def _build_lm_eval_specs(tasks: Tuple[str, ...]) -> List[LmEvalTaskSpec]:
    """Build LmEvalTaskSpec list from task names using the catalog."""
    from dojo.eval.lm_eval_tasks import LM_EVAL_STANDARD_TASKS
    specs: List[LmEvalTaskSpec] = []
    for name in tasks:
        catalog = LM_EVAL_STANDARD_TASKS.get(name)
        if catalog is not None:
            specs.append(LmEvalTaskSpec(
                name=name,
                num_fewshot=catalog.default_num_fewshot,
                primary_metric_key=catalog.primary_metric_key,
                alt_metric_keys=catalog.alt_metric_keys,
            ))
        else:
            specs.append(LmEvalTaskSpec(
                name=name,
                num_fewshot=0,
                primary_metric_key="acc,none",
                alt_metric_keys=("acc_norm,none", "exact_match,none"),
            ))
    return specs


def _run_lm_eval_seed(
    model: ModelArtifact,
    suite: EvalSuite,
    seed: int,
    allow_synthetic: bool,
) -> Tuple[Dict[str, float], Dict[str, str], bool]:
    """Run lm_eval suite for one seed.

    Returns:
        (task_scores, metric_keys, is_synthetic)
    """
    specs = _build_lm_eval_specs(suite.tasks)
    results: Dict[str, LmEvalRunResult] = run_lm_eval(
        model,
        list(suite.tasks),
        seed=seed,
        task_configs=specs,
        allow_synthetic=allow_synthetic,
    )
    scores = {k: v.score for k, v in results.items()}
    metric_keys = {k: v.metric_key for k, v in results.items()}
    is_synthetic = any(v.is_synthetic for v in results.values())
    return scores, metric_keys, is_synthetic


def _run_single_seed(
    model: ModelArtifact,
    suite: EvalSuite,
    seed: int,
) -> Dict[str, float]:
    """Run the suite for a single seed. Returns {task: score}.

    Legacy path for dante/livebench suites — uses synthetic fallback freely.
    """
    if suite.suite_type in ("lm_eval", "livebench"):
        scores, _, _ = _run_lm_eval_seed(model, suite, seed, allow_synthetic=True)
        return scores

    # dante suite — use lm_eval adapter or synthetic fallback
    if is_lm_eval_available():
        scores, _, _ = _run_lm_eval_seed(model, suite, seed, allow_synthetic=True)
        return scores

    return {
        task: _synthetic_scores_for_task(task, model.artifact_id, seed)
        for task in suite.tasks
    }


# ---------------------------------------------------------------------------
# One-door entry
# ---------------------------------------------------------------------------

@one_door("eval.run")
def run_eval(
    model: ModelArtifact,
    suite: EvalSuite,
    seeds: int = 5,
    *,
    contamination_hashes: Optional[FrozenSet[str]] = None,
    allow_synthetic_fallback: bool = True,
) -> EvalReceipt:
    """Run evaluation suite and return a Merkle-stamped EvalReceipt.

    For lm_eval suite type:
    - If lm-eval installed → runs real benchmarks with per-task few-shot config.
    - If lm-eval absent and allow_synthetic_fallback=True → synthetic scores.
    - If lm-eval absent and allow_synthetic_fallback=False → EvalError raised.

    For dante suite type:
    - Uses lm-eval if available, else synthetic fallback.

    Always:
    - Checks contamination before running.
    - Computes bootstrap CIs over seed scores.
    - Checks seed stability (std > 2σ → EvalError).
    - Emits EVAL_START, EVAL_COMPLETE, and source evidence.
    - Returns EvalReceipt with merkle_hash and declared source.

    Args:
        model: The ModelArtifact to evaluate.
        suite: EvalSuite describing tasks and suite type.
        seeds: Number of seed runs to perform.
        contamination_hashes: Optional training fingerprints to check.
        allow_synthetic_fallback: When False, EvalError if lm-eval absent.

    Returns:
        EvalReceipt with all scores, CIs, provenance hash, and source.

    Raises:
        EvalContaminationError: If contamination is detected.
        EvalError: If seed scores are unstable or lm-eval unavailable with
            allow_synthetic_fallback=False.
    """
    logger = EvidenceLogger.get_instance()
    evaluated_at = datetime.now(timezone.utc).isoformat()

    # Default tasks for lm_eval suite with no tasks specified
    effective_suite = suite
    if suite.suite_type == "lm_eval" and not suite.tasks:
        from dojo.eval.lm_eval_tasks import LM_EVAL_DEFAULT_SUITE
        effective_suite = EvalSuite(
            suite_type=suite.suite_type,
            tasks=LM_EVAL_DEFAULT_SUITE,
            seeds=suite.seeds,
        )

    logger.emit(
        {
            "type": EvidenceType.EVAL_START.value,
            "data": {
                "model_artifact_id": model.artifact_id,
                "suite_type": effective_suite.suite_type,
                "tasks": list(effective_suite.tasks),
                "seeds": seeds,
                "evaluated_at": evaluated_at,
                "lm_eval_available": is_lm_eval_available(),
            },
        },
        organ="eval",
    )

    # --- Contamination check ---
    contamination_passed = True
    if contamination_hashes:
        for task in effective_suite.tasks:
            task_hash = hashlib.sha256(task.encode()).hexdigest()
            if task_hash in contamination_hashes:
                contamination_passed = False
                logger.emit(
                    {
                        "type": EvidenceType.EVAL_CONTAMINATION.value,
                        "data": {"task": task, "hash": task_hash},
                    },
                    organ="eval",
                )
                raise EvalContaminationError(
                    f"Contamination detected: eval task '{task}' hash appears in training data",
                    reason=f"task_hash={task_hash} found in contamination_hashes",
                )

    # --- Multi-seed evaluation ---
    seed_scores: Dict[str, List[float]] = {task: [] for task in effective_suite.tasks}
    all_metric_keys: Dict[str, str] = {}
    source_flags: List[bool] = []  # True = synthetic

    effective_seeds = seeds if seeds > 0 else effective_suite.seeds

    for i in range(effective_seeds):
        seed_val = 42 + i

        if effective_suite.suite_type in ("lm_eval", "livebench"):
            task_scores, metric_keys, is_syn = _run_lm_eval_seed(
                model, effective_suite, seed_val, allow_synthetic=allow_synthetic_fallback
            )
            source_flags.append(is_syn)
            if not all_metric_keys:
                all_metric_keys = metric_keys
        else:
            task_scores = _run_single_seed(model, effective_suite, seed_val)
            source_flags.append(not is_lm_eval_available())

        for task in effective_suite.tasks:
            score = task_scores.get(
                task, _synthetic_scores_for_task(task, model.artifact_id, seed_val)
            )
            seed_scores[task].append(score)

    # Determine source
    any_synthetic = any(source_flags)
    if effective_suite.suite_type in ("lm_eval", "livebench"):
        source = "lm_eval_synthetic_fallback" if any_synthetic else "lm_eval_real"
    else:
        source = "lm_eval_synthetic_fallback" if any_synthetic else "dante_suite"

    # Emit source event so downstream agents know which path was taken
    logger.emit(
        {
            "type": EvidenceType.EVAL_START.value,
            "data": {
                "event": "eval.source",
                "source": source,
                "tasks": list(effective_suite.tasks),
                "num_fewshot": {s.name: s.num_fewshot for s in _build_lm_eval_specs(effective_suite.tasks)},
                "model_artifact_id": model.artifact_id,
            },
        },
        organ="eval",
    )

    # --- Stability check ---
    from dojo.eval.statistics import seed_variance
    for task, scores in seed_scores.items():
        if not seeds_are_stable(scores, max_std=2.0):
            std = seed_variance(scores)
            raise EvalError(
                f"Seed instability detected for task '{task}': std={std:.4f} > 2.0",
                reason=f"Model scores vary too much across seeds. Std={std:.4f}.",
            )

    # --- Judge drift gate (multi-seed, minimum 5 tasks required) ---
    # Compares seed-score polarities across tasks to detect systematic judge
    # drift.  Each task's per-seed scores are projected onto a binary polarity
    # (>= 0.5 → "positive", < 0.5 → "negative").  The first seed acts as the
    # reference "gold label"; each subsequent seed casts a "vote".  Drift is
    # declared when fewer than 80% of votes agree with the reference — but only
    # once at least 5 vote-pairs have been recorded (low-task runs lack
    # statistical power to distinguish noise from genuine drift).
    _MIN_DRIFT_RECORDS = 5
    if effective_seeds > 1:
        drift_detector = JudgeDriftDetector(
            judge_family="eval.seed_agreement",
            window_days=1,
        )
        for task, scores in seed_scores.items():
            # Use the first seed's polarity as the reference "gold label"
            reference_polarity = "positive" if scores[0] >= 0.5 else "negative"
            for score in scores[1:]:
                vote = "positive" if score >= 0.5 else "negative"
                drift_detector.record_agreement(vote, reference_polarity)

        # Only gate when there are enough records to be meaningful
        total_records = sum(
            len(scores) - 1 for scores in seed_scores.values()
        )
        if total_records >= _MIN_DRIFT_RECORDS:
            try:
                drift_detector.check_drift()
            except JudgeDriftError as exc:
                logger.emit(
                    {
                        "type": EvidenceType.EVAL_JUDGE_DRIFT.value,
                        "data": {
                            "model_artifact_id": model.artifact_id,
                            "suite_type": effective_suite.suite_type,
                            "agreement_rate": drift_detector.agreement_rate(),
                            "threshold": JudgeDriftDetector.MIN_AGREEMENT_THRESHOLD,
                            "evaluated_at": evaluated_at,
                            "error": str(exc),
                        },
                    },
                    organ="eval",
                )
                raise

    # --- Bootstrap CIs ---
    mean_scores: Dict[str, float] = {}
    ci_lower: Dict[str, float] = {}
    ci_upper: Dict[str, float] = {}

    for task, scores in seed_scores.items():
        if len(scores) >= 2:
            mean_val, lower, upper = bootstrap_ci(scores, n_bootstrap=1000, confidence=0.95, seed=42)
        else:
            mean_val = scores[0] if scores else 0.0
            lower = mean_val
            upper = mean_val
        mean_scores[task] = mean_val
        ci_lower[task] = lower
        ci_upper[task] = upper

    merkle_hash = _compute_merkle_hash(seed_scores)

    receipt = EvalReceipt(
        model_artifact_id=model.artifact_id,
        suite_type=effective_suite.suite_type,
        tasks=tuple(effective_suite.tasks),
        seed_scores=seed_scores,
        mean_scores=mean_scores,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        contamination_check_passed=contamination_passed,
        merkle_hash=merkle_hash,
        evaluated_at=evaluated_at,
        source=source,
        metric_keys=all_metric_keys,
    )

    logger.emit(
        {
            "type": EvidenceType.EVAL_COMPLETE.value,
            "data": {
                "model_artifact_id": model.artifact_id,
                "suite_type": effective_suite.suite_type,
                "merkle_hash": merkle_hash,
                "mean_scores": mean_scores,
                "contamination_check_passed": contamination_passed,
                "source": source,
                "evaluated_at": evaluated_at,
            },
        },
        organ="eval",
    )

    return receipt


__all__ = [
    "EvalSuite",
    "EvalReceipt",
    "run_eval",
]

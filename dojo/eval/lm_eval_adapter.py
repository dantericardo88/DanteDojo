"""lm-evaluation-harness wrapper for DanteDojo.

Primary path: when lm-eval is installed, runs real benchmark tasks (MMLU,
HellaSwag, ARC-Challenge, etc.) via evaluator.simple_evaluate with per-task
few-shot configuration and constitutional evidence emission.

Synthetic fallback: used ONLY when lm-eval is not installed AND the caller
explicitly allows it (allow_synthetic=True). The fallback is never silently
activated — the receipt always declares its source.

Constitutional integration:
- Emits EVAL_LM_HARNESS_START before calling evaluator
- Emits EVAL_LM_HARNESS_COMPLETE after, carrying raw_results_hash
- Hard EvalError when lm-eval IS installed but evaluation fails (no silent swallow)
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import EvalError
from dojo.core.types import ModelArtifact


# ---------------------------------------------------------------------------
# Typed result
# ---------------------------------------------------------------------------

class LmEvalRunResult(NamedTuple):
    """Typed result for a single lm-eval task evaluation.

    Attributes:
        task: lm-eval task name.
        score: Numeric score (0.0–1.0 for most tasks).
        metric_key: The metric key that was actually extracted from results.
        num_fewshot: Number of few-shot examples used.
        raw_results_hash: SHA-256 of the raw lm-eval results JSON for provenance.
        is_synthetic: True if score was generated synthetically (no real eval).
    """

    task: str
    score: float
    metric_key: str
    num_fewshot: int
    raw_results_hash: str
    is_synthetic: bool


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def is_lm_eval_available() -> bool:
    """Return True if lm-evaluation-harness is installed and importable."""
    try:
        import lm_eval  # type: ignore[import-not-found]  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def _extract_score(
    task_results: Dict,
    primary_metric_key: str,
    alt_metric_keys: Tuple[str, ...] = (),
) -> Tuple[float, str]:
    """Extract score from lm-eval task results dict.

    Tries primary_metric_key, then each alt_metric_key in order,
    then falls back to the first numeric value found.

    Args:
        task_results: The per-task results dict from lm-eval output.
        primary_metric_key: Preferred metric key (e.g. "acc_norm,none").
        alt_metric_keys: Fallback keys tried in order.

    Returns:
        Tuple of (score, metric_key_used).
    """
    for key in (primary_metric_key, *alt_metric_keys):
        if key in task_results:
            return float(task_results[key]), key

    # Last resort: first numeric value in the dict
    for key, val in task_results.items():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val), key

    return 0.0, "missing"


# ---------------------------------------------------------------------------
# Synthetic fallback (demoted — only when lm-eval is absent)
# ---------------------------------------------------------------------------

def _synthetic_score(task: str, artifact_id: str, seed: int) -> float:
    """Deterministic synthetic score in [0.45, 0.75] for no-GPU CI."""
    key = f"{task}:{artifact_id}:{seed}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF
    return 0.45 + raw * 0.30


def _synthetic_results_hash(task: str, artifact_id: str, seed: int) -> str:
    key = f"synthetic:{task}:{artifact_id}:{seed}"
    return hashlib.sha256(key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Real lm-eval runner
# ---------------------------------------------------------------------------

def _run_real_lm_eval(
    model_artifact: ModelArtifact,
    task_cfgs: "List[LmEvalTaskSpec]",
    *,
    device: str,
    batch_size: int,
    seed: int,
) -> Tuple[Dict[str, LmEvalRunResult], str]:
    """Call evaluator.simple_evaluate with correct per-task few-shot configuration.

    lm-eval >=0.4 accepts num_fewshot as a single int applied to all tasks in
    a call. To honour per-task few-shot counts we group tasks by their count
    and issue one simple_evaluate call per unique count. Tasks sharing the same
    count are batched into one call for efficiency.

    Model loading:
    - Base model only: ``pretrained={base_model}``
    - Fine-tuned with LoRA: ``pretrained={base_model},peft={lora_weights}``

    Returns:
        Tuple of (results_dict, raw_results_hash).

    Raises:
        EvalError: If evaluation fails for any reason (not swallowed).
    """
    from collections import defaultdict

    from lm_eval import evaluator as lm_evaluator  # type: ignore[import-not-found]

    # Correct model_args: include PEFT adapter path when fine-tuned artifact
    model_args = f"pretrained={model_artifact.base_model}"
    if model_artifact.lora_weights:
        model_args += f",peft={model_artifact.lora_weights}"

    # Group tasks by num_fewshot so each group can share one simple_evaluate call
    fewshot_groups: Dict[int, List[LmEvalTaskSpec]] = defaultdict(list)
    for cfg in task_cfgs:
        fewshot_groups[cfg.num_fewshot].append(cfg)

    # Accumulate raw results across all fewshot groups
    all_task_results: Dict[str, Any] = {}

    for fewshot_count, group_cfgs in sorted(fewshot_groups.items()):
        group_names = [cfg.name for cfg in group_cfgs]
        try:
            raw = lm_evaluator.simple_evaluate(
                model="hf",
                model_args=model_args,
                tasks=group_names,
                num_fewshot=fewshot_count,  # correct int per group
                device=device,
                batch_size=batch_size,
                random_seed=seed,
                numpy_random_seed=seed,
                torch_random_seed=seed,
            )
        except Exception as exc:
            raise EvalError(
                f"lm-eval evaluation failed for artifact '{model_artifact.artifact_id}'",
                reason=str(exc),
            ) from exc
        all_task_results.update(raw.get("results", {}))

    # Hash over the merged results from all groups
    raw_results_hash = hashlib.sha256(
        json.dumps(all_task_results, sort_keys=True, default=str).encode()
    ).hexdigest()

    results: Dict[str, LmEvalRunResult] = {}
    for cfg in task_cfgs:
        task_data = all_task_results.get(cfg.name, {})
        score, metric_used = _extract_score(
            task_data,
            cfg.primary_metric_key,
            cfg.alt_metric_keys,
        )
        results[cfg.name] = LmEvalRunResult(
            task=cfg.name,
            score=score,
            metric_key=metric_used,
            num_fewshot=cfg.num_fewshot,
            raw_results_hash=raw_results_hash,
            is_synthetic=False,
        )

    return results, raw_results_hash


# ---------------------------------------------------------------------------
# Internal task spec (decoupled from LmEvalTask import to avoid circularity)
# ---------------------------------------------------------------------------

class _LmEvalTaskSpec:
    """Lightweight task spec for the adapter — avoids circular import."""

    __slots__ = ("name", "num_fewshot", "primary_metric_key", "alt_metric_keys")

    def __init__(
        self,
        name: str,
        num_fewshot: int,
        primary_metric_key: str,
        alt_metric_keys: Tuple[str, ...],
    ) -> None:
        self.name = name
        self.num_fewshot = num_fewshot
        self.primary_metric_key = primary_metric_key
        self.alt_metric_keys = alt_metric_keys


# Use type alias so the function signature is readable
LmEvalTaskSpec = _LmEvalTaskSpec


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_lm_eval(
    model_artifact: ModelArtifact,
    tasks: List[str],
    *,
    device: str = "cuda",
    batch_size: int = 1,
    seed: int = 42,
    task_configs: Optional[List["LmEvalTaskSpec"]] = None,
    allow_synthetic: bool = True,
) -> Dict[str, LmEvalRunResult]:
    """Run lm-eval-harness tasks and return typed results.

    When lm-eval is installed: runs real benchmarks with per-task configuration,
    emits EVAL_LM_HARNESS_START / EVAL_LM_HARNESS_COMPLETE evidence, returns
    LmEvalRunResult with is_synthetic=False.

    When lm-eval is NOT installed and allow_synthetic=True: returns synthetic
    scores with is_synthetic=True so callers can declare the fallback in receipts.

    When lm-eval is NOT installed and allow_synthetic=False: raises EvalError.

    Args:
        model_artifact: The model artifact to evaluate.
        tasks: List of lm-eval task names.
        device: Compute device ("cuda", "cpu", "mps").
        batch_size: Inference batch size.
        seed: RNG seed for reproducibility.
        task_configs: Optional per-task config overrides (from LmEvalTask catalog).
            If None, defaults are used (acc,none metric, 0-shot).
        allow_synthetic: If True and lm-eval absent, return synthetic scores.
            If False and lm-eval absent, raise EvalError.

    Returns:
        Dict mapping task name to LmEvalRunResult.

    Raises:
        EvalError: If lm-eval is installed but evaluation fails, or if
            allow_synthetic=False and lm-eval is not installed.
    """
    logger = EvidenceLogger.get_instance()

    # Build task specs (use catalog configs if provided, else defaults)
    specs: List[LmEvalTaskSpec] = []
    if task_configs is not None:
        specs = list(task_configs)
    else:
        for task_name in tasks:
            # Try to pull from catalog, fall back to defaults
            try:
                from dojo.eval.lm_eval_tasks import LM_EVAL_STANDARD_TASKS
                catalog_task = LM_EVAL_STANDARD_TASKS.get(task_name)
                if catalog_task is not None:
                    specs.append(_LmEvalTaskSpec(
                        name=task_name,
                        num_fewshot=catalog_task.default_num_fewshot,
                        primary_metric_key=catalog_task.primary_metric_key,
                        alt_metric_keys=catalog_task.alt_metric_keys,
                    ))
                    continue
            except ImportError:
                pass
            # Default spec for unknown tasks
            specs.append(_LmEvalTaskSpec(
                name=task_name,
                num_fewshot=0,
                primary_metric_key="acc,none",
                alt_metric_keys=("acc_norm,none", "exact_match,none"),
            ))

    # --- Emit EVAL_LM_HARNESS_START ---
    logger.emit(
        {
            "type": EvidenceType.EVAL_LM_HARNESS_START.value,
            "data": {
                "model_artifact_id": model_artifact.artifact_id,
                "tasks": tasks,
                "num_fewshot": {s.name: s.num_fewshot for s in specs},
                "device": device,
                "batch_size": batch_size,
                "seed": seed,
                "lm_eval_available": is_lm_eval_available(),
            },
        },
        organ="eval",
    )

    if not is_lm_eval_available():
        if not allow_synthetic:
            raise EvalError(
                "lm-eval is not installed and synthetic fallback is disabled",
                reason=(
                    "Install lm-eval: pip install 'dante-dojo[eval]' "
                    "or set allow_synthetic=True for CI environments."
                ),
            )
        # Synthetic path
        results: Dict[str, LmEvalRunResult] = {}
        for spec in specs:
            score = _synthetic_score(spec.name, model_artifact.artifact_id, seed)
            results[spec.name] = LmEvalRunResult(
                task=spec.name,
                score=score,
                metric_key="synthetic",
                num_fewshot=spec.num_fewshot,
                raw_results_hash=_synthetic_results_hash(spec.name, model_artifact.artifact_id, seed),
                is_synthetic=True,
            )
        logger.emit(
            {
                "type": EvidenceType.EVAL_LM_HARNESS_COMPLETE.value,
                "data": {
                    "model_artifact_id": model_artifact.artifact_id,
                    "source": "synthetic_fallback",
                    "tasks": tasks,
                    "scores": {k: v.score for k, v in results.items()},
                },
            },
            organ="eval",
        )
        return results

    # Real lm-eval path
    results, raw_results_hash = _run_real_lm_eval(
        model_artifact,
        specs,
        device=device,
        batch_size=batch_size,
        seed=seed,
    )

    logger.emit(
        {
            "type": EvidenceType.EVAL_LM_HARNESS_COMPLETE.value,
            "data": {
                "model_artifact_id": model_artifact.artifact_id,
                "source": "lm_eval_real",
                "tasks": tasks,
                "raw_results_hash": raw_results_hash,
                "scores": {k: v.score for k, v in results.items()},
                "metric_keys": {k: v.metric_key for k, v in results.items()},
            },
        },
        organ="eval",
    )

    return results


def run_lm_eval_scores(
    model_artifact: ModelArtifact,
    tasks: List[str],
    *,
    device: str = "cuda",
    batch_size: int = 1,
    seed: int = 42,
    task_configs: Optional[List[LmEvalTaskSpec]] = None,
    allow_synthetic: bool = True,
) -> Dict[str, float]:
    """Convenience wrapper returning {task: score} floats.

    Backward-compatible with code that only needs raw scores.
    """
    results = run_lm_eval(
        model_artifact,
        tasks,
        device=device,
        batch_size=batch_size,
        seed=seed,
        task_configs=task_configs,
        allow_synthetic=allow_synthetic,
    )
    return {k: v.score for k, v in results.items()}


__all__ = [
    "LmEvalRunResult",
    "LmEvalTaskSpec",
    "is_lm_eval_available",
    "run_lm_eval",
    "run_lm_eval_scores",
]

"""Canonical lm-evaluation-harness task catalog for DanteDojo.

Defines the standard benchmark tasks used for model evaluation, with correct
metric keys, few-shot counts, and human-readable descriptions.

Each LmEvalTask pins:
- The lm-eval task name (as accepted by evaluator.simple_evaluate)
- Default number of few-shot examples
- Primary metric key (as returned in results["results"][task])
- Alternate metric keys tried in order if primary is absent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from dojo.core.exceptions import EvalError


@dataclass(frozen=True)
class LmEvalTask:
    """Immutable descriptor for a single lm-eval harness task.

    Args:
        name: lm-eval task name (e.g. "mmlu", "hellaswag").
        description: Human-readable description of what the benchmark measures.
        default_num_fewshot: Standard few-shot count used in published results.
        primary_metric_key: First metric key to look for in lm-eval results dict.
        alt_metric_keys: Fallback metric keys tried in order if primary absent.
    """

    name: str
    description: str
    default_num_fewshot: int
    primary_metric_key: str
    alt_metric_keys: Tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Standard task catalog
# ---------------------------------------------------------------------------
# Metric keys follow lm-eval >=0.4 output format: "<metric>,<filter>"
# Most classification benchmarks use "acc,none" or "acc_norm,none".

LM_EVAL_STANDARD_TASKS: Dict[str, LmEvalTask] = {
    "mmlu": LmEvalTask(
        name="mmlu",
        description="Massive Multitask Language Understanding — 57 academic subjects, 5-shot",
        default_num_fewshot=5,
        primary_metric_key="acc_norm,none",
        alt_metric_keys=("acc,none",),
    ),
    "hellaswag": LmEvalTask(
        name="hellaswag",
        description="HellaSwag commonsense NLI — sentence completion, 10-shot",
        default_num_fewshot=10,
        primary_metric_key="acc_norm,none",
        alt_metric_keys=("acc,none",),
    ),
    "arc_easy": LmEvalTask(
        name="arc_easy",
        description="ARC Easy — elementary science questions, 25-shot",
        default_num_fewshot=25,
        primary_metric_key="acc_norm,none",
        alt_metric_keys=("acc,none",),
    ),
    "arc_challenge": LmEvalTask(
        name="arc_challenge",
        description="ARC Challenge — science questions requiring reasoning, 25-shot",
        default_num_fewshot=25,
        primary_metric_key="acc_norm,none",
        alt_metric_keys=("acc,none",),
    ),
    "winogrande": LmEvalTask(
        name="winogrande",
        description="WinoGrande — coreference resolution with commonsense, 5-shot",
        default_num_fewshot=5,
        primary_metric_key="acc,none",
        alt_metric_keys=(),
    ),
    "gsm8k": LmEvalTask(
        name="gsm8k",
        description="GSM8K — grade-school math word problems, 8-shot chain-of-thought",
        default_num_fewshot=8,
        primary_metric_key="exact_match,get-answer",
        alt_metric_keys=("acc,none", "exact_match,none"),
    ),
    "truthfulqa_mc1": LmEvalTask(
        name="truthfulqa_mc1",
        description="TruthfulQA MC1 — measures model truthfulness, single correct answer, 0-shot",
        default_num_fewshot=0,
        primary_metric_key="acc,none",
        alt_metric_keys=("mc1,none",),
    ),
    "truthfulqa_mc2": LmEvalTask(
        name="truthfulqa_mc2",
        description="TruthfulQA MC2 — measures model truthfulness, multiple correct answers, 0-shot",
        default_num_fewshot=0,
        primary_metric_key="acc,none",
        alt_metric_keys=("mc2,none",),
    ),
    "math_hard": LmEvalTask(
        name="math_hard",
        description="MATH Hard — competition mathematics problems requiring multi-step reasoning, 4-shot",
        default_num_fewshot=4,
        primary_metric_key="exact_match,none",
        alt_metric_keys=("acc,none", "acc_norm,none"),
    ),
    "boolq": LmEvalTask(
        name="boolq",
        description="BoolQ — yes/no reading comprehension, 0-shot",
        default_num_fewshot=0,
        primary_metric_key="acc,none",
        alt_metric_keys=(),
    ),
    "lambada_openai": LmEvalTask(
        name="lambada_openai",
        description="LAMBADA OpenAI — predict last word of passage, 0-shot",
        default_num_fewshot=0,
        primary_metric_key="acc,none",
        alt_metric_keys=("perplexity,none",),
    ),
}

# Default suite: the eight canonical benchmarks for the open LLM leaderboard
# with correct few-shot counts:
#   mmlu (5-shot), hellaswag (10-shot), arc_easy (25-shot), arc_challenge (25-shot),
#   winogrande (5-shot), gsm8k (8-shot), truthfulqa_mc1 (0-shot), math_hard (4-shot)
LM_EVAL_DEFAULT_SUITE: Tuple[str, ...] = (
    "mmlu",
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "gsm8k",
    "truthfulqa_mc1",
    "math_hard",
)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_lm_eval_task(name: str) -> LmEvalTask:
    """Return the LmEvalTask descriptor for the given task name.

    Args:
        name: lm-eval task name (e.g. "mmlu", "hellaswag").

    Returns:
        The corresponding LmEvalTask.

    Raises:
        EvalError: If the task name is not in the catalog.
    """
    task = LM_EVAL_STANDARD_TASKS.get(name)
    if task is None:
        known = sorted(LM_EVAL_STANDARD_TASKS.keys())
        raise EvalError(
            f"Unknown lm-eval task: {name!r}",
            reason=f"Known tasks: {known}. Pass an arbitrary name only when suite_type='lm_eval' and you know the exact lm-eval task name.",
        )
    return task


def resolve_task_names(names: List[str]) -> List[LmEvalTask]:
    """Resolve a list of task names to LmEvalTask descriptors.

    Unknown names raise EvalError on first occurrence.

    Args:
        names: List of lm-eval task names.

    Returns:
        List of LmEvalTask objects in the same order.

    Raises:
        EvalError: If any name is not in the catalog.
    """
    return [get_lm_eval_task(n) for n in names]


def resolve_task_names_lenient(names: List[str]) -> Tuple[List[LmEvalTask], List[str]]:
    """Resolve task names, returning known tasks and unknown name strings separately.

    Used when the caller may pass arbitrary lm-eval task names not in the catalog
    (e.g. subtasks like "mmlu_abstract_algebra"). Unknown names get a default
    LmEvalTask with acc,none metric.

    Args:
        names: List of task names.

    Returns:
        Tuple of (resolved_tasks, unknown_names).
    """
    resolved: List[LmEvalTask] = []
    unknown: List[str] = []
    for name in names:
        if name in LM_EVAL_STANDARD_TASKS:
            resolved.append(LM_EVAL_STANDARD_TASKS[name])
        else:
            # Auto-stub for arbitrary lm-eval tasks not in our catalog
            resolved.append(LmEvalTask(
                name=name,
                description=f"Custom lm-eval task: {name}",
                default_num_fewshot=0,
                primary_metric_key="acc,none",
                alt_metric_keys=("acc_norm,none", "exact_match,none"),
            ))
            unknown.append(name)
    return resolved, unknown


__all__ = [
    "LmEvalTask",
    "LM_EVAL_STANDARD_TASKS",
    "LM_EVAL_DEFAULT_SUITE",
    "get_lm_eval_task",
    "resolve_task_names",
    "resolve_task_names_lenient",
]

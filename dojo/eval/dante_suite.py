"""Dante-specific eval suite definitions.

This module defines the canonical set of evaluation tasks for DanteDojo models,
aligned to Real Empanada business workflows. Each DanteEvalSpec pins the task,
sample count, scoring metric, passing threshold, and optional judge LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from dojo.core.exceptions import EvalError


class DanteEvalTask(str, Enum):
    """Enumeration of all Dante-specific evaluation tasks."""

    EMPANADA_BOOKKEEPING_F1 = "empanada_bookkeeping_f1"
    B2B_EMAIL_QUALITY = "b2b_email_quality"
    SOP_EXTRACTION_FAITHFULNESS = "sop_extraction_faithfulness"
    CODE_REVIEW_ACTIONABILITY = "code_review_actionability"
    OCR_CLEANUP_EXACT_MATCH = "ocr_cleanup_exact_match"


@dataclass(frozen=True)
class DanteEvalSpec:
    """Immutable specification for a single Dante evaluation task."""

    task: DanteEvalTask
    sample_count: int
    metric: str           # e.g. "f1", "quality_score", "faithfulness"
    passing_threshold: float
    judge_family: Optional[str]   # None = no judge LLM needed
    judge_sha256: Optional[str]   # pinned hash for judge model


DANTE_EVAL_SPECS: Dict[DanteEvalTask, DanteEvalSpec] = {
    DanteEvalTask.EMPANADA_BOOKKEEPING_F1: DanteEvalSpec(
        task=DanteEvalTask.EMPANADA_BOOKKEEPING_F1,
        sample_count=1000,
        metric="f1",
        passing_threshold=0.75,
        judge_family=None,
        judge_sha256=None,
    ),
    DanteEvalTask.B2B_EMAIL_QUALITY: DanteEvalSpec(
        task=DanteEvalTask.B2B_EMAIL_QUALITY,
        sample_count=200,
        metric="quality_score",
        passing_threshold=0.70,
        judge_family="claude-sonnet-4-6",
        judge_sha256=None,  # pinned in production
    ),
    DanteEvalTask.SOP_EXTRACTION_FAITHFULNESS: DanteEvalSpec(
        task=DanteEvalTask.SOP_EXTRACTION_FAITHFULNESS,
        sample_count=500,
        metric="faithfulness",
        passing_threshold=0.80,
        judge_family="claude-sonnet-4-6",
        judge_sha256=None,  # pinned in production
    ),
    DanteEvalTask.CODE_REVIEW_ACTIONABILITY: DanteEvalSpec(
        task=DanteEvalTask.CODE_REVIEW_ACTIONABILITY,
        sample_count=300,
        metric="actionability_score",
        passing_threshold=0.65,
        judge_family="claude-sonnet-4-6",
        judge_sha256=None,  # pinned in production
    ),
    DanteEvalTask.OCR_CLEANUP_EXACT_MATCH: DanteEvalSpec(
        task=DanteEvalTask.OCR_CLEANUP_EXACT_MATCH,
        sample_count=800,
        metric="exact_match",
        passing_threshold=0.85,
        judge_family=None,
        judge_sha256=None,
    ),
}


def get_spec(task: DanteEvalTask) -> DanteEvalSpec:
    """Return the DanteEvalSpec for a given task.

    Args:
        task: The DanteEvalTask enum value to look up.

    Returns:
        The corresponding DanteEvalSpec.

    Raises:
        EvalError: If no spec is registered for the given task.
    """
    spec = DANTE_EVAL_SPECS.get(task)
    if spec is None:
        raise EvalError(
            f"No eval spec registered for task: {task!r}",
            reason=f"Valid tasks: {[t.value for t in DanteEvalTask]}",
        )
    return spec


__all__ = [
    "DanteEvalTask",
    "DanteEvalSpec",
    "DANTE_EVAL_SPECS",
    "get_spec",
]

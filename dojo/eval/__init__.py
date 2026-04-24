"""DanteDojo eval subsystem — Phase 4."""

from __future__ import annotations

from dojo.eval.dante_suite import DANTE_EVAL_SPECS, DanteEvalSpec, DanteEvalTask, get_spec
from dojo.eval.harness import EvalReceipt, EvalSuite, run_eval
from dojo.eval.judge_drift import JudgeDriftDetector
from dojo.eval.lm_eval_adapter import is_lm_eval_available, run_lm_eval
from dojo.eval.statistics import bootstrap_ci, seed_variance, seeds_are_stable

__all__ = [
    "DANTE_EVAL_SPECS",
    "DanteEvalSpec",
    "DanteEvalTask",
    "EvalReceipt",
    "EvalSuite",
    "JudgeDriftDetector",
    "bootstrap_ci",
    "get_spec",
    "is_lm_eval_available",
    "run_eval",
    "run_lm_eval",
    "seed_variance",
    "seeds_are_stable",
]

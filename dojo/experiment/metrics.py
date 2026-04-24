"""Metric aggregation and comparison utilities for DanteDojo experiment tracking."""

from __future__ import annotations

import math
from typing import Dict, List


def aggregate_metrics(
    runs: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across multiple runs.

    Args:
        runs: List of metric dicts, one per run. Each dict maps metric name
              to float value. Keys may differ across runs; missing values are
              ignored for that metric.

    Returns:
        Dict mapping metric_name → {"mean": X, "std": Y, "min": Z, "max": W}.
        An empty list of runs returns an empty dict.
    """
    if not runs:
        return {}

    # Collect all metric names across all runs
    all_keys: set[str] = set()
    for run in runs:
        all_keys.update(run.keys())

    result: Dict[str, Dict[str, float]] = {}

    for key in sorted(all_keys):
        values = [run[key] for run in runs if key in run]
        if not values:
            continue

        n = len(values)
        mean_val = sum(values) / n
        variance = sum((v - mean_val) ** 2 for v in values) / n if n > 1 else 0.0
        std_val = math.sqrt(variance)
        min_val = min(values)
        max_val = max(values)

        result[key] = {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
        }

    return result


def compare_runs(
    baseline: Dict[str, float],
    candidate: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """Return per-metric comparison between a baseline and candidate run.

    For each metric present in either dict, computes the absolute delta and
    relative delta percentage.

    Args:
        baseline: Dict of metric_name → float for the baseline model/run.
        candidate: Dict of metric_name → float for the candidate model/run.

    Returns:
        Dict mapping metric_name → {
            "baseline": X,
            "candidate": Y,
            "delta": Y - X,
            "delta_pct": (Y - X) / X * 100 if X != 0 else 0.0,
        }.
    """
    all_keys = set(baseline.keys()) | set(candidate.keys())
    result: Dict[str, Dict[str, float]] = {}

    for key in sorted(all_keys):
        base_val = baseline.get(key, 0.0)
        cand_val = candidate.get(key, 0.0)
        delta = cand_val - base_val
        delta_pct = (delta / base_val * 100.0) if base_val != 0.0 else 0.0

        result[key] = {
            "baseline": base_val,
            "candidate": cand_val,
            "delta": delta,
            "delta_pct": delta_pct,
        }

    return result


__all__ = [
    "aggregate_metrics",
    "compare_runs",
]

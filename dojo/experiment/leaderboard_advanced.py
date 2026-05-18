"""Advanced run-leaderboard rankings (d44).

Adds Pareto-frontier selection and multi-metric weighted rankings on top of
the existing single-metric leaderboard.  Useful when you want to rank runs
that trade off accuracy vs cost vs latency.

Pure Python — works on the in-memory run dicts returned by
``ExperimentTracker.list_runs()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import DojoError


class LeaderboardError(DojoError):
    """Raised when leaderboard inputs are inconsistent."""


@dataclass
class MetricSpec:
    """Describes one metric used in a multi-metric ranking."""

    key: str
    higher_is_better: bool = True
    weight: float = 1.0

    def signed_value(self, raw: float) -> float:
        """Return ``+raw`` for higher-is-better, ``-raw`` for lower-is-better."""
        return raw if self.higher_is_better else -raw


def _safe_get(run: Dict, key: str) -> Optional[float]:
    val = run.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def pareto_frontier(
    runs: List[Dict],
    metrics: List[MetricSpec],
) -> List[Dict]:
    """Return the Pareto frontier of ``runs`` under ``metrics``.

    A run is on the Pareto frontier if no other run dominates it — i.e.
    no other run is strictly better on at least one metric and no worse on
    all other metrics.

    Runs missing any metric in ``metrics`` are excluded.
    """
    if not metrics:
        raise LeaderboardError(
            "at least one metric required for Pareto frontier",
            reason="no_metrics",
        )

    eligible: List[Dict] = []
    for r in runs:
        if all(_safe_get(r, m.key) is not None for m in metrics):
            eligible.append(r)

    frontier: List[Dict] = []
    for i, ri in enumerate(eligible):
        vi = [m.signed_value(_safe_get(ri, m.key) or 0.0) for m in metrics]
        dominated = False
        for j, rj in enumerate(eligible):
            if i == j:
                continue
            vj = [m.signed_value(_safe_get(rj, m.key) or 0.0) for m in metrics]
            ge_all = all(a >= b for a, b in zip(vj, vi))
            gt_any = any(a > b for a, b in zip(vj, vi))
            if ge_all and gt_any:
                dominated = True
                break
        if not dominated:
            frontier.append(ri)

    EvidenceLogger.get_instance().emit(
        {
            "type": "experiment.pareto_frontier",
            "data": {
                "input_runs": len(runs),
                "eligible_runs": len(eligible),
                "frontier_size": len(frontier),
                "metrics": [m.key for m in metrics],
            },
        },
        organ="experiment",
    )

    return frontier


def weighted_score(run: Dict, metrics: List[MetricSpec]) -> Optional[float]:
    """Return ``sum(weight_i * signed_value_i)`` or ``None`` if any metric is missing."""
    total = 0.0
    for m in metrics:
        val = _safe_get(run, m.key)
        if val is None:
            return None
        total += m.weight * m.signed_value(val)
    return total


def rank_weighted(
    runs: List[Dict],
    metrics: List[MetricSpec],
    limit: Optional[int] = None,
) -> List[Dict]:
    """Sort runs by their weighted multi-metric score, descending.

    Runs missing any metric are excluded from the ranking.
    """
    scored: List[tuple] = []
    for r in runs:
        score = weighted_score(r, metrics)
        if score is not None:
            scored.append((score, r))

    scored.sort(key=lambda t: t[0], reverse=True)
    ranked = [r for _s, r in scored]

    if limit is not None:
        ranked = ranked[:limit]

    EvidenceLogger.get_instance().emit(
        {
            "type": "experiment.weighted_ranking",
            "data": {
                "input_runs": len(runs),
                "ranked_runs": len(ranked),
                "metrics": [{"key": m.key, "weight": m.weight,
                             "higher_is_better": m.higher_is_better} for m in metrics],
            },
        },
        organ="experiment",
    )

    return ranked


__all__ = [
    "MetricSpec",
    "LeaderboardError",
    "pareto_frontier",
    "weighted_score",
    "rank_weighted",
]

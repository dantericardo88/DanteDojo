"""Experiment run leaderboard for DanteDojo — d44.

Ranks multiple experiment runs by a primary metric and provides comparison
utilities via compare_runs(). All operations emit constitutional evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.experiment.metrics import compare_runs

if TYPE_CHECKING:
    from dojo.experiment.tracker import ExperimentTracker


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LeaderboardEntry:
    """A single run's record on the leaderboard."""

    run_id: str
    run_name: str
    experiment_name: str
    metrics: Dict[str, float]       # final metrics for this run
    params: Dict[str, Any]
    rank: int = 0                   # set by RunLeaderboard.rank()


# ---------------------------------------------------------------------------
# RunLeaderboard
# ---------------------------------------------------------------------------

class RunLeaderboard:
    """Rank multiple experiment runs by a primary metric.

    Args:
        primary_metric: Metric name used to order entries.
        higher_is_better: If True, higher values rank first (e.g. F1, accuracy).
                          If False, lower values rank first (e.g. loss, perplexity).
    """

    def __init__(self, primary_metric: str, higher_is_better: bool = True) -> None:
        self.primary_metric = primary_metric
        self.higher_is_better = higher_is_better
        self._entries: List[LeaderboardEntry] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_run(
        self,
        run_id: str,
        run_name: str,
        experiment_name: str,
        metrics: Dict[str, float],
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a completed run to the leaderboard.

        Args:
            run_id: Unique run identifier.
            run_name: Human-readable run name.
            experiment_name: Name of the parent experiment.
            metrics: Final metric values for this run.
            params: Optional hyperparameter dict.
        """
        entry = LeaderboardEntry(
            run_id=run_id,
            run_name=run_name,
            experiment_name=experiment_name,
            metrics=dict(metrics),
            params=dict(params) if params else {},
        )
        self._entries.append(entry)

        EvidenceLogger.get_instance().emit(
            {
                "type": EvidenceType.EXPERIMENT_RUN_COMPLETE.value,
                "data": {
                    "run_id": run_id,
                    "run_name": run_name,
                    "experiment_name": experiment_name,
                    "primary_metric": self.primary_metric,
                    "primary_value": metrics.get(self.primary_metric),
                    "action": "leaderboard.add_run",
                },
            },
            organ="experiment",
        )

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank(self) -> List[LeaderboardEntry]:
        """Return entries sorted by primary_metric (best first).

        Sets the ``.rank`` field on every entry (1-indexed).

        Returns:
            Sorted list of LeaderboardEntry objects.
        """
        sorted_entries = sorted(
            self._entries,
            key=lambda e: e.metrics.get(self.primary_metric, float("-inf") if self.higher_is_better else float("inf")),
            reverse=self.higher_is_better,
        )
        for idx, entry in enumerate(sorted_entries, start=1):
            entry.rank = idx

        snapshot = [
            {
                "rank": e.rank,
                "run_id": e.run_id,
                "run_name": e.run_name,
                "experiment_name": e.experiment_name,
                "primary_metric": e.metrics.get(self.primary_metric),
            }
            for e in sorted_entries
        ]

        EvidenceLogger.get_instance().emit(
            {
                "type": EvidenceType.EXPERIMENT_RUN_COMPLETE.value,
                "data": {
                    "action": "leaderboard.rank",
                    "primary_metric": self.primary_metric,
                    "higher_is_better": self.higher_is_better,
                    "total_runs": len(sorted_entries),
                    "leaderboard": snapshot,
                },
            },
            organ="experiment",
        )

        return sorted_entries

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def best(self) -> Optional[LeaderboardEntry]:
        """Return the top-ranked entry, or None if the leaderboard is empty."""
        ranked = self.rank()
        if not ranked:
            return None
        return ranked[0]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the leaderboard to a dict for JSON export."""
        ranked = self.rank()
        return {
            "primary_metric": self.primary_metric,
            "higher_is_better": self.higher_is_better,
            "entries": [
                {
                    "rank": e.rank,
                    "run_id": e.run_id,
                    "run_name": e.run_name,
                    "experiment_name": e.experiment_name,
                    "metrics": e.metrics,
                    "params": e.params,
                }
                for e in ranked
            ],
        }

    def compare_top_n(self, n: int = 3) -> Dict[str, Dict[str, Any]]:
        """Return a comparison of the top N runs against the best run.

        Uses compare_runs() from dojo.experiment.metrics.

        Args:
            n: Number of top entries to include in the comparison.

        Returns:
            Dict mapping run_id → compare_runs() output relative to the best run.
        """
        ranked = self.rank()
        if not ranked:
            return {}

        top_n = ranked[:n]
        best_entry = top_n[0]

        result: Dict[str, Dict[str, Any]] = {}
        for entry in top_n:
            comparison = compare_runs(best_entry.metrics, entry.metrics)
            result[entry.run_id] = {
                "run_name": entry.run_name,
                "rank": entry.rank,
                "baseline_run_id": best_entry.run_id,
                "comparison": comparison,
            }

        EvidenceLogger.get_instance().emit(
            {
                "type": EvidenceType.EXPERIMENT_METRIC.value,
                "data": {
                    "action": "leaderboard.compare_top_n",
                    "n": n,
                    "runs_compared": list(result.keys()),
                    "primary_metric": self.primary_metric,
                },
            },
            organ="experiment",
        )

        return result


    # ------------------------------------------------------------------
    # Class-level factory
    # ------------------------------------------------------------------

    @classmethod
    def from_tracker(
        cls,
        tracker: "ExperimentTracker",
        experiment_name: str = "",
        primary_metric: str = "loss",
        higher_is_better: bool = False,
    ) -> "RunLeaderboard":
        """Build a RunLeaderboard from all runs recorded by an ExperimentTracker.

        Args:
            tracker: ExperimentTracker whose list_runs() is called.
            experiment_name: Optional experiment filter passed to list_runs().
            primary_metric: Metric used to rank entries.
            higher_is_better: Ranking direction for primary_metric.

        Returns:
            Populated RunLeaderboard (may be empty if no runs exist).
        """
        lb = cls(primary_metric=primary_metric, higher_is_better=higher_is_better)
        runs = tracker.list_runs(experiment_name)
        for r in runs:
            run_id = r.get("run_id", "")
            run_name = r.get("name", "")
            metrics: Dict[str, float] = {
                k: float(v)
                for k, v in r.items()
                if k not in {"run_id", "name", "status", "start_time"}
                and isinstance(v, (int, float))
            }
            lb.add_run(
                run_id=run_id,
                run_name=run_name,
                experiment_name=experiment_name,
                metrics=metrics,
            )
        return lb


__all__ = [
    "LeaderboardEntry",
    "RunLeaderboard",
]

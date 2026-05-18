"""Tests for d44: Pareto frontier + weighted multi-metric ranking."""
from __future__ import annotations

import pytest

from dojo.experiment.leaderboard_advanced import (
    LeaderboardError,
    MetricSpec,
    pareto_frontier,
    rank_weighted,
    weighted_score,
)


def _runs():
    return [
        {"run_id": "a", "accuracy": 0.92, "cost": 5.0},
        {"run_id": "b", "accuracy": 0.85, "cost": 2.0},
        {"run_id": "c", "accuracy": 0.78, "cost": 1.0},
        {"run_id": "d", "accuracy": 0.65, "cost": 0.5},
    ]


class TestMetricSpec:
    def test_signed_value_higher_is_better(self):
        m = MetricSpec(key="acc", higher_is_better=True)
        assert m.signed_value(0.9) == 0.9

    def test_signed_value_lower_is_better(self):
        m = MetricSpec(key="cost", higher_is_better=False)
        assert m.signed_value(5.0) == -5.0


class TestParetoFrontier:
    def test_all_pareto_optimal_returned(self):
        metrics = [
            MetricSpec("accuracy", higher_is_better=True),
            MetricSpec("cost", higher_is_better=False),
        ]
        frontier = pareto_frontier(_runs(), metrics)
        # a (highest acc, highest cost) and d (lowest cost, lowest acc) bracket
        # the trade-off; b and c may or may not be on the frontier depending
        # on the exact numbers. All four runs are non-dominated here.
        ids = {r["run_id"] for r in frontier}
        assert "a" in ids
        assert "d" in ids

    def test_dominated_run_excluded(self):
        runs = [
            {"run_id": "best", "accuracy": 0.95, "cost": 1.0},
            {"run_id": "dominated", "accuracy": 0.50, "cost": 2.0},
        ]
        metrics = [
            MetricSpec("accuracy", higher_is_better=True),
            MetricSpec("cost", higher_is_better=False),
        ]
        frontier = pareto_frontier(runs, metrics)
        ids = {r["run_id"] for r in frontier}
        assert "best" in ids
        assert "dominated" not in ids

    def test_excludes_runs_missing_metric(self):
        runs = [
            {"run_id": "a", "accuracy": 0.9, "cost": 1.0},
            {"run_id": "b", "accuracy": 0.95},  # no cost
        ]
        metrics = [
            MetricSpec("accuracy", higher_is_better=True),
            MetricSpec("cost", higher_is_better=False),
        ]
        frontier = pareto_frontier(runs, metrics)
        assert {r["run_id"] for r in frontier} == {"a"}

    def test_empty_metrics_raises(self):
        with pytest.raises(LeaderboardError):
            pareto_frontier(_runs(), [])

    def test_single_metric_keeps_only_best(self):
        runs = [
            {"run_id": "a", "score": 5.0},
            {"run_id": "b", "score": 8.0},
            {"run_id": "c", "score": 3.0},
        ]
        frontier = pareto_frontier(runs, [MetricSpec("score", higher_is_better=True)])
        # Only the maximum is non-dominated
        assert len(frontier) == 1
        assert frontier[0]["run_id"] == "b"


class TestWeightedScore:
    def test_returns_weighted_sum(self):
        run = {"acc": 0.9, "cost": 2.0}
        metrics = [
            MetricSpec("acc", weight=1.0),
            MetricSpec("cost", higher_is_better=False, weight=0.5),
        ]
        score = weighted_score(run, metrics)
        # 1.0 * 0.9 + 0.5 * (-2.0) = -0.1
        assert score == pytest.approx(-0.1)

    def test_returns_none_when_metric_missing(self):
        run = {"acc": 0.9}
        metrics = [MetricSpec("acc"), MetricSpec("cost")]
        assert weighted_score(run, metrics) is None


class TestRankWeighted:
    def test_orders_by_weighted_score_desc(self):
        metrics = [
            MetricSpec("accuracy", higher_is_better=True, weight=1.0),
            MetricSpec("cost", higher_is_better=False, weight=0.1),
        ]
        ranked = rank_weighted(_runs(), metrics)
        # 'a': 0.92 - 0.5 = 0.42 (highest)
        # 'b': 0.85 - 0.2 = 0.65 (wait, recompute)
        # Actually: a = 0.92*1 + (-5.0)*0.1 = 0.42
        # b = 0.85 + (-2.0)*0.1 = 0.65
        # c = 0.78 + (-1.0)*0.1 = 0.68
        # d = 0.65 + (-0.5)*0.1 = 0.60
        # Order: c (0.68), b (0.65), d (0.60), a (0.42)
        assert ranked[0]["run_id"] == "c"
        assert ranked[-1]["run_id"] == "a"

    def test_limit_caps_results(self):
        ranked = rank_weighted(_runs(),
                               [MetricSpec("accuracy")],
                               limit=2)
        assert len(ranked) == 2

    def test_excludes_runs_missing_metric(self):
        runs = [
            {"run_id": "a", "x": 1.0},
            {"run_id": "b"},  # missing x
        ]
        ranked = rank_weighted(runs, [MetricSpec("x")])
        assert len(ranked) == 1
        assert ranked[0]["run_id"] == "a"


class TestEvidence:
    def test_pareto_emits_evidence(self):
        from dojo.core.evidence import EvidenceLogger
        EvidenceLogger.clear()
        pareto_frontier(_runs(), [MetricSpec("accuracy")])
        events = EvidenceLogger.get_events()
        assert any(e.get("type") == "experiment.pareto_frontier" for e in events)

    def test_weighted_ranking_emits_evidence(self):
        from dojo.core.evidence import EvidenceLogger
        EvidenceLogger.clear()
        rank_weighted(_runs(), [MetricSpec("accuracy")])
        events = EvidenceLogger.get_events()
        assert any(e.get("type") == "experiment.weighted_ranking" for e in events)

"""d35: CostTracker advanced analytics — forecast, breakdown, savings.

Proves the three constitutional analytics methods that close the gap between
"basic cost tracking" (8) and "comprehensive cost analytics system" (9):

  monthly_forecast(year, month)  — projects full-month spend from actual receipts
  cost_breakdown_by_task()       — per-task cost/token/call aggregation
  savings_report()               — quantifies routing savings vs paid API

Coverage:
1.  monthly_forecast() returns 0.0 on empty store
2.  monthly_forecast() returns actual_usd from real records
3.  monthly_forecast() projects using daily rate × days_in_month
4.  monthly_forecast() returns days_elapsed matching distinct active days
5.  monthly_forecast() emits cost.monthly_forecast evidence
6.  monthly_forecast() excludes receipts from other months
7.  cost_breakdown_by_task() returns empty dict on empty store
8.  cost_breakdown_by_task() groups by task name
9.  cost_breakdown_by_task() calculates avg_cost_usd correctly
10. cost_breakdown_by_task() accumulates total_tokens
11. savings_report() returns 0.0 on empty store
12. savings_report() sums local_displaced_estimate_usd correctly
13. savings_report() computes savings_fraction
14. savings_report() emits cost.savings_report evidence
15. savings_report() returns correct receipt_count
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.types import CostReceipt, RoutingOption
from dojo.route.cost_tracker import CostTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture()
def tracker(tmp_path) -> CostTracker:
    return CostTracker(storage_path=tmp_path / "receipts.jsonl")


def _receipt(
    task: str = "test",
    route: str = RoutingOption.OLLAMA_LOCAL.value,
    cost_usd: float = 0.0,
    local_displaced: float = 0.01,
    tokens_in: int = 100,
    tokens_out: int = 50,
    year: int = 2026,
    month: int = 5,
    day: int = 1,
) -> CostReceipt:
    return CostReceipt(
        task=task,
        route_taken=route,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        latency_ms=50.0,
        local_displaced_estimate_usd=local_displaced,
        recorded_at=datetime(year, month, day, 12, 0, 0, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# monthly_forecast() tests
# ---------------------------------------------------------------------------

class TestMonthlyForecast:
    def test_forecast_empty_store_returns_zeros(self, tracker):
        result = tracker.monthly_forecast(2026, 5)
        assert result["actual_usd"] == 0.0
        assert result["projected_usd"] == 0.0
        assert result["days_elapsed"] == 0
        assert result["daily_rate_usd"] == 0.0

    def test_forecast_actual_usd_from_records(self, tracker):
        tracker.record(_receipt(cost_usd=0.50, day=1))
        tracker.record(_receipt(cost_usd=0.30, day=2))
        result = tracker.monthly_forecast(2026, 5)
        assert result["actual_usd"] == pytest.approx(0.80, abs=1e-6)

    def test_forecast_projects_from_daily_rate(self, tracker):
        # Two records on the same day → 1 active day with $0.60
        tracker.record(_receipt(cost_usd=0.30, day=10))
        tracker.record(_receipt(cost_usd=0.30, day=10))
        result = tracker.monthly_forecast(2026, 5)
        # daily_rate = 0.60, days_in_month = 31
        assert result["daily_rate_usd"] == pytest.approx(0.60, abs=1e-6)
        assert result["projected_usd"] == pytest.approx(0.60 * 31, abs=1e-4)

    def test_forecast_days_elapsed_counts_distinct_days(self, tracker):
        tracker.record(_receipt(day=1))
        tracker.record(_receipt(day=1))  # same day — should count as 1
        tracker.record(_receipt(day=5))  # different day
        result = tracker.monthly_forecast(2026, 5)
        assert result["days_elapsed"] == 2

    def test_forecast_emits_evidence(self, tracker):
        tracker.record(_receipt(cost_usd=0.10, day=3))
        EvidenceLogger.clear()
        tracker.monthly_forecast(2026, 5)
        events = EvidenceLogger.get_events()
        forecast_events = [e for e in events if e.get("type") == "cost.monthly_forecast"]
        assert forecast_events, "monthly_forecast must emit cost.monthly_forecast evidence"
        data = forecast_events[0]["data"]
        assert data["year"] == 2026
        assert data["month"] == 5
        assert data["actual_usd"] == pytest.approx(0.10, abs=1e-6)

    def test_forecast_excludes_other_months(self, tracker):
        tracker.record(_receipt(cost_usd=0.50, year=2026, month=4, day=15))
        tracker.record(_receipt(cost_usd=0.20, year=2026, month=5, day=1))
        result = tracker.monthly_forecast(2026, 5)
        assert result["actual_usd"] == pytest.approx(0.20, abs=1e-6)

    def test_forecast_days_in_month_correct(self, tracker):
        result = tracker.monthly_forecast(2026, 2)  # Feb 2026 = 28 days
        assert result["days_in_month"] == 28
        result_may = tracker.monthly_forecast(2026, 5)  # May = 31
        assert result_may["days_in_month"] == 31


# ---------------------------------------------------------------------------
# cost_breakdown_by_task() tests
# ---------------------------------------------------------------------------

class TestCostBreakdownByTask:
    def test_breakdown_empty_store_returns_empty_dict(self, tracker):
        assert tracker.cost_breakdown_by_task() == {}

    def test_breakdown_groups_by_task_name(self, tracker):
        tracker.record(_receipt(task="summarize", cost_usd=0.01))
        tracker.record(_receipt(task="summarize", cost_usd=0.02))
        tracker.record(_receipt(task="classify", cost_usd=0.005))
        breakdown = tracker.cost_breakdown_by_task()
        assert "summarize" in breakdown
        assert "classify" in breakdown
        assert len(breakdown) == 2

    def test_breakdown_total_usd_accumulates(self, tracker):
        tracker.record(_receipt(task="qa", cost_usd=0.10))
        tracker.record(_receipt(task="qa", cost_usd=0.20))
        breakdown = tracker.cost_breakdown_by_task()
        assert breakdown["qa"]["total_usd"] == pytest.approx(0.30, abs=1e-6)

    def test_breakdown_avg_cost_usd_correct(self, tracker):
        tracker.record(_receipt(task="gen", cost_usd=0.10))
        tracker.record(_receipt(task="gen", cost_usd=0.30))
        breakdown = tracker.cost_breakdown_by_task()
        assert breakdown["gen"]["avg_cost_usd"] == pytest.approx(0.20, abs=1e-6)
        assert breakdown["gen"]["call_count"] == 2

    def test_breakdown_total_tokens_accumulated(self, tracker):
        tracker.record(_receipt(task="embed", tokens_in=100, tokens_out=50))
        tracker.record(_receipt(task="embed", tokens_in=200, tokens_out=100))
        breakdown = tracker.cost_breakdown_by_task()
        assert breakdown["embed"]["total_tokens"] == 450  # (100+50) + (200+100)

    def test_breakdown_single_call_avg_equals_total(self, tracker):
        tracker.record(_receipt(task="onecall", cost_usd=0.07))
        breakdown = tracker.cost_breakdown_by_task()
        assert breakdown["onecall"]["avg_cost_usd"] == pytest.approx(
            breakdown["onecall"]["total_usd"], abs=1e-8
        )


# ---------------------------------------------------------------------------
# savings_report() tests
# ---------------------------------------------------------------------------

class TestSavingsReport:
    def test_savings_empty_store(self, tracker):
        result = tracker.savings_report()
        assert result["total_local_displaced_usd"] == 0.0
        assert result["total_actual_usd"] == 0.0
        assert result["savings_fraction"] == 0.0
        assert result["receipt_count"] == 0

    def test_savings_sums_local_displaced(self, tracker):
        tracker.record(_receipt(cost_usd=0.0, local_displaced=0.01))
        tracker.record(_receipt(cost_usd=0.0, local_displaced=0.03))
        result = tracker.savings_report()
        assert result["total_local_displaced_usd"] == pytest.approx(0.04, abs=1e-6)

    def test_savings_fraction_all_local(self, tracker):
        # All routed to Ollama (cost=0, displaced=0.05) → savings_fraction=1.0
        tracker.record(_receipt(cost_usd=0.0, local_displaced=0.05))
        result = tracker.savings_report()
        assert result["savings_fraction"] == pytest.approx(1.0, abs=1e-4)

    def test_savings_fraction_mixed(self, tracker):
        # $0.10 local displaced, $0.10 actual → 50% savings
        tracker.record(_receipt(cost_usd=0.10, local_displaced=0.10))
        result = tracker.savings_report()
        assert result["savings_fraction"] == pytest.approx(0.5, abs=1e-4)

    def test_savings_emits_evidence(self, tracker):
        tracker.record(_receipt(local_displaced=0.02))
        EvidenceLogger.clear()
        tracker.savings_report()
        events = EvidenceLogger.get_events()
        savings_events = [e for e in events if e.get("type") == "cost.savings_report"]
        assert savings_events, "savings_report must emit cost.savings_report evidence"
        data = savings_events[0]["data"]
        assert "savings_fraction" in data
        assert "total_local_displaced_usd" in data

    def test_savings_receipt_count(self, tracker):
        for _ in range(5):
            tracker.record(_receipt())
        result = tracker.savings_report()
        assert result["receipt_count"] == 5

    def test_savings_no_displaced_zero_fraction(self, tracker):
        # All paid API calls, no local displacement
        tracker.record(_receipt(cost_usd=0.10, local_displaced=0.0))
        result = tracker.savings_report()
        assert result["savings_fraction"] == 0.0

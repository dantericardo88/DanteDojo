"""Cost tracker for DanteDojo routing layer.

Persists CostReceipt records to a JSONL file and provides monthly summaries.
Emits ROUTE_COST_RECEIPT evidence on every record() call.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import BudgetExceeded
from dojo.core.types import CostReceipt, RoutingOption


class BudgetStatus(str, Enum):
    """Return value from check_budget() indicating the budget state.

    - OK: usage < 80% of the monthly limit
    - SOFT_WARN: 80% <= usage < 100% of the monthly limit
    - EXCEEDED: usage >= 100% (BudgetExceeded is also raised)
    """
    OK = "ok"
    SOFT_WARN = "soft_warn"
    EXCEEDED = "exceeded"


# Soft-warn threshold fraction (80%)
_SOFT_WARN_FRACTION: float = 0.80

_DEFAULT_STORAGE = Path.home() / ".dante" / "cost_receipts.jsonl"

# Cost per 1k tokens by provider key (route_taken value)
_API_COST_PER_1K: dict[str, float] = {
    RoutingOption.API_HAIKU.value: 0.00025,
    RoutingOption.API_SONNET.value: 0.003,
    RoutingOption.API_DEEPSEEK.value: 0.00014,
    RoutingOption.API_GPT5_MINI.value: 0.00015,
}


class CostTracker:
    """Append-only JSONL cost receipt store with monthly aggregation."""

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        monthly_limit_usd: Optional[float] = None,
    ) -> None:
        self._storage_path = storage_path or _DEFAULT_STORAGE
        self.monthly_limit_usd = monthly_limit_usd
        # Track whether a soft-warn has already been emitted this month to
        # avoid duplicate evidence for repeated calls at the same spend level.
        self._soft_warn_emitted: bool = False

    def reset_monthly(self) -> None:
        """Reset soft-warn state (call when starting a new billing month)."""
        self._soft_warn_emitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, receipt: CostReceipt) -> None:
        """Persist a cost receipt and emit ROUTE_COST_RECEIPT evidence.

        The receipt is appended to the JSONL file atomically (per line).

        Args:
            receipt: Populated CostReceipt to persist.
        """
        logger = EvidenceLogger.get_instance()

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self._storage_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(receipt.to_dict(), separators=(",", ":")) + "\n")

        logger.emit(
            {
                "type": EvidenceType.ROUTE_COST_RECEIPT.value,
                "data": receipt.to_dict(),
            },
            organ="route",
        )

    def record_from_response(
        self,
        receipt: CostReceipt,
        response_tokens: int,
        prompt_tokens: int,
    ) -> CostReceipt:
        """Compute actual cost from real token counts and persist the updated receipt.

        Creates a new CostReceipt with the actual prompt/response token counts
        and the corresponding cost, then persists it and emits evidence.

        Args:
            receipt: Original CostReceipt (used for task, route, latency, etc.).
            response_tokens: Actual output token count from the API response.
            prompt_tokens: Actual input token count from the API response.

        Returns:
            A new CostReceipt updated with real token counts and cost.
        """
        logger = EvidenceLogger.get_instance()
        total_tokens = prompt_tokens + response_tokens
        cost_per_1k = _API_COST_PER_1K.get(receipt.route_taken, 0.0)
        actual_cost = round((total_tokens / 1000.0) * cost_per_1k, 8)

        updated = CostReceipt(
            task=receipt.task,
            route_taken=receipt.route_taken,
            tokens_in=prompt_tokens,
            tokens_out=response_tokens,
            cost_usd=actual_cost,
            latency_ms=receipt.latency_ms,
            local_displaced_estimate_usd=receipt.local_displaced_estimate_usd,
            recorded_at=receipt.recorded_at,
        )

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self._storage_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(updated.to_dict(), separators=(",", ":")) + "\n")

        logger.emit(
            {
                "type": EvidenceType.ROUTE_COST_RECEIPT.value,
                "data": {
                    **updated.to_dict(),
                    "actual_tokens": True,
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "total_tokens": total_tokens,
                },
            },
            organ="route",
        )

        return updated

    def get_provider_stats(self) -> dict[str, dict]:
        """Return per-provider aggregated stats across all stored receipts.

        Returns:
            dict mapping route_taken value → dict with:
              - total_usd: sum of cost_usd
              - total_tokens: sum of tokens_in + tokens_out
              - call_count: number of receipts
        """
        receipts = self.load_receipts()
        stats: dict[str, dict] = {}

        for r in receipts:
            key = r.route_taken
            if key not in stats:
                stats[key] = {"total_usd": 0.0, "total_tokens": 0, "call_count": 0}
            stats[key]["total_usd"] = round(stats[key]["total_usd"] + r.cost_usd, 8)
            stats[key]["total_tokens"] += r.tokens_in + r.tokens_out
            stats[key]["call_count"] += 1

        return stats

    def monthly_summary(self, year: int, month: int) -> dict:
        """Return aggregated cost data for the given calendar month.

        Args:
            year: Calendar year (e.g. 2026).
            month: Calendar month 1–12.

        Returns:
            dict with keys:
              - total_cost_usd: sum of cost_usd for the month
              - local_displaced_usd: sum of local_displaced_estimate_usd
              - receipts: count of receipts in the month
        """
        receipts = self.load_receipts()
        total_cost = 0.0
        local_displaced = 0.0
        count = 0

        for r in receipts:
            recorded_at = r.recorded_at
            if recorded_at.year == year and recorded_at.month == month:
                total_cost += r.cost_usd
                local_displaced += r.local_displaced_estimate_usd
                count += 1

        return {
            "total_cost_usd": round(total_cost, 6),
            "local_displaced_usd": round(local_displaced, 6),
            "receipts": count,
        }

    def check_budget(self, estimated_cost_usd: float) -> BudgetStatus:
        """Check whether adding estimated_cost_usd would exceed the monthly limit.

        Emits ROUTE_COST_RECEIPT evidence with budget status regardless of outcome.
        Emits ``cost.budget_soft_warn`` evidence the first time spend crosses 80%.
        Raises BudgetExceeded if the projected total would exceed monthly_limit_usd.

        Args:
            estimated_cost_usd: Estimated cost in USD for the pending request.

        Returns:
            BudgetStatus.OK if projected < 80% of limit.
            BudgetStatus.SOFT_WARN if 80% <= projected < 100% of limit.

        Raises:
            BudgetExceeded: If current spend + estimated_cost_usd > monthly_limit_usd.
        """
        from datetime import date

        now = date.today()
        summary = self.monthly_summary(now.year, now.month)
        projected = summary["total_cost_usd"] + estimated_cost_usd

        logger = EvidenceLogger.get_instance()

        # Determine status
        if self.monthly_limit_usd and projected > self.monthly_limit_usd:
            status = BudgetStatus.EXCEEDED
        elif (
            self.monthly_limit_usd
            and projected >= self.monthly_limit_usd * _SOFT_WARN_FRACTION
        ):
            status = BudgetStatus.SOFT_WARN
        else:
            status = BudgetStatus.OK

        # Use legacy-compatible strings in the evidence payload so existing
        # consumers checking "within_budget" / "over_budget" continue to work.
        _legacy_status = {
            BudgetStatus.OK: "within_budget",
            BudgetStatus.SOFT_WARN: "soft_warn",
            BudgetStatus.EXCEEDED: "over_budget",
        }[status]
        logger.emit(
            {
                "type": EvidenceType.ROUTE_COST_RECEIPT.value,
                "data": {
                    "budget_check": True,
                    "monthly_limit_usd": self.monthly_limit_usd,
                    "current_spend_usd": summary["total_cost_usd"],
                    "estimated_new_cost_usd": estimated_cost_usd,
                    "projected_total_usd": projected,
                    "status": _legacy_status,
                },
            },
            organ="route",
        )

        # Emit soft-warn evidence on first crossing
        if status == BudgetStatus.SOFT_WARN and not self._soft_warn_emitted:
            self._soft_warn_emitted = True
            logger.emit(
                {
                    "type": "cost.budget_soft_warn",
                    "data": {
                        "monthly_limit_usd": self.monthly_limit_usd,
                        "projected_total_usd": projected,
                        "usage_fraction": projected / self.monthly_limit_usd,
                        "threshold_fraction": _SOFT_WARN_FRACTION,
                    },
                },
                organ="route",
            )

        if status == BudgetStatus.EXCEEDED:
            raise BudgetExceeded(
                f"Request would exceed monthly budget: ${projected:.4f} > ${self.monthly_limit_usd:.4f}",
                reason=(
                    f"current=${summary['total_cost_usd']:.4f}, "
                    f"estimated=${estimated_cost_usd:.6f}, "
                    f"limit=${self.monthly_limit_usd:.4f}"
                ),
            )

        return status

    def monthly_forecast(self, year: int, month: int) -> dict:
        """Project full-month cost based on actual spend and days elapsed.

        Uses the daily spend rate from actual JSONL records to extrapolate the
        remaining days in the month, giving an evidence-backed budget forecast.

        Args:
            year: Calendar year (e.g. 2026).
            month: Calendar month 1–12.

        Returns:
            dict with:
              - actual_usd: spend so far this month
              - projected_usd: extrapolated full-month total
              - days_elapsed: calendar days with at least one receipt
              - days_in_month: total calendar days in the month
              - daily_rate_usd: actual_usd / days_elapsed (0.0 if no data)
        """
        import calendar
        from datetime import date

        receipts = self.load_receipts()
        days_in_month = calendar.monthrange(year, month)[1]

        month_receipts = [
            r for r in receipts
            if r.recorded_at.year == year and r.recorded_at.month == month
        ]

        actual_usd = round(sum(r.cost_usd for r in month_receipts), 6)

        if not month_receipts:
            return {
                "actual_usd": 0.0,
                "projected_usd": 0.0,
                "days_elapsed": 0,
                "days_in_month": days_in_month,
                "daily_rate_usd": 0.0,
            }

        active_days = len({r.recorded_at.date() for r in month_receipts})
        daily_rate = round(actual_usd / active_days, 8) if active_days > 0 else 0.0
        projected_usd = round(daily_rate * days_in_month, 6)

        logger = EvidenceLogger.get_instance()
        logger.emit(
            {
                "type": "cost.monthly_forecast",
                "data": {
                    "year": year,
                    "month": month,
                    "actual_usd": actual_usd,
                    "projected_usd": projected_usd,
                    "daily_rate_usd": daily_rate,
                    "days_elapsed": active_days,
                    "days_in_month": days_in_month,
                },
            },
            organ="route",
        )

        return {
            "actual_usd": actual_usd,
            "projected_usd": projected_usd,
            "days_elapsed": active_days,
            "days_in_month": days_in_month,
            "daily_rate_usd": daily_rate,
        }

    def cost_breakdown_by_task(self) -> dict[str, dict]:
        """Return per-task cost aggregation across all stored receipts.

        Groups all receipts by their ``task`` field and returns cost/token/call
        statistics per task — useful for identifying the most expensive workloads.

        Returns:
            dict mapping task name → dict with:
              - total_usd: cumulative cost for this task
              - total_tokens: total input + output tokens
              - call_count: number of receipts
              - avg_cost_usd: mean cost per call
        """
        receipts = self.load_receipts()
        breakdown: dict[str, dict] = {}

        for r in receipts:
            key = r.task
            if key not in breakdown:
                breakdown[key] = {
                    "total_usd": 0.0,
                    "total_tokens": 0,
                    "call_count": 0,
                    "avg_cost_usd": 0.0,
                }
            breakdown[key]["total_usd"] = round(breakdown[key]["total_usd"] + r.cost_usd, 8)
            breakdown[key]["total_tokens"] += r.tokens_in + r.tokens_out
            breakdown[key]["call_count"] += 1

        for task_data in breakdown.values():
            n = task_data["call_count"]
            task_data["avg_cost_usd"] = round(task_data["total_usd"] / n, 8) if n > 0 else 0.0

        return breakdown

    def savings_report(self) -> dict:
        """Compute total cost savings from routing to local inference vs paid APIs.

        Sums the ``local_displaced_estimate_usd`` field across all receipts to
        show how much routing to Ollama/local saved compared to what the same
        workload would have cost on a paid API provider.

        Returns:
            dict with:
              - total_local_displaced_usd: total estimated savings (sum of all
                ``local_displaced_estimate_usd`` values across all receipts)
              - total_actual_usd: total real spend (sum of cost_usd)
              - savings_fraction: local_displaced / (local_displaced + actual)
                — fraction of potential spend that was avoided
              - receipt_count: total number of receipts
        """
        receipts = self.load_receipts()
        total_displaced = sum(r.local_displaced_estimate_usd for r in receipts)
        total_actual = sum(r.cost_usd for r in receipts)
        total_potential = total_displaced + total_actual
        savings_fraction = (
            round(total_displaced / total_potential, 4)
            if total_potential > 0
            else 0.0
        )

        logger = EvidenceLogger.get_instance()
        logger.emit(
            {
                "type": "cost.savings_report",
                "data": {
                    "total_local_displaced_usd": round(total_displaced, 6),
                    "total_actual_usd": round(total_actual, 6),
                    "savings_fraction": savings_fraction,
                    "receipt_count": len(receipts),
                },
            },
            organ="route",
        )

        return {
            "total_local_displaced_usd": round(total_displaced, 6),
            "total_actual_usd": round(total_actual, 6),
            "savings_fraction": savings_fraction,
            "receipt_count": len(receipts),
        }

    def load_receipts(self) -> list[CostReceipt]:
        """Load all receipts from JSONL storage.

        Returns:
            List of CostReceipt objects, oldest first.
        """
        if not self._storage_path.exists():
            return []

        receipts: list[CostReceipt] = []
        with self._storage_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    receipts.append(_dict_to_receipt(data))
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

        return receipts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dict_to_receipt(data: dict) -> CostReceipt:
    """Reconstruct a CostReceipt from a dict (loaded from JSONL)."""
    recorded_at_raw = data.get("recorded_at", "")
    if recorded_at_raw:
        recorded_at = datetime.fromisoformat(recorded_at_raw)
        if recorded_at.tzinfo is None:
            recorded_at = recorded_at.replace(tzinfo=timezone.utc)
    else:
        recorded_at = datetime.now(timezone.utc)

    return CostReceipt(
        task=data["task"],
        route_taken=data["route_taken"],
        tokens_in=int(data["tokens_in"]),
        tokens_out=int(data["tokens_out"]),
        cost_usd=float(data["cost_usd"]),
        latency_ms=float(data["latency_ms"]),
        local_displaced_estimate_usd=float(data["local_displaced_estimate_usd"]),
        recorded_at=recorded_at,
    )


__all__ = ["BudgetStatus", "CostTracker"]

"""Cost tracker for DanteDojo routing layer.

Persists CostReceipt records to a JSONL file and provides monthly summaries.
Emits ROUTE_COST_RECEIPT evidence on every record() call.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import BudgetExceeded
from dojo.core.types import CostReceipt

_DEFAULT_STORAGE = Path.home() / ".dante" / "cost_receipts.jsonl"


class CostTracker:
    """Append-only JSONL cost receipt store with monthly aggregation."""

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        monthly_limit_usd: Optional[float] = None,
    ) -> None:
        self._storage_path = storage_path or _DEFAULT_STORAGE
        self.monthly_limit_usd = monthly_limit_usd

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

    def check_budget(self, estimated_cost_usd: float) -> None:
        """Check whether adding estimated_cost_usd would exceed the monthly limit.

        Emits ROUTE_COST_RECEIPT evidence with budget status regardless of outcome.
        Raises BudgetExceeded if the projected total would exceed monthly_limit_usd.

        Args:
            estimated_cost_usd: Estimated cost in USD for the pending request.

        Raises:
            BudgetExceeded: If current spend + estimated_cost_usd > monthly_limit_usd.
        """
        from datetime import date

        now = date.today()
        summary = self.monthly_summary(now.year, now.month)
        projected = summary["total_cost_usd"] + estimated_cost_usd

        logger = EvidenceLogger.get_instance()
        logger.emit(
            {
                "type": EvidenceType.ROUTE_COST_RECEIPT.value,
                "data": {
                    "budget_check": True,
                    "monthly_limit_usd": self.monthly_limit_usd,
                    "current_spend_usd": summary["total_cost_usd"],
                    "estimated_new_cost_usd": estimated_cost_usd,
                    "projected_total_usd": projected,
                    "status": "within_budget" if projected <= self.monthly_limit_usd else "over_budget",
                },
            },
            organ="route",
        )

        if projected > self.monthly_limit_usd:
            raise BudgetExceeded(
                f"Request would exceed monthly budget: ${projected:.4f} > ${self.monthly_limit_usd:.4f}",
                reason=(
                    f"current=${summary['total_cost_usd']:.4f}, "
                    f"estimated=${estimated_cost_usd:.6f}, "
                    f"limit=${self.monthly_limit_usd:.4f}"
                ),
            )

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


__all__ = ["CostTracker"]

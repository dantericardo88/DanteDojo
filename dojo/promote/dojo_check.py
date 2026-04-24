"""DanteDojo promotion gate — Dojo evaluation check.

Verifies that a candidate model beats the baseline by the required margin
and that the 95% CI lower bound clears the baseline (when provided).

Fail-closed: any unexpected error → BLOCKED.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType

if TYPE_CHECKING:
    from dojo.promote.gate import GateCheckResult


def _make_receipt(gate: str, passed: bool, reasons: list[str]) -> bytes:
    payload = json.dumps(
        {"gate": gate, "passed": passed, "reasons": reasons, "ts": time.time()},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).digest()


def check_dojo(
    candidate_f1: float,
    baseline_f1: float,
    eval_receipt_merkle: str,
    min_improvement_pts: float = 5.0,
    ci_lower_bound: Optional[float] = None,
) -> "GateCheckResult":
    """Check Dojo eval results.

    Checks:
    - candidate_f1 >= baseline_f1 + min_improvement_pts (in percentage points)
    - 95% CI lower bound >= baseline_f1 (if ci_lower_bound provided)

    Returns GateCheckResult. Emits PROMOTE_DOJO_GREEN or PROMOTE_DOJO_BLOCKED.
    """
    from dojo.promote.gate import GateCheckResult  # noqa: PLC0415

    logger = EvidenceLogger.get_instance()
    reasons: list[str] = []
    gate = "dojo_check"

    try:
        # Convert F1 to percentage points (0.76 → 76.0 pts, or treat as-is if already 0-1 scale).
        # The spec says "5.0 points" — interpret as 0.05 on the 0-1 scale or 5.0 on 0-100 scale.
        # Looking at usage: baseline_f1=0.70, candidate_f1=0.80 → 10 pts improvement (> 5 → pass).
        # baseline_f1=0.70, candidate_f1=0.76 → 6 pts improvement (> 5 → pass).
        # baseline_f1=0.70, candidate_f1=0.74 → 4 pts improvement (< 5 → blocked).
        # So 1 pt = 0.01 on the 0-1 scale.  min_improvement_pts=5.0 → threshold = 0.05.

        improvement = candidate_f1 - baseline_f1
        threshold = min_improvement_pts / 100.0  # 5.0 pts → 0.05

        if improvement < threshold:
            reasons.append(
                f"candidate_f1 {candidate_f1:.4f} does not beat baseline_f1 {baseline_f1:.4f} "
                f"by the required {min_improvement_pts:.1f} percentage points "
                f"(improvement={improvement * 100:.2f}pt)"
            )

        # 95% CI lower bound check
        if ci_lower_bound is not None and ci_lower_bound < baseline_f1:
            reasons.append(
                f"95% CI lower bound {ci_lower_bound:.4f} is below baseline_f1 {baseline_f1:.4f}"
            )

    except Exception as exc:
        reasons.append(f"Unexpected error in dojo_check: {exc}")

    passed = len(reasons) == 0
    receipt = _make_receipt(gate, passed, reasons)

    result = GateCheckResult(gate=gate, passed=passed, reasons=reasons, receipt=receipt)

    ev_type = EvidenceType.PROMOTE_DOJO_GREEN if passed else EvidenceType.PROMOTE_DOJO_BLOCKED
    logger.emit(
        {
            "type": ev_type.value,
            "data": {
                "gate": gate,
                "passed": passed,
                "candidate_f1": candidate_f1,
                "baseline_f1": baseline_f1,
                "improvement_pts": round((candidate_f1 - baseline_f1) * 100, 4),
                "min_improvement_pts": min_improvement_pts,
                "ci_lower_bound": ci_lower_bound,
                "eval_receipt_merkle": eval_receipt_merkle,
                "reasons": reasons,
            },
        },
        organ="promote",
    )

    return result

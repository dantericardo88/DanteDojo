"""Post-quantization quality validator for DanteDojo.

Enforces perplexity and F1 regression thresholds after quantization.
Emits QUANT_QUALITY_GATE evidence on every validation run.
"""

from __future__ import annotations

from typing import Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import QuantQualityRejected


class QuantValidator:
    """Validate quantized model quality against baseline metrics.

    Raises QuantQualityRejected if either threshold is breached.
    """

    PERPLEXITY_MAX_DELTA_PCT = 15.0   # reject if perplexity regresses > 15 %
    TASK_F1_MAX_DELTA_PTS = 5.0       # reject if F1 regresses > 5 points

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        base_perplexity: float,
        quant_perplexity: float,
        base_f1: Optional[float] = None,
        quant_f1: Optional[float] = None,
    ) -> tuple[bool, str]:
        """Run quality gate.

        Returns:
            (passed, reason) tuple.

        Raises:
            QuantQualityRejected: when any threshold is breached.
        """
        logger = EvidenceLogger.get_instance()

        perplexity_delta_pct = _pct_regression(base_perplexity, quant_perplexity)
        f1_delta_pts: Optional[float] = None
        if base_f1 is not None and quant_f1 is not None:
            f1_delta_pts = base_f1 - quant_f1

        passed = True
        reason = "quality gate passed"

        if perplexity_delta_pct > self.PERPLEXITY_MAX_DELTA_PCT:
            passed = False
            reason = (
                f"perplexity regression {perplexity_delta_pct:.2f}% "
                f"exceeds max {self.PERPLEXITY_MAX_DELTA_PCT}%"
            )

        if (
            passed
            and f1_delta_pts is not None
            and f1_delta_pts > self.TASK_F1_MAX_DELTA_PTS
        ):
            passed = False
            reason = (
                f"F1 regression {f1_delta_pts:.2f} pts "
                f"exceeds max {self.TASK_F1_MAX_DELTA_PTS} pts"
            )

        evidence_payload: dict = {
            "type": EvidenceType.QUANT_QUALITY_GATE.value,
            "data": {
                "passed": passed,
                "reason": reason,
                "base_perplexity": base_perplexity,
                "quant_perplexity": quant_perplexity,
                "perplexity_delta_pct": round(perplexity_delta_pct, 4),
                "base_f1": base_f1,
                "quant_f1": quant_f1,
                "f1_delta_pts": round(f1_delta_pts, 4) if f1_delta_pts is not None else None,
                "perplexity_threshold_pct": self.PERPLEXITY_MAX_DELTA_PCT,
                "f1_threshold_pts": self.TASK_F1_MAX_DELTA_PTS,
            },
        }
        logger.emit(evidence_payload, organ="quant")

        if not passed:
            raise QuantQualityRejected(reason, reason=reason)

        return passed, reason


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pct_regression(base: float, quant: float) -> float:
    """Return percentage increase of quant over base (positive = regression)."""
    if base == 0.0:
        return 0.0
    return max(0.0, (quant - base) / base * 100.0)


__all__ = ["QuantValidator"]

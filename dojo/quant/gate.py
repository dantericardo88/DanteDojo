"""dojo/quant/gate.py — standalone quality gate for post-quantization checks.

Provides check_quality_gate(), a thin wrapper around QuantValidator that:
  - accepts explicit metric values or stub defaults when none are supplied
  - emits quant.gate_check evidence (organ="quant")
  - returns a result dict on pass
  - raises QuantQualityRejected on fail

This module is the backing implementation for `dojo quant gate-check`.
"""

from __future__ import annotations

from typing import Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import QuantQualityRejected

# Stub metric defaults used when no real metrics are provided (CI / offline mode)
_STUB_METRICS: dict[str, tuple[float, float]] = {
    "gguf":  (25.0, 26.0),   # (ppl_before, ppl_after) — ~4% regression
    "gptq":  (25.0, 26.2),   # ~4.8% regression
    "awq":   (25.0, 25.8),   # ~3.2% regression
}


def check_quality_gate(
    quant_type: str = "gguf",
    threshold: float = 0.10,
    ppl_before: Optional[float] = None,
    ppl_after: Optional[float] = None,
) -> dict:
    """Run the post-quantization quality gate.

    Computes fractional perplexity regression ``(ppl_after - ppl_before) /
    ppl_before`` and rejects if it exceeds *threshold*.

    Uses stub metric defaults when *ppl_before* / *ppl_after* are not supplied,
    allowing the gate to be exercised in CI without a real model.

    Args:
        quant_type: One of ``"gguf"``, ``"gptq"``, or ``"awq"``.
        threshold: Maximum allowed fractional degradation (0.10 = 10%).
        ppl_before: Baseline perplexity. ``None`` → use stub default.
        ppl_after:  Post-quant perplexity. ``None`` → use stub default.

    Returns:
        A dict with keys ``passed``, ``quant_type``, ``threshold``,
        ``ppl_before``, ``ppl_after``, and ``degradation``.

    Raises:
        QuantQualityRejected: when degradation exceeds *threshold*.
    """
    logger = EvidenceLogger.get_instance()

    # Resolve metric values
    stub_before, stub_after = _STUB_METRICS.get(quant_type.lower(), (25.0, 26.0))
    actual_before = ppl_before if ppl_before is not None else stub_before
    actual_after  = ppl_after  if ppl_after  is not None else stub_after

    degradation = (
        (actual_after - actual_before) / actual_before
        if actual_before > 0
        else 0.0
    )

    passed = degradation <= threshold
    reason = (
        "passed"
        if passed
        else (
            f"{quant_type} quality gate rejected: degradation "
            f"{degradation * 100:.2f}% exceeds threshold "
            f"{threshold * 100:.2f}%"
        )
    )

    result = {
        "passed": passed,
        "quant_type": quant_type,
        "threshold": threshold,
        "ppl_before": actual_before,
        "ppl_after": actual_after,
        "degradation": round(degradation, 6),
        "reason": reason,
    }

    # Always emit evidence — pass or fail
    logger.emit(
        {
            "type": EvidenceType.QUANT_QUALITY_GATE.value,
            "data": result,
        },
        organ="quant",
    )

    if not passed:
        raise QuantQualityRejected(reason, reason=reason)

    return result


__all__ = ["check_quality_gate"]

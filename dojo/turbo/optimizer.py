"""dojo/turbo/optimizer.py — Pareto-optimal config selection.

Given a list of ProfileResults and a PPL budget (max regression fraction),
selects the config that maximizes throughput while keeping PPL within budget.
"""

from __future__ import annotations

from dojo.turbo.config import ProfileResult


def select_optimal(
    results: list[ProfileResult],
    ppl_budget: float = 0.15,
) -> ProfileResult:
    """Return the fastest config whose PPL regression stays within *ppl_budget*.

    Args:
        results:    All benchmark results, including the BF16/FP16 baseline.
        ppl_budget: Maximum allowed PPL regression as a fraction (0.15 = 15%).

    Returns:
        The Pareto-optimal ProfileResult.  Falls back to the baseline (lowest
        PPL, highest accuracy) if every quantized config exceeds the budget.
    """
    if not results:
        raise ValueError("No benchmark results to select from")

    successful = [r for r in results if r.success and r.tokens_per_sec > 0]
    if not successful:
        raise ValueError("All benchmark configs failed — cannot select optimal")

    # Baseline = unquantized run (lowest PPL)
    baseline = min(successful, key=lambda r: r.ppl)
    baseline_ppl = baseline.ppl

    def within_budget(r: ProfileResult) -> bool:
        if baseline_ppl <= 0:
            return True
        delta = (r.ppl - baseline_ppl) / baseline_ppl
        return delta <= ppl_budget

    candidates = [r for r in successful if within_budget(r)]
    if not candidates:
        # Nothing fits the budget — return baseline itself
        return baseline

    # Pareto-optimal: highest throughput among budget-compliant configs
    return max(candidates, key=lambda r: r.tokens_per_sec)


def build_pareto_table(
    results: list[ProfileResult],
    baseline_ppl: float,
) -> list[dict]:
    """Return a sorted display table for all results."""
    rows = []
    for r in results:
        delta = (r.ppl - baseline_ppl) / baseline_ppl if baseline_ppl > 0 else 0.0
        rows.append({
            "precision": r.precision,
            "quant": r.quant_scheme,
            "tps": r.tokens_per_sec,
            "ppl": r.ppl,
            "ppl_delta_pct": round(delta * 100, 2),
            "vram_gb": r.vram_gb,
            "ok": r.success,
        })
    return sorted(rows, key=lambda x: x["tps"], reverse=True)

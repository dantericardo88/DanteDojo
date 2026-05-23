"""dojo/research/interaction_model.py — Interaction effect taxonomy for combination tests."""

from __future__ import annotations

from enum import Enum


class InteractionEffect(str, Enum):
    """Classification of how two techniques interact when combined."""

    ADDITIVE    = "additive"     # A+B ≈ A_gain + B_gain
    SYNERGISTIC = "synergistic"  # A+B > A_gain + B_gain (super-additive)
    DESTRUCTIVE = "destructive"  # A+B < min(A_gain, B_gain) (sub-additive)
    ORTHOGONAL  = "orthogonal"   # A+B ≈ A_gain (B has no effect in presence of A)
    UNKNOWN     = "unknown"      # insufficient data


def classify_interaction(
    baseline: float,
    a_only: float,
    b_only: float,
    combined: float,
    tolerance: float = 0.05,
) -> InteractionEffect:
    """Classify the interaction effect from four metric measurements.

    All values should be on the same scale where HIGHER is BETTER
    (e.g. tokens/sec). For PPL where lower is better, pass (1/ppl) values.

    Args:
        baseline: Metric with neither A nor B applied.
        a_only:   Metric with only technique A applied.
        b_only:   Metric with only technique B applied.
        combined: Metric with both A and B applied.
        tolerance: Fraction tolerance for additive classification (5%).

    Returns:
        InteractionEffect enum value.
    """
    if baseline <= 0:
        return InteractionEffect.UNKNOWN

    gain_a = (a_only - baseline) / baseline
    gain_b = (b_only - baseline) / baseline
    gain_combined = (combined - baseline) / baseline
    expected_additive = gain_a + gain_b

    # Synergistic: combined > expected additive + tolerance
    if gain_combined > expected_additive + tolerance:
        return InteractionEffect.SYNERGISTIC

    # Destructive: combined < min individual gain - tolerance
    if gain_combined < min(gain_a, gain_b) - tolerance:
        return InteractionEffect.DESTRUCTIVE

    # Orthogonal: combined ≈ larger individual gain (B has no effect)
    larger_gain = max(gain_a, gain_b)
    if abs(gain_combined - larger_gain) < tolerance and abs(gain_b - 0) < tolerance * 2:
        return InteractionEffect.ORTHOGONAL

    # Additive: within tolerance of expected sum
    if abs(gain_combined - expected_additive) <= tolerance:
        return InteractionEffect.ADDITIVE

    # Default: additive (catches edge cases)
    return InteractionEffect.ADDITIVE

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


def describe_interaction(
    baseline: float,
    a_only: float,
    b_only: float,
    combined: float,
    technique_a: str,
    technique_b: str,
    metric: str = "tokens_per_sec",
    higher_is_better: bool = True,
) -> str:
    """Return a plain-English explanation of the measured interaction effect.

    Accepts raw metric values (in their natural units) regardless of direction.
    For higher_is_better=False (e.g. PPL), values are inverted internally before
    classification so the interaction math is correct; the text shows raw values.
    """
    if not higher_is_better:
        bl_n = 1 / baseline if baseline > 0 else 0
        a_n = 1 / a_only if a_only > 0 else 0
        b_n = 1 / b_only if b_only > 0 else 0
        c_n = 1 / combined if combined > 0 else 0
    else:
        bl_n, a_n, b_n, c_n = baseline, a_only, b_only, combined

    effect = classify_interaction(bl_n, a_n, b_n, c_n)
    gain_a = (a_n - bl_n) / bl_n if bl_n > 0 else 0
    gain_b = (b_n - bl_n) / bl_n if bl_n > 0 else 0
    gain_c = (c_n - bl_n) / bl_n if bl_n > 0 else 0
    expected = gain_a + gain_b
    synergy = gain_c - expected

    a_pct = f"{gain_a:+.1%}"
    b_pct = f"{gain_b:+.1%}"
    c_pct = f"{gain_c:+.1%}"

    lines = [
        f"  Baseline:          {baseline:.1f}",
        f"  {technique_a} only:  {a_only:.1f}  ({a_pct} vs baseline)",
        f"  {technique_b} only:  {b_only:.1f}  ({b_pct} vs baseline)",
        f"  Combined:          {combined:.1f}  ({c_pct} vs baseline)",
        f"  Expected additive: {baseline * (1 + expected):.1f}  ({expected:+.1%})",
        f"  Synergy delta:     {synergy:+.1%}",
        "",
    ]

    if effect == InteractionEffect.SYNERGISTIC:
        lines.append(
            f"  SYNERGISTIC: the combined effect ({c_pct}) exceeds the simple sum of "
            f"{technique_a} ({a_pct}) + {technique_b} ({b_pct}) = {expected:+.1%}. "
            f"One technique partially cancels the overhead of the other."
        )
        if gain_b < 0:
            lines.append(
                f"  Note: {technique_b} alone degrades {metric} ({b_pct}) on this model size — "
                f"dequantization overhead exceeds memory-bandwidth savings for <7B models. "
                f"But {technique_a} partially cancels that overhead, so the combined loss "
                f"({c_pct}) is less than the additive prediction ({expected:+.1%})."
            )
    elif effect == InteractionEffect.ADDITIVE:
        lines.append(
            f"  ADDITIVE: combined effect ({c_pct}) ≈ {technique_a} ({a_pct}) + "
            f"{technique_b} ({b_pct}) = {expected:+.1%}. No amplification or interference."
        )
    elif effect == InteractionEffect.DESTRUCTIVE:
        lines.append(
            f"  DESTRUCTIVE: combined ({c_pct}) is worse than either technique alone "
            f"({technique_a}: {a_pct}, {technique_b}: {b_pct}). "
            f"The techniques interfere with each other."
        )
    elif effect == InteractionEffect.ORTHOGONAL:
        lines.append(
            f"  ORTHOGONAL: {technique_b} adds no benefit when {technique_a} is already applied. "
            f"Combined ({c_pct}) ≈ {technique_a} alone ({a_pct})."
        )
    else:
        lines.append(f"  UNKNOWN: insufficient data to classify interaction.")

    return "\n".join(lines)

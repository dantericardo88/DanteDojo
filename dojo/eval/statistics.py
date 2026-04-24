"""Bootstrap confidence intervals and seed stability for DanteDojo eval.

All functions are pure (no side-effects) and deterministic given the same seed.
"""

from __future__ import annotations

import math
import random
from typing import List


def bootstrap_ci(
    samples: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, lower_bound, upper_bound) using percentile bootstrap CI.

    Deterministic given seed. Requires len(samples) >= 2.

    Args:
        samples: Raw score values from evaluation runs.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Coverage probability (e.g. 0.95 for 95% CI).
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of (mean, lower_ci, upper_ci).

    Raises:
        ValueError: If len(samples) < 2.
    """
    if len(samples) < 2:
        raise ValueError(f"bootstrap_ci requires at least 2 samples, got {len(samples)}")

    rng = random.Random(seed)
    n = len(samples)
    mean_val = sum(samples) / n

    boot_means: List[float] = []
    for _ in range(n_bootstrap):
        resample = [rng.choice(samples) for _ in range(n)]
        boot_means.append(sum(resample) / n)

    boot_means.sort()
    alpha = 1.0 - confidence
    lower_idx = int(math.floor(alpha / 2 * n_bootstrap))
    upper_idx = int(math.ceil((1.0 - alpha / 2) * n_bootstrap)) - 1
    lower_idx = max(0, min(lower_idx, n_bootstrap - 1))
    upper_idx = max(0, min(upper_idx, n_bootstrap - 1))

    return mean_val, boot_means[lower_idx], boot_means[upper_idx]


def seed_variance(scores: list[float]) -> float:
    """Return the standard deviation across seed runs.

    Uses population std (ddof=0) for consistency with mathematical std.

    Args:
        scores: Scores from multiple seed runs.

    Returns:
        Standard deviation as a float (0.0 for empty or single-element lists).
    """
    if len(scores) < 2:
        return 0.0
    n = len(scores)
    mean = sum(scores) / n
    variance = sum((x - mean) ** 2 for x in scores) / n
    return math.sqrt(variance)


def seeds_are_stable(scores: list[float], max_std: float = 2.0) -> bool:
    """Return True if std(scores) <= max_std. Unstable models get blocked.

    Args:
        scores: Scores from multiple seed runs.
        max_std: Maximum allowed standard deviation (inclusive).

    Returns:
        True if the seed runs are stable enough to proceed.
    """
    return seed_variance(scores) <= max_std


__all__ = [
    "bootstrap_ci",
    "seed_variance",
    "seeds_are_stable",
]

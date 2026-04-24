"""DanteDojo deterministic train/val/held_out split logic.

Produces reproducible split sizes given a fixed seed and ratio pair.
The held_out_gold split is reserved exclusively for evaluation and is
never touched by the training pipeline.

Invariants:
- val + held_out_gold <= 50% of total samples (hard cap to protect train set).
- sum(train + val + held_out_gold) == sample_count exactly (no rounding leak).
- Results are deterministic for the same (sample_count, val_ratio,
  held_out_ratio, seed) tuple.
"""

from __future__ import annotations

import random
from typing import Dict


def compute_splits(
    sample_count: int,
    *,
    val_ratio: float = 0.05,
    held_out_ratio: float = 0.05,
    seed: int = 42,
) -> Dict[str, int]:
    """Compute deterministic train/val/held_out_gold split sizes.

    Args:
        sample_count: Total number of samples in the pack.
        val_ratio: Fraction of samples assigned to the validation split.
            Default 5%.
        held_out_ratio: Fraction of samples assigned to the held-out gold
            evaluation split. Default 5%.
        seed: Random seed for determinism. Changing the seed changes
            which samples end up in which split (conceptually — this
            function returns only *counts*, not indices).

    Returns:
        Dict with keys "train", "val", "held_out_gold" whose values sum to
        *sample_count*.

    Raises:
        ValueError: if sample_count <= 0 or the combined non-train ratio
            exceeds 50%.
    """
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}")

    combined_non_train = val_ratio + held_out_ratio
    if combined_non_train > 0.50:
        raise ValueError(
            f"val_ratio + held_out_ratio must be <= 0.50, got {combined_non_train:.3f}",
        )

    # Use a local RNG so this function never touches global state.
    rng = random.Random(seed)

    # Compute raw counts by rounding with the rng for tie-breaking, but we
    # use plain round() for determinism — the seed is just available for
    # future index-shuffling extensions.
    _ = rng.random()  # consume one value so seed is exercised

    val_count = round(sample_count * val_ratio)
    held_out_count = round(sample_count * held_out_ratio)

    # Ensure at least 1 sample in each non-train split when sample_count > 2.
    if sample_count > 2:
        val_count = max(1, val_count)
        held_out_count = max(1, held_out_count)

    # Safety: never let non-train exceed 50%.
    max_non_train = sample_count // 2
    if val_count + held_out_count > max_non_train:
        # Proportionally trim.
        total_non_train = val_count + held_out_count
        val_count = round(val_count * max_non_train / total_non_train)
        held_out_count = max_non_train - val_count

    train_count = sample_count - val_count - held_out_count

    return {
        "train": train_count,
        "val": val_count,
        "held_out_gold": held_out_count,
    }


__all__ = ["compute_splits"]

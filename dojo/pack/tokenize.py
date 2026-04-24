"""DanteDojo pack tokenization utilities.

Provides fast, tokenizer-free helpers for token budgeting and contamination
hashing. Actual tokenization runs in the training pipeline, not at import time.
"""

from __future__ import annotations

import hashlib
import re


def normalize_text(text: str) -> str:
    """Strip whitespace, collapse multiple spaces, and lowercase *text*.

    Used as the canonical form for contamination hash computation so that
    minor formatting differences (trailing newlines, double spaces) do not
    escape detection.
    """
    collapsed = re.sub(r"\s+", " ", text)
    return collapsed.strip().lower()


def compute_sample_hash(text: str) -> str:
    """Return the SHA-256 hex digest of the normalized form of *text*.

    Used for contamination checks between training packs and eval gold sets.
    The hash is deterministic: identical text always yields the same digest.
    """
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def estimate_token_count(text: str, chars_per_token: float = 3.5) -> int:
    """Fast token count estimator for budgeting (no tokenizer required).

    Uses a simple character-to-token ratio.  This is accurate enough for
    resource planning and pack sizing; the training pipeline uses the real
    tokenizer for all precision-sensitive operations.

    Args:
        text: The text to estimate.
        chars_per_token: Average characters per token.  3.5 is a reasonable
            heuristic for English prose with a BPE tokenizer.

    Returns:
        Estimated token count (always >= 1 for non-empty input, 0 for empty).
    """
    if not text:
        return 0
    return max(1, round(len(text) / chars_per_token))


__all__ = [
    "compute_sample_hash",
    "estimate_token_count",
    "normalize_text",
]

"""dojo/research/combination_composer.py — Combination test engine.

Given two independent evidence streams (technique A and technique B),
generates a test plan measuring whether combining them is additive,
synergistic, destructive, or orthogonal.

The composer works entirely on CPU — it reads existing evidence, proposes
a test plan, and optionally executes it on GPU when DOJO_REAL_GPU=1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from dojo.core.evidence import EvidenceLogger
from dojo.research.interaction_model import InteractionEffect, classify_interaction, describe_interaction


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CombinationPair:
    """A pair of techniques to test together."""

    technique_a: str
    technique_b: str
    model: str
    metric: str         # "tokens_per_sec" | "ppl" | etc.
    higher_is_better: bool = True
    description: str = ""


@dataclass
class CombinationResult:
    """Outcome of running a combination test."""

    pair: CombinationPair
    baseline: float
    a_only: float
    b_only: float
    combined: float
    interaction: InteractionEffect
    synergy_score: float = 0.0  # (combined - expected_additive) / baseline
    elapsed_s: float = 0.0
    gpu: str = ""
    error: str = ""
    success: bool = True
    description: str = ""  # plain-English explanation filled by run()


# ---------------------------------------------------------------------------
# Pre-defined high-value combination pairs
# ---------------------------------------------------------------------------

COMBINATION_CANDIDATES: list[CombinationPair] = [
    CombinationPair(
        technique_a="gptq",
        technique_b="flash_attention",
        model="qwen/Qwen2-1.5B",
        metric="tokens_per_sec",
        higher_is_better=True,
        description="GPTQ 4-bit (d22) + Flash Attention (d15): combined memory + speed",
    ),
    CombinationPair(
        technique_a="lora",
        technique_b="gptq",
        model="qwen/Qwen2-1.5B",
        metric="ppl",
        higher_is_better=False,
        description="QLoRA fine-tune (d17) → GPTQ quantize (d22): does fine-tuning survive compression?",
    ),
    CombinationPair(
        technique_a="fp8",
        technique_b="gptq",
        model="qwen/Qwen2-1.5B",
        metric="tokens_per_sec",
        higher_is_better=True,
        description="FP8 training (d10) + GPTQ inference (d22): train FP8, serve GPTQ",
    ),
]


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------

class CombinationComposer:
    """Compose and run combination tests from evidence pairs."""

    def __init__(self, emit_evidence: bool = True):
        self._logger = EvidenceLogger.get_instance() if emit_evidence else None

    def _emit(self, event_type: str, data: dict) -> None:
        if self._logger:
            self._logger.emit({"type": event_type, "data": data}, organ="research")

    def propose(self, pair: CombinationPair) -> dict:
        """Emit a COMBINATION_TEST_PROPOSED event and return the test plan."""
        plan = {
            "technique_a": pair.technique_a,
            "technique_b": pair.technique_b,
            "model": pair.model,
            "metric": pair.metric,
            "steps": [
                f"1. Measure {pair.metric} baseline (no techniques)",
                f"2. Measure {pair.metric} with {pair.technique_a} only",
                f"3. Measure {pair.metric} with {pair.technique_b} only",
                f"4. Measure {pair.metric} with both {pair.technique_a} + {pair.technique_b}",
                "5. Classify interaction effect",
            ],
        }
        self._emit("COMBINATION_TEST_PROPOSED", plan)
        return plan

    def run(
        self,
        pair: CombinationPair,
        baseline_fn,
        a_fn,
        b_fn,
        combined_fn,
    ) -> CombinationResult:
        """Execute a combination test using provided measurement functions.

        Args:
            pair:        The CombinationPair describing the test.
            baseline_fn: Callable() → float  (no techniques applied)
            a_fn:        Callable() → float  (technique A only)
            b_fn:        Callable() → float  (technique B only)
            combined_fn: Callable() → float  (both A and B)

        Returns:
            CombinationResult with interaction classification.
        """
        self.propose(pair)
        t0 = time.monotonic()

        try:
            baseline = baseline_fn()
            self._emit("COMBINATION_BASELINE_A", {
                "label": "baseline", "value": baseline,
                "technique_a": pair.technique_a, "technique_b": pair.technique_b,
            })

            a_val = a_fn()
            self._emit("COMBINATION_BASELINE_A", {
                "label": pair.technique_a, "value": a_val,
                "technique_a": pair.technique_a,
            })

            b_val = b_fn()
            self._emit("COMBINATION_BASELINE_B", {
                "label": pair.technique_b, "value": b_val,
                "technique_b": pair.technique_b,
            })

            combined = combined_fn()
            self._emit("COMBINATION_JOINT_RESULT", {
                "label": f"{pair.technique_a}+{pair.technique_b}",
                "value": combined,
            })

            # Normalize for classifier: higher = better
            if pair.higher_is_better:
                bl, a_n, b_n, c_n = baseline, a_val, b_val, combined
            else:
                bl = 1 / baseline if baseline > 0 else 0
                a_n = 1 / a_val if a_val > 0 else 0
                b_n = 1 / b_val if b_val > 0 else 0
                c_n = 1 / combined if combined > 0 else 0

            effect = classify_interaction(bl, a_n, b_n, c_n)
            gain_a = (a_n - bl) / bl if bl > 0 else 0
            gain_b = (b_n - bl) / bl if bl > 0 else 0
            gain_combined = (c_n - bl) / bl if bl > 0 else 0
            synergy = gain_combined - (gain_a + gain_b)

            elapsed = time.monotonic() - t0
            # Pass raw metric values so description shows human-readable numbers.
            # describe_interaction normalizes internally for higher_is_better=False.
            description = describe_interaction(
                baseline, a_val, b_val, combined,
                pair.technique_a, pair.technique_b,
                pair.metric, pair.higher_is_better,
            )
            result = CombinationResult(
                pair=pair,
                baseline=baseline,
                a_only=a_val,
                b_only=b_val,
                combined=combined,
                interaction=effect,
                synergy_score=round(synergy, 4),
                elapsed_s=round(elapsed, 1),
                success=True,
                description=description,
            )

            self._emit("COMBINATION_INTERACTION_CLASSIFIED", {
                "technique_a": pair.technique_a,
                "technique_b": pair.technique_b,
                "model": pair.model,
                "metric": pair.metric,
                "baseline": baseline,
                "a_only": a_val,
                "b_only": b_val,
                "combined": combined,
                "interaction": effect.value,
                "synergy_score": result.synergy_score,
                "elapsed_s": elapsed,
                "description": description,
            })

            return result

        except Exception as e:
            elapsed = time.monotonic() - t0
            return CombinationResult(
                pair=pair,
                baseline=0.0, a_only=0.0, b_only=0.0, combined=0.0,
                interaction=InteractionEffect.UNKNOWN,
                elapsed_s=round(elapsed, 1),
                error=str(e),
                success=False,
            )

    def find_pairs(
        self,
        new_technique: str,
        existing_techniques: list[str],
    ) -> list[CombinationPair]:
        """Return combination candidates involving *new_technique*."""
        candidates = []
        for pair in COMBINATION_CANDIDATES:
            if (pair.technique_a == new_technique and pair.technique_b in existing_techniques) or \
               (pair.technique_b == new_technique and pair.technique_a in existing_techniques):
                candidates.append(pair)
        return candidates

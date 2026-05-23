"""Unit tests for dojo/research/combination_composer.py and interaction_model.py."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_evidence():
    from dojo.core.evidence import EvidenceLogger
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


class TestInteractionClassifier:
    """Test the classify_interaction() function directly."""

    def test_synergistic(self):
        from dojo.research.interaction_model import InteractionEffect, classify_interaction
        # baseline=100, A=120 (+20%), B=115 (+15%), combined=145 (+45% > 35% expected)
        effect = classify_interaction(100, 120, 115, 145)
        assert effect == InteractionEffect.SYNERGISTIC

    def test_additive(self):
        from dojo.research.interaction_model import InteractionEffect, classify_interaction
        # baseline=100, A=110 (+10%), B=115 (+15%), combined=125 (+25% ≈ 10+15)
        effect = classify_interaction(100, 110, 115, 125)
        assert effect == InteractionEffect.ADDITIVE

    def test_destructive(self):
        from dojo.research.interaction_model import InteractionEffect, classify_interaction
        # baseline=100, A=120 (+20%), B=115 (+15%), combined=105 (+5% < min(20,15))
        effect = classify_interaction(100, 120, 115, 105)
        assert effect == InteractionEffect.DESTRUCTIVE

    def test_zero_baseline_returns_unknown(self):
        from dojo.research.interaction_model import InteractionEffect, classify_interaction
        effect = classify_interaction(0, 10, 15, 20)
        assert effect == InteractionEffect.UNKNOWN


class TestCombinationComposer:
    """Test the CombinationComposer orchestrator."""

    def _make_pair(self):
        from dojo.research.combination_composer import CombinationPair
        return CombinationPair(
            technique_a="gptq",
            technique_b="flash_attention",
            model="qwen/Qwen2-1.5B",
            metric="tokens_per_sec",
            higher_is_better=True,
            description="test pair",
        )

    def test_propose_emits_evidence(self):
        from dojo.core.evidence import EvidenceLogger
        from dojo.research.combination_composer import CombinationComposer

        composer = CombinationComposer(emit_evidence=True)
        pair = self._make_pair()
        plan = composer.propose(pair)

        assert "steps" in plan
        events = EvidenceLogger.get_events()
        event_types = [e.get("type", "") for e in events]
        assert "COMBINATION_TEST_PROPOSED" in event_types

    def test_run_synergistic(self):
        from dojo.research.combination_composer import CombinationComposer
        from dojo.research.interaction_model import InteractionEffect

        composer = CombinationComposer(emit_evidence=True)
        pair = self._make_pair()

        result = composer.run(
            pair,
            baseline_fn=lambda: 1000.0,
            a_fn=lambda: 1200.0,       # GPTQ: +20%
            b_fn=lambda: 1150.0,       # FlashAttn: +15%
            combined_fn=lambda: 1450.0, # Combined: +45% > 35% expected
        )

        assert result.success
        assert result.interaction == InteractionEffect.SYNERGISTIC
        assert result.synergy_score > 0

    def test_run_destructive(self):
        from dojo.research.combination_composer import CombinationComposer
        from dojo.research.interaction_model import InteractionEffect

        composer = CombinationComposer(emit_evidence=True)
        pair = self._make_pair()

        result = composer.run(
            pair,
            baseline_fn=lambda: 1000.0,
            a_fn=lambda: 1200.0,       # GPTQ: +20%
            b_fn=lambda: 1150.0,       # FlashAttn: +15%
            combined_fn=lambda: 1050.0, # Only +5% — destructive
        )

        assert result.success
        assert result.interaction == InteractionEffect.DESTRUCTIVE

    def test_run_emits_all_evidence_types(self):
        from dojo.core.evidence import EvidenceLogger
        from dojo.research.combination_composer import CombinationComposer

        composer = CombinationComposer(emit_evidence=True)
        pair = self._make_pair()

        composer.run(
            pair,
            baseline_fn=lambda: 1000.0,
            a_fn=lambda: 1100.0,
            b_fn=lambda: 1050.0,
            combined_fn=lambda: 1160.0,
        )

        events = EvidenceLogger.get_events()
        event_types = {e.get("type", "") for e in events}
        assert "COMBINATION_TEST_PROPOSED" in event_types
        assert "COMBINATION_JOINT_RESULT" in event_types
        assert "COMBINATION_INTERACTION_CLASSIFIED" in event_types

    def test_run_handles_exception_gracefully(self):
        from dojo.research.combination_composer import CombinationComposer
        from dojo.research.interaction_model import InteractionEffect

        composer = CombinationComposer(emit_evidence=False)
        pair = self._make_pair()

        def _fail():
            raise RuntimeError("GPU exploded")

        result = composer.run(
            pair,
            baseline_fn=_fail,
            a_fn=lambda: 0.0,
            b_fn=lambda: 0.0,
            combined_fn=lambda: 0.0,
        )

        assert not result.success
        assert result.interaction == InteractionEffect.UNKNOWN
        assert "GPU exploded" in result.error

    def test_lower_is_better_ppl(self):
        """PPL (lower is better) interaction is classified correctly when inverted."""
        from dojo.research.combination_composer import CombinationComposer, CombinationPair
        from dojo.research.interaction_model import InteractionEffect

        pair = CombinationPair(
            technique_a="gptq",
            technique_b="lora",
            model="qwen/Qwen2-1.5B",
            metric="ppl",
            higher_is_better=False,
        )
        composer = CombinationComposer(emit_evidence=False)
        result = composer.run(
            pair,
            baseline_fn=lambda: 10.0,  # baseline PPL
            a_fn=lambda: 12.0,         # GPTQ raises PPL (worse)
            b_fn=lambda: 9.0,          # LoRA improves PPL
            combined_fn=lambda: 11.0,  # combined
        )
        assert result.success
        assert result.interaction != InteractionEffect.UNKNOWN

    def test_find_pairs_returns_relevant(self):
        from dojo.research.combination_composer import CombinationComposer

        composer = CombinationComposer(emit_evidence=False)
        pairs = composer.find_pairs(
            new_technique="gptq",
            existing_techniques=["flash_attention", "lora", "grpo"],
        )
        assert len(pairs) > 0
        assert all(
            p.technique_a == "gptq" or p.technique_b == "gptq"
            for p in pairs
        )

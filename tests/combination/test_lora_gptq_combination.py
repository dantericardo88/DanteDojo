"""tests/combination/test_lora_gptq_combination.py — Combination test: LoRA → INT4.

GPU INTEGRATION TEST — requires DOJO_REAL_GPU=1.

Tests the interaction effect between:
  A: LoRA fine-tuning (peft — d17)
  B: bitsandbytes NF4 INT4 quantization (d25)

Scientific question: "Does fine-tuning survive quantization?"

The CombinationComposer measures all 4 configurations using PPL (lower=better):
  baseline  = base model PPL
  A only    = LoRA fine-tuned model PPL
  B only    = base model + INT4 quantization PPL
  combined  = LoRA fine-tuned → merged → INT4 quantization PPL

If SYNERGISTIC (or ADDITIVE): LoRA fine-tuning survives INT4 compression.
If DESTRUCTIVE: quantization destroys most of the fine-tuning gain.

Evidence emitted: COMBINATION_TEST_PROPOSED, COMBINATION_INTERACTION_CLASSIFIED.
"""

from __future__ import annotations

import os

import pytest

_REAL_GPU = os.environ.get("DOJO_REAL_GPU", "0") == "1"
pytestmark = pytest.mark.skipif(
    not _REAL_GPU,
    reason="LoRA+INT4 combination test requires DOJO_REAL_GPU=1",
)

_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(autouse=True)
def _clear_evidence():
    from dojo.core.evidence import EvidenceLogger
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


class TestLoRAInt4Combination:
    def test_lora_survives_quantization(self):
        """Run all 4 PPL measurements and classify the interaction effect.

        Practical question: if we LoRA fine-tune then INT4 quantize,
        does the fine-tuning gain survive compression?
        """
        from dojo.core.evidence import EvidenceLogger
        from dojo.research.combination_composer import CombinationComposer, CombinationPair
        from dojo.research.gpu_measurements import make_ppl_measurement_fns
        from dojo.research.interaction_model import InteractionEffect

        baseline_fn, lora_fn, int4_fn, combined_fn = make_ppl_measurement_fns(
            _MODEL,
            finetune_steps=3,
            lora_r=4,
        )

        pair = CombinationPair(
            technique_a="lora",
            technique_b="int4_bnb",
            model=_MODEL,
            metric="ppl",
            higher_is_better=False,
            description="LoRA fine-tuning (d17) → bitsandbytes NF4 INT4 (d25): "
                        "does fine-tuning survive quantization?",
        )

        composer = CombinationComposer(emit_evidence=True)
        result = composer.run(pair, baseline_fn, lora_fn, int4_fn, combined_fn)

        assert result.success, f"Combination test failed: {result.error}"
        assert result.baseline > 0, f"Baseline PPL must be positive, got {result.baseline}"
        assert result.a_only > 0, f"LoRA PPL must be positive, got {result.a_only}"
        assert result.b_only > 0, f"INT4 PPL must be positive, got {result.b_only}"
        assert result.combined > 0, f"Combined PPL must be positive, got {result.combined}"
        assert result.interaction != InteractionEffect.UNKNOWN, (
            "Interaction must be classified (not UNKNOWN)"
        )
        assert result.elapsed_s > 0
        assert result.description != "", "Result must carry plain-English description"
        assert "lora" in result.description

        # Evidence chain must be sealed
        events = EvidenceLogger.get_events()
        event_strs = [str(e) for e in events]
        assert any("COMBINATION_INTERACTION_CLASSIFIED" in s for s in event_strs), (
            "COMBINATION_INTERACTION_CLASSIFIED evidence must be emitted"
        )

        # Print results
        print(f"\n=== LoRA + INT4 Combination Results (PPL — lower is better) ===")
        print(f"  Model:    {_MODEL}")
        print(f"  Baseline (base model):     {result.baseline:.2f} PPL")
        print(f"  A only   (LoRA fine-tuned): {result.a_only:.2f} PPL")
        print(f"  B only   (INT4 quantized):  {result.b_only:.2f} PPL")
        print(f"  Combined (LoRA → INT4):     {result.combined:.2f} PPL")
        print(f"  Interaction:               {result.interaction.value}")
        print(f"  Synergy score:             {result.synergy_score:+.4f}")
        print(f"  Elapsed:                   {result.elapsed_s:.1f}s")
        print(f"  Evidence events:           {len(events)}")
        print(f"\n=== Interaction Explanation ===")
        print(result.description)

    def test_all_four_ppl_measurements_are_positive(self):
        """All 4 PPL measurements must be finite and positive."""
        from dojo.research.combination_composer import CombinationComposer, CombinationPair
        from dojo.research.gpu_measurements import make_ppl_measurement_fns

        baseline_fn, lora_fn, int4_fn, combined_fn = make_ppl_measurement_fns(
            _MODEL,
            finetune_steps=2,
            lora_r=4,
        )

        pair = CombinationPair(
            technique_a="lora",
            technique_b="int4_bnb",
            model=_MODEL,
            metric="ppl",
            higher_is_better=False,
        )
        composer = CombinationComposer(emit_evidence=False)
        result = composer.run(pair, baseline_fn, lora_fn, int4_fn, combined_fn)

        assert result.success, result.error
        assert result.baseline > 0, f"Baseline must be positive: {result.baseline}"
        assert result.a_only > 0, f"LoRA PPL must be positive: {result.a_only}"
        assert result.b_only > 0, f"INT4 PPL must be positive: {result.b_only}"
        assert result.combined > 0, f"Combined PPL must be positive: {result.combined}"
        assert result.elapsed_s > 0

    def test_evidence_chain_is_sealed(self):
        """COMBINATION_INTERACTION_CLASSIFIED must appear in the evidence log."""
        from dojo.core.evidence import EvidenceLogger
        from dojo.research.combination_composer import CombinationComposer, CombinationPair
        from dojo.research.gpu_measurements import make_ppl_measurement_fns

        baseline_fn, lora_fn, int4_fn, combined_fn = make_ppl_measurement_fns(
            _MODEL,
            finetune_steps=2,
            lora_r=4,
        )

        pair = CombinationPair(
            technique_a="lora",
            technique_b="int4_bnb",
            model=_MODEL,
            metric="ppl",
            higher_is_better=False,
        )
        composer = CombinationComposer(emit_evidence=True)
        result = composer.run(pair, baseline_fn, lora_fn, int4_fn, combined_fn)

        assert result.success, result.error
        events = EvidenceLogger.get_events()
        event_strs = [str(e) for e in events]
        assert any("COMBINATION_TEST_PROPOSED" in s for s in event_strs)
        assert any("COMBINATION_INTERACTION_CLASSIFIED" in s for s in event_strs)
        # Description must be propagated into evidence
        classified = [e for e in events if e.get("type") == "COMBINATION_INTERACTION_CLASSIFIED"]
        assert classified, "COMBINATION_INTERACTION_CLASSIFIED event not found"
        assert "description" in classified[-1].get("data", {}), (
            "description must be present in evidence payload"
        )

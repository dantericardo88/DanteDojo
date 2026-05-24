"""tests/combination/test_flash_int4_combination.py — Combination test: Flash Attention + INT4.

GPU INTEGRATION TEST — requires DOJO_REAL_GPU=1.

Tests the interaction effect between:
  A: Flash Attention (PyTorch SDPA — d15)
  B: bitsandbytes NF4 INT4 quantization (d25)

The CombinationComposer measures all 4 configurations and classifies the interaction:
  baseline  = FP16 eager
  A only    = FP16 + SDPA
  B only    = INT4 NF4 + eager
  combined  = INT4 NF4 + SDPA

Expected: ADDITIVE or SYNERGISTIC — both independently increase throughput;
combined should be at least as fast as either alone.

Evidence emitted: COMBINATION_TEST_PROPOSED, COMBINATION_BASELINE_A/B,
COMBINATION_JOINT_RESULT, COMBINATION_INTERACTION_CLASSIFIED.
"""

from __future__ import annotations

import os

import pytest

_REAL_GPU = os.environ.get("DOJO_REAL_GPU", "0") == "1"
pytestmark = pytest.mark.skipif(
    not _REAL_GPU,
    reason="Flash+INT4 combination test requires DOJO_REAL_GPU=1",
)

_MODEL = "qwen/Qwen2-1.5B"


@pytest.fixture(autouse=True)
def _clear_evidence():
    from dojo.core.evidence import EvidenceLogger
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


class TestFlashInt4Combination:
    def test_combination_classifies_interaction(self):
        from dojo.core.evidence import EvidenceLogger
        from dojo.research.combination_composer import CombinationComposer, CombinationPair
        from dojo.research.gpu_measurements import make_measurement_fns
        from dojo.research.interaction_model import InteractionEffect

        baseline_fn, flash_fn, int4_fn, combined_fn = make_measurement_fns(
            _MODEL,
            bench_steps=10,
            warmup_steps=2,
            seq_len=256,
        )

        pair = CombinationPair(
            technique_a="flash_attention",
            technique_b="int4_bnb",
            model=_MODEL,
            metric="tokens_per_sec",
            higher_is_better=True,
            description="Flash Attention SDPA (d15) + bitsandbytes NF4 INT4 (d25): speed interaction",
        )

        composer = CombinationComposer(emit_evidence=True)
        result = composer.run(pair, baseline_fn, flash_fn, int4_fn, combined_fn)

        assert result.success, f"Combination test failed: {result.error}"
        assert result.baseline > 0, f"Baseline tps must be positive, got {result.baseline}"
        assert result.a_only > 0, f"Flash Attention tps must be positive, got {result.a_only}"
        assert result.b_only > 0, f"INT4 tps must be positive, got {result.b_only}"
        assert result.combined > 0, f"Combined tps must be positive, got {result.combined}"
        assert result.interaction != InteractionEffect.UNKNOWN, (
            "Interaction must be classified (not UNKNOWN)"
        )
        assert result.elapsed_s > 0

        # Evidence chain must be sealed
        events = EvidenceLogger.get_events()
        event_strs = [str(e) for e in events]
        assert any("COMBINATION_INTERACTION_CLASSIFIED" in s for s in event_strs), (
            "COMBINATION_INTERACTION_CLASSIFIED evidence must be emitted"
        )
        assert any("COMBINATION_TEST_PROPOSED" in s for s in event_strs), (
            "COMBINATION_TEST_PROPOSED evidence must be emitted"
        )

        # Report results
        print(f"\n=== Flash Attention + INT4 Combination Results ===")
        print(f"  Model:    {_MODEL}")
        print(f"  Baseline (FP16 eager):    {result.baseline:.1f} tps")
        print(f"  A only   (FP16 SDPA):     {result.a_only:.1f} tps")
        print(f"  B only   (INT4 eager):    {result.b_only:.1f} tps")
        print(f"  Combined (INT4 SDPA):     {result.combined:.1f} tps")
        print(f"  Interaction:              {result.interaction.value}")
        print(f"  Synergy score:            {result.synergy_score:+.4f}")
        print(f"  Elapsed:                  {result.elapsed_s:.1f}s")
        print(f"  Evidence events:          {len(events)}")

    def test_all_four_measurements_are_positive(self):
        """All 4 configurations must produce valid positive tps measurements.

        Note: bitsandbytes NF4 INT4 can be SLOWER than BF16 on small models (1.5B)
        because dequantization overhead exceeds the memory-bandwidth savings on fast
        GPUs. This is expected and documented — INT4 gains manifest on 7B+ models
        where VRAM bandwidth is the bottleneck. The combination test still classifies
        the interaction effect accurately.
        """
        from dojo.research.combination_composer import CombinationComposer, CombinationPair
        from dojo.research.gpu_measurements import make_measurement_fns

        baseline_fn, flash_fn, int4_fn, combined_fn = make_measurement_fns(
            _MODEL,
            bench_steps=8,
            warmup_steps=2,
            seq_len=256,
        )

        pair = CombinationPair(
            technique_a="flash_attention",
            technique_b="int4_bnb",
            model=_MODEL,
            metric="tokens_per_sec",
            higher_is_better=True,
        )
        composer = CombinationComposer(emit_evidence=False)
        result = composer.run(pair, baseline_fn, flash_fn, int4_fn, combined_fn)

        assert result.success, result.error
        assert result.baseline > 0, f"Baseline must be positive: {result.baseline}"
        assert result.a_only > 0, f"Flash Attention measurement must be positive: {result.a_only}"
        assert result.b_only > 0, f"INT4 measurement must be positive: {result.b_only}"
        assert result.combined > 0, f"Combined measurement must be positive: {result.combined}"
        assert result.elapsed_s > 0

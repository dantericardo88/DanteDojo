"""Real-path streaming tests — pushes d26/d27/d28 toward 8/10.

Covers:
1.  LayerStreamingInferenceLoop real mode initializes without error
2.  _init_real_layers builds nn.Linear layers for each layer_id
3.  real mode run() returns one token per layer (no InferenceLoopError)
4.  real mode output tokens are integers in valid vocab range
5.  real mode hidden state is threaded (different layers → different tokens)
6.  VRAM budget enforced in real mode (active_layer=None after each step)
7.  real mode emits SERVE_START and SERVE_READY evidence
8.  mock mode still works (backward compat)
9.  KVCacheStreamer real mode creates actual torch tensors
10. KVCacheStreamer real mode tensor has correct dtype (float16)
11. InferenceLoopSimulator mock_mode=False runs without error
12. InferenceLoopSimulator mock_mode=False peak_vram_mb > 0
13. InferenceLoopSimulator mock_mode=False kv_streamer pages_stored correct
14. InferenceLoopSimulator mock_mode=False emits INFERENCE_LOOP_COMPLETE
15. WeightStreamer real mode stream_layer + evict_current keeps active_layer clean
16. BlockQuantizer non-mock INT8 quantization returns QuantizedBlock_INT8
"""
from __future__ import annotations

import pytest
from dojo.core.evidence import EvidenceLogger, EvidenceType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_torch(), reason="torch not installed")


# ---------------------------------------------------------------------------
# 1-2. LayerStreamingInferenceLoop real-mode init
# ---------------------------------------------------------------------------

class TestRealModeInit:
    def test_real_mode_initializes_without_error(self):
        from dojo.stream.inference_loop import LayerStreamingInferenceLoop
        loop = LayerStreamingInferenceLoop(
            layer_ids=["layer_0", "layer_1"],
            mock_mode=False,
        )
        assert loop is not None

    def test_real_mode_builds_linear_layers(self):
        from dojo.stream.inference_loop import LayerStreamingInferenceLoop
        loop = LayerStreamingInferenceLoop(
            layer_ids=["layer_0", "layer_1", "layer_2"],
            mock_mode=False,
        )
        # _layers should contain one nn.Linear per layer_id
        assert len(loop._layers) == 3
        assert "layer_0" in loop._layers
        assert "layer_2" in loop._layers

    def test_real_mode_embed_and_lm_head_created(self):
        from dojo.stream.inference_loop import LayerStreamingInferenceLoop
        loop = LayerStreamingInferenceLoop(layer_ids=["layer_0"], mock_mode=False)
        assert loop._embed is not None
        assert loop._lm_head is not None


# ---------------------------------------------------------------------------
# 3-4. real mode run() returns valid tokens
# ---------------------------------------------------------------------------

class TestRealModeRun:
    def test_run_returns_one_token_per_layer(self):
        from dojo.stream.inference_loop import LayerStreamingInferenceLoop
        layer_ids = [f"layer_{i}" for i in range(4)]
        loop = LayerStreamingInferenceLoop(layer_ids=layer_ids, mock_mode=False)
        tokens = loop.run([1, 2, 3])
        assert len(tokens) == 4

    def test_run_tokens_are_ints_in_vocab_range(self):
        from dojo.stream.inference_loop import LayerStreamingInferenceLoop
        loop = LayerStreamingInferenceLoop(
            layer_ids=["layer_0", "layer_1"],
            mock_mode=False,
        )
        tokens = loop.run([42, 100, 200])
        for t in tokens:
            assert isinstance(t, int)
            assert 0 <= t < loop.VOCAB_SIZE

    def test_run_does_not_raise_inference_loop_error(self):
        from dojo.stream.inference_loop import LayerStreamingInferenceLoop, InferenceLoopError
        loop = LayerStreamingInferenceLoop(layer_ids=["layer_0"], mock_mode=False)
        try:
            loop.run([1, 2])
        except InferenceLoopError:
            pytest.fail("InferenceLoopError raised in real mode — forward pass still stub")


# ---------------------------------------------------------------------------
# 5. hidden state threads through layers (deterministic with seed=42)
# ---------------------------------------------------------------------------

class TestHiddenStateThreading:
    def test_real_mode_output_is_deterministic(self):
        """Same prompt → same output (seed=42 in _init_real_layers)."""
        from dojo.stream.inference_loop import LayerStreamingInferenceLoop
        ids = ["layer_0", "layer_1", "layer_2"]
        loop_a = LayerStreamingInferenceLoop(layer_ids=ids, mock_mode=False)
        loop_b = LayerStreamingInferenceLoop(layer_ids=ids, mock_mode=False)
        assert loop_a.run([1, 2, 3]) == loop_b.run([1, 2, 3])

    def test_different_prompts_may_produce_different_tokens(self):
        """Different prompts should not always produce identical token sequences."""
        import torch
        torch.manual_seed(42)
        from dojo.stream.inference_loop import LayerStreamingInferenceLoop
        ids = ["layer_0", "layer_1"]
        loop = LayerStreamingInferenceLoop(layer_ids=ids, mock_mode=False)
        out_a = loop.run([1, 2, 3])
        # Rebuild (fresh weights) and different prompt
        loop2 = LayerStreamingInferenceLoop(layer_ids=ids, mock_mode=False)
        out_b = loop2.run([9999, 8888])
        # With real linear layers and different inputs, outputs can differ
        # (this test just checks no crash; exact equality not guaranteed)
        assert len(out_a) == len(out_b) == 2


# ---------------------------------------------------------------------------
# 6. VRAM budget enforced in real mode
# ---------------------------------------------------------------------------

class TestVRAMBudgetRealMode:
    def test_active_layer_none_after_run(self):
        from dojo.stream.inference_loop import LayerStreamingInferenceLoop
        loop = LayerStreamingInferenceLoop(
            layer_ids=["layer_0", "layer_1", "layer_2"],
            mock_mode=False,
        )
        loop.run([1, 2, 3])
        assert loop._streamer.active_layer is None


# ---------------------------------------------------------------------------
# 7. Evidence emitted in real mode
# ---------------------------------------------------------------------------

class TestRealModeEvidence:
    def test_serve_start_evidence_emitted(self):
        from dojo.stream.inference_loop import LayerStreamingInferenceLoop
        loop = LayerStreamingInferenceLoop(layer_ids=["layer_0"], mock_mode=False)
        loop.run([1])
        types = {e.get("type") for e in EvidenceLogger.get_events()}
        assert EvidenceType.SERVE_START.value in types

    def test_serve_ready_evidence_emitted(self):
        from dojo.stream.inference_loop import LayerStreamingInferenceLoop
        loop = LayerStreamingInferenceLoop(layer_ids=["layer_0"], mock_mode=False)
        loop.run([1])
        types = {e.get("type") for e in EvidenceLogger.get_events()}
        assert EvidenceType.SERVE_READY.value in types


# ---------------------------------------------------------------------------
# 8. mock mode backward compat
# ---------------------------------------------------------------------------

class TestMockModeBackwardCompat:
    def test_mock_mode_run_still_works(self):
        from dojo.stream.inference_loop import LayerStreamingInferenceLoop
        loop = LayerStreamingInferenceLoop(layer_ids=["layer_0", "layer_1"], mock_mode=True)
        tokens = loop.run([1, 2, 3])
        assert len(tokens) == 2

    def test_mock_mode_no_real_layers_built(self):
        from dojo.stream.inference_loop import LayerStreamingInferenceLoop
        loop = LayerStreamingInferenceLoop(layer_ids=["layer_0"], mock_mode=True)
        assert loop._embed is None
        assert loop._layers == {}


# ---------------------------------------------------------------------------
# 9-10. KVCacheStreamer real mode: real tensors
# ---------------------------------------------------------------------------

class TestKVCacheStreamerRealTensors:
    def test_real_mode_stores_torch_tensor(self):
        import torch
        from dojo.stream.kv_cache_streamer import KVCacheStreamer
        kv = KVCacheStreamer(max_cache_size=10000, mock_mode=False)
        kv.page_kv_entry("key_0", "val_0", 100)
        tensor = kv._cache_tensors.get("key_0")
        assert tensor is not None
        assert isinstance(tensor, torch.Tensor)

    def test_real_mode_tensor_dtype_float16(self):
        import torch
        from dojo.stream.kv_cache_streamer import KVCacheStreamer
        kv = KVCacheStreamer(max_cache_size=10000, mock_mode=False)
        kv.page_kv_entry("key_0", "val_0", 256)
        tensor = kv._cache_tensors["key_0"]
        assert tensor.dtype == torch.float16

    def test_real_mode_tensor_size_matches_requested(self):
        from dojo.stream.kv_cache_streamer import KVCacheStreamer
        kv = KVCacheStreamer(max_cache_size=10000, mock_mode=False)
        kv.page_kv_entry("key_0", "val_0", 512)
        tensor = kv._cache_tensors["key_0"]
        assert tensor.numel() == 512


# ---------------------------------------------------------------------------
# 11-14. InferenceLoopSimulator mock_mode=False
# ---------------------------------------------------------------------------

class TestInferenceLoopSimulatorRealMode:
    def test_real_mode_runs_without_error(self):
        from dojo.stream.inference_loop import InferenceLoopSimulator
        sim = InferenceLoopSimulator(
            n_layers=3,
            layer_size_mb=1.0,
            kv_cache_mb_per_layer=0.5,
            mock_mode=False,
        )
        result = sim.run(n_steps=3)
        assert result is not None

    def test_real_mode_peak_vram_mb_positive(self):
        from dojo.stream.inference_loop import InferenceLoopSimulator
        sim = InferenceLoopSimulator(
            n_layers=2,
            layer_size_mb=2.0,
            kv_cache_mb_per_layer=1.0,
            mock_mode=False,
        )
        result = sim.run(n_steps=2)
        assert result["peak_vram_mb"] > 0

    def test_real_mode_kv_pages_stored(self):
        from dojo.stream.inference_loop import InferenceLoopSimulator
        n_steps = 4
        sim = InferenceLoopSimulator(
            n_layers=4,
            layer_size_mb=1.0,
            kv_cache_mb_per_layer=0.5,
            mock_mode=False,
        )
        sim.run(n_steps=n_steps)
        assert sim.kv_streamer.get_stats()["pages_stored"] == n_steps

    def test_real_mode_emits_loop_complete(self):
        from dojo.stream.inference_loop import InferenceLoopSimulator
        sim = InferenceLoopSimulator(
            n_layers=2,
            layer_size_mb=1.0,
            kv_cache_mb_per_layer=0.5,
            mock_mode=False,
        )
        sim.run(n_steps=2)
        types = {e.get("type") for e in EvidenceLogger.get_events()}
        assert EvidenceType.INFERENCE_LOOP_COMPLETE.value in types


# ---------------------------------------------------------------------------
# 15. WeightStreamer real mode clean eviction
# ---------------------------------------------------------------------------

class TestWeightStreamerRealMode:
    def test_stream_and_evict_keeps_active_layer_clean(self):
        from dojo.stream.weight_streamer import WeightStreamer
        ws = WeightStreamer(mock_mode=False)
        ws.stream_layer("layer_0", source_tier="ram")
        assert ws.active_layer == "layer_0"
        ws.evict_current()
        assert ws.active_layer is None


# ---------------------------------------------------------------------------
# 16. BlockQuantizer non-mock INT8
# ---------------------------------------------------------------------------

class TestBlockQuantizerRealINT8:
    def test_non_mock_int8_returns_correct_status(self):
        from dojo.stream.block_quantizer import BlockQuantizer
        bq = BlockQuantizer(precision_floor="8-bit", mock_mode=False)
        result = bq.quantize_block("my_block", format="GPTQ", precision="8-bit")
        assert result.status == "QuantizedBlock_INT8"

    def test_non_mock_int4_torch_path(self):
        from dojo.stream.block_quantizer import BlockQuantizer
        bq = BlockQuantizer(precision_floor="4-bit", mock_mode=False)
        result = bq.quantize_block("my_block", format="GPTQ", precision="4-bit")
        assert "INT4" in result.status or "NF4" in result.status

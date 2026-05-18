"""Tests for d26: VRAM-bounded layer streaming with realistic hidden dims + byte tracking.

Verifies that LayerStreamingInferenceLoop can scale to production-size hidden
dimensions, computes real VRAM byte budgets, enforces a configurable budget,
and tracks peak VRAM during execution.
"""
from __future__ import annotations

import pytest

from dojo.stream.inference_loop import (
    InferenceLoopError,
    LayerStreamingInferenceLoop,
)


# ---------------------------------------------------------------------------
# Configurable hidden_dim
# ---------------------------------------------------------------------------

class TestConfigurableHiddenDim:
    def test_default_hidden_dim(self):
        loop = LayerStreamingInferenceLoop(["L0"], mock_mode=True)
        assert loop.HIDDEN_DIM == 64

    def test_realistic_hidden_dim_768(self):
        loop = LayerStreamingInferenceLoop(["L0", "L1"], mock_mode=True, hidden_dim=768)
        assert loop.HIDDEN_DIM == 768

    def test_large_hidden_dim_1024(self):
        loop = LayerStreamingInferenceLoop(["L0"], mock_mode=True, hidden_dim=1024)
        assert loop.HIDDEN_DIM == 1024


# ---------------------------------------------------------------------------
# VRAM byte calculations
# ---------------------------------------------------------------------------

class TestVramByteCalculations:
    def test_per_layer_bytes_64_dim_float32(self):
        loop = LayerStreamingInferenceLoop(["L0"], mock_mode=True, hidden_dim=64)
        # 64 * 64 * 4 = 16384 bytes
        assert loop.layer_vram_bytes() == 16_384

    def test_per_layer_bytes_768_dim_float32(self):
        loop = LayerStreamingInferenceLoop(["L0"], mock_mode=True, hidden_dim=768)
        # 768 * 768 * 4 = 2_359_296 bytes
        assert loop.layer_vram_bytes() == 2_359_296

    def test_per_layer_bytes_bfloat16_is_half_of_float32(self):
        loop = LayerStreamingInferenceLoop(["L0"], mock_mode=True, hidden_dim=768)
        assert loop.layer_vram_bytes(dtype_bytes=2) == loop.layer_vram_bytes(dtype_bytes=4) // 2

    def test_total_model_bytes_scales_with_layer_count(self):
        loop = LayerStreamingInferenceLoop(["L0", "L1", "L2", "L3"], mock_mode=True, hidden_dim=64)
        assert loop.total_model_vram_bytes() == loop.layer_vram_bytes() * 4

    def test_streaming_savings_is_n_layers(self):
        n = 12
        layer_ids = [f"L{i}" for i in range(n)]
        loop = LayerStreamingInferenceLoop(layer_ids, mock_mode=True, hidden_dim=512)
        # streaming holds 1 layer; total holds n — savings ratio = (n-1)/n
        ratio = (loop.total_model_vram_bytes() - loop.layer_vram_bytes()) / loop.total_model_vram_bytes()
        assert ratio == (n - 1) / n


# ---------------------------------------------------------------------------
# VRAM budget enforcement
# ---------------------------------------------------------------------------

class TestVramBudgetEnforcement:
    def test_run_succeeds_when_layer_fits_budget(self):
        # 64-dim layer = 16KB, budget 1MB — fits easily
        loop = LayerStreamingInferenceLoop(
            ["L0", "L1"], mock_mode=True, hidden_dim=64, vram_budget_bytes=1_000_000,
        )
        tokens = loop.run([1, 2, 3])
        assert len(tokens) == 2

    def test_run_raises_when_layer_exceeds_budget(self):
        # 1024-dim layer = 4MB, budget 1KB — refuse to run
        loop = LayerStreamingInferenceLoop(
            ["L0"], mock_mode=True, hidden_dim=1024, vram_budget_bytes=1024,
        )
        with pytest.raises(InferenceLoopError, match="exceeds budget"):
            loop.run([1])

    def test_budget_zero_disables_enforcement(self):
        loop = LayerStreamingInferenceLoop(
            ["L0"], mock_mode=True, hidden_dim=1024, vram_budget_bytes=0,
        )
        tokens = loop.run([1])  # should not raise even though no budget set
        assert len(tokens) == 1


# ---------------------------------------------------------------------------
# Peak VRAM tracking
# ---------------------------------------------------------------------------

class TestPeakVramTracking:
    def test_peak_zero_before_run(self):
        loop = LayerStreamingInferenceLoop(["L0"], mock_mode=True)
        assert loop.peak_vram_bytes == 0

    def test_peak_equals_one_layer_after_run(self):
        loop = LayerStreamingInferenceLoop(["L0", "L1", "L2"], mock_mode=True, hidden_dim=128)
        loop.run([1, 2])
        assert loop.peak_vram_bytes == loop.layer_vram_bytes()

    def test_peak_independent_of_layer_count(self):
        loop_3 = LayerStreamingInferenceLoop(["L0", "L1", "L2"], mock_mode=True, hidden_dim=128)
        loop_10 = LayerStreamingInferenceLoop([f"L{i}" for i in range(10)], mock_mode=True, hidden_dim=128)
        loop_3.run([1])
        loop_10.run([1])
        # Peak VRAM does not grow with layer count under streaming
        assert loop_3.peak_vram_bytes == loop_10.peak_vram_bytes


# ---------------------------------------------------------------------------
# Real torch forward pass with larger hidden dim
# ---------------------------------------------------------------------------

class TestRealForwardLargerHiddenDim:
    def test_run_with_hidden_dim_256_completes(self):
        loop = LayerStreamingInferenceLoop(
            ["L0", "L1"], mock_mode=False, hidden_dim=256,
        )
        tokens = loop.run([1, 2, 3])
        assert len(tokens) == 2
        assert all(isinstance(t, int) for t in tokens)

    def test_run_with_hidden_dim_512_completes(self):
        loop = LayerStreamingInferenceLoop(
            ["L0"], mock_mode=False, hidden_dim=512,
        )
        tokens = loop.run([1])
        assert len(tokens) == 1

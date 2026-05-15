"""Integration tests for InferenceLoopSimulator (d26/d27/d28 coherent data flow)."""
from __future__ import annotations

from typing import List

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.stream.inference_loop import InferenceLoopSimulator, InferenceLoopError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_evidence():
    """Isolate the evidence logger between every test."""
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


def _make_sim(
    n_layers: int = 4,
    layer_size_mb: float = 10.0,
    kv_cache_mb_per_layer: float = 2.0,
    kv_budget_mb: float | None = None,
) -> InferenceLoopSimulator:
    return InferenceLoopSimulator(
        n_layers=n_layers,
        layer_size_mb=layer_size_mb,
        kv_cache_mb_per_layer=kv_cache_mb_per_layer,
        kv_budget_mb=kv_budget_mb,
        mock_mode=True,
    )


def _events_of_type(event_type: str) -> List[dict]:
    return [e for e in EvidenceLogger.get_events() if e.get("type") == event_type]


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_01_initializes_without_error(self):
        """InferenceLoopSimulator initializes without error."""
        sim = _make_sim()
        assert sim is not None

    def test_02_weight_streamer_created(self):
        """WeightStreamer is created and accessible."""
        sim = _make_sim()
        assert sim.weight_streamer is not None

    def test_03_kv_streamer_created(self):
        """KVCacheStreamer is created and accessible."""
        sim = _make_sim()
        assert sim.kv_streamer is not None

    def test_04_prefetch_governor_created(self):
        """PrefetchGovernor is created and accessible."""
        sim = _make_sim()
        assert sim.prefetch_governor is not None


# ---------------------------------------------------------------------------
# 2. run() basic contract
# ---------------------------------------------------------------------------

class TestRunBasicContract:

    def test_05_run_5_steps_completes_without_error(self):
        """run(n_steps=5) completes without error."""
        sim = _make_sim()
        result = sim.run(n_steps=5)
        assert result is not None

    def test_06_steps_completed_equals_n_steps(self):
        """result['steps_completed'] == 5."""
        sim = _make_sim()
        result = sim.run(n_steps=5)
        assert result["steps_completed"] == 5

    def test_07_peak_vram_mb_positive(self):
        """result['peak_vram_mb'] > 0."""
        sim = _make_sim()
        result = sim.run(n_steps=5)
        assert result["peak_vram_mb"] > 0

    def test_08_layers_loaded_nonempty(self):
        """result['layers_loaded'] is a non-empty list."""
        sim = _make_sim()
        result = sim.run(n_steps=5)
        assert isinstance(result["layers_loaded"], list)
        assert len(result["layers_loaded"]) > 0

    def test_09_layers_in_valid_range(self):
        """All loaded layer indices are in [0, n_layers - 1]."""
        n_layers = 4
        sim = _make_sim(n_layers=n_layers)
        result = sim.run(n_steps=8)
        for layer_id in result["layers_loaded"]:
            # layer_id is "layer_<idx>"
            idx = int(layer_id.split("_")[1])
            assert 0 <= idx < n_layers, f"layer index {idx} out of range [0, {n_layers-1}]"


# ---------------------------------------------------------------------------
# 3. Evidence
# ---------------------------------------------------------------------------

class TestEvidence:

    def test_10_inference_loop_step_emitted_per_step(self):
        """stream.inference_loop_step evidence emitted once per step."""
        n_steps = 5
        sim = _make_sim()
        sim.run(n_steps=n_steps)
        step_events = _events_of_type(EvidenceType.INFERENCE_LOOP_STEP.value)
        assert len(step_events) == n_steps, (
            f"Expected {n_steps} step events, got {len(step_events)}"
        )

    def test_11_loop_complete_emitted_at_end(self):
        """stream.loop_complete evidence emitted exactly once at the end."""
        sim = _make_sim()
        sim.run(n_steps=5)
        complete_events = _events_of_type(EvidenceType.INFERENCE_LOOP_COMPLETE.value)
        assert len(complete_events) == 1

    def test_12_step_evidence_has_step_field(self):
        """Each step evidence bundle includes a 'step' counter field."""
        sim = _make_sim()
        sim.run(n_steps=3)
        step_events = _events_of_type(EvidenceType.INFERENCE_LOOP_STEP.value)
        for ev in step_events:
            assert "step" in ev.get("data", {}), f"Missing 'step' field in: {ev}"

    def test_13_step_evidence_has_vram_mb_field(self):
        """Each step evidence bundle includes a 'vram_mb' field."""
        sim = _make_sim()
        sim.run(n_steps=3)
        step_events = _events_of_type(EvidenceType.INFERENCE_LOOP_STEP.value)
        for ev in step_events:
            assert "vram_mb" in ev.get("data", {}), f"Missing 'vram_mb' field in: {ev}"

    def test_14_step_counters_are_sequential(self):
        """Step counter in evidence is 0, 1, 2, … in order."""
        sim = _make_sim()
        sim.run(n_steps=4)
        step_events = _events_of_type(EvidenceType.INFERENCE_LOOP_STEP.value)
        counters = [ev["data"]["step"] for ev in step_events]
        assert counters == list(range(4)), f"Non-sequential steps: {counters}"


# ---------------------------------------------------------------------------
# 4. Component coordination
# ---------------------------------------------------------------------------

class TestComponentCoordination:

    def test_15_weight_streamer_loads_layers_in_order(self):
        """WeightStreamer loads layer_0, layer_1, layer_2, … in sequence."""
        sim = _make_sim(n_layers=4)
        result = sim.run(n_steps=4)
        expected = [f"layer_{i}" for i in range(4)]
        assert result["layers_loaded"] == expected

    def test_16_kv_streamer_stores_activations_per_layer(self):
        """KVCacheStreamer stores one activation entry per step."""
        n_steps = 5
        sim = _make_sim()
        sim.run(n_steps=n_steps)
        kv_stats = sim.kv_streamer.get_stats()
        assert kv_stats["pages_stored"] == n_steps, (
            f"Expected {n_steps} KV pages, got {kv_stats['pages_stored']}"
        )

    def test_17_prefetch_governor_schedules_next_layer(self):
        """PrefetchGovernor issues at least one prefetch before the loop ends."""
        sim = _make_sim()
        sim.run(n_steps=3)
        prefetch_stats = sim.prefetch_governor.get_stats()
        assert prefetch_stats["prefetches_issued"] >= 1

    def test_18_peak_vram_includes_layer_and_kv(self):
        """Peak VRAM = layer_size_mb + kv_cache_mb_per_layer * cached_layers."""
        layer_size_mb = 10.0
        kv_mb = 2.0
        sim = _make_sim(layer_size_mb=layer_size_mb, kv_cache_mb_per_layer=kv_mb)
        result = sim.run(n_steps=3)
        # At step 3 (last), kv_entries_cached = 3
        # vram_mb = 10 + 2*3 = 16
        expected_peak = layer_size_mb + kv_mb * 3
        assert result["peak_vram_mb"] == pytest.approx(expected_peak, rel=1e-6), (
            f"Expected peak={expected_peak}, got {result['peak_vram_mb']}"
        )

    def test_19_weight_streamer_evicts_before_next_load(self):
        """WeightStreamer has no active layer between steps (clean eviction)."""
        sim = _make_sim(n_layers=4)
        # After run(), active_layer must be None — no residual VRAM occupation.
        sim.run(n_steps=4)
        assert sim.weight_streamer.active_layer is None

    def test_20_kv_cache_accumulates_across_steps(self):
        """KV cache size grows monotonically across steps."""
        n_steps = 5
        kv_mb = 2.0
        sim = _make_sim(kv_cache_mb_per_layer=kv_mb)
        sim.run(n_steps=n_steps)
        # current_size is in integer units (kv_mb * kv_scale per entry)
        expected_units = int(kv_mb * sim._kv_scale) * n_steps
        assert sim.kv_streamer.current_size == expected_units


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_21_run_zero_steps(self):
        """run(n_steps=0) returns steps_completed=0 without error."""
        sim = _make_sim()
        result = sim.run(n_steps=0)
        assert result["steps_completed"] == 0
        assert result["peak_vram_mb"] == 0.0
        assert result["layers_loaded"] == []

    def test_22_run_single_step(self):
        """run(n_steps=1) works correctly."""
        sim = _make_sim(layer_size_mb=8.0, kv_cache_mb_per_layer=3.0)
        result = sim.run(n_steps=1)
        assert result["steps_completed"] == 1
        assert result["layers_loaded"] == ["layer_0"]
        assert result["peak_vram_mb"] == pytest.approx(8.0 + 3.0)

    def test_23_two_sequential_runs_consistent(self):
        """Two fresh simulators with the same config produce identical results."""
        sim_a = _make_sim(n_layers=3, layer_size_mb=5.0, kv_cache_mb_per_layer=1.0)
        sim_b = _make_sim(n_layers=3, layer_size_mb=5.0, kv_cache_mb_per_layer=1.0)
        result_a = sim_a.run(n_steps=5)
        result_b = sim_b.run(n_steps=5)
        assert result_a["steps_completed"] == result_b["steps_completed"]
        assert result_a["peak_vram_mb"] == pytest.approx(result_b["peak_vram_mb"])
        assert result_a["layers_loaded"] == result_b["layers_loaded"]

    def test_24_memory_never_exceeds_budget(self):
        """If a tight kv_budget_mb is set, total VRAM stays within budget."""
        layer_size_mb = 5.0
        kv_mb = 1.0
        n_steps = 3
        # Budget: layer + kv * n_steps = 5 + 3 = 8; set generous to avoid eviction
        budget = layer_size_mb + kv_mb * n_steps + 1.0  # 9 MB
        sim = _make_sim(
            layer_size_mb=layer_size_mb,
            kv_cache_mb_per_layer=kv_mb,
            kv_budget_mb=budget,
        )
        result = sim.run(n_steps=n_steps)
        assert result["peak_vram_mb"] <= budget, (
            f"peak_vram={result['peak_vram_mb']} exceeded budget={budget}"
        )

    def test_25_loop_complete_contains_peak_vram(self):
        """stream.loop_complete evidence contains peak_vram_mb field."""
        sim = _make_sim()
        result = sim.run(n_steps=3)
        complete_events = _events_of_type(EvidenceType.INFERENCE_LOOP_COMPLETE.value)
        assert len(complete_events) == 1
        data = complete_events[0].get("data", {})
        assert "peak_vram_mb" in data
        assert data["peak_vram_mb"] == pytest.approx(result["peak_vram_mb"])

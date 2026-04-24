"""Unit tests for dojo/idle/ — Phase 2 idle training subsystem.

Covers:
- IdleDetector.is_idle() returns bool
- ThermalGuard.should_pause() returns False when pynvml unavailable
- ThermalGuard.can_resume() returns True when temp=0
- PowerGuard.is_on_battery() returns bool (doesn't raise)
- ResourceGuard.all_guards_clear() returns (True, []) when all mocked safe
- GradientAccumulator: record 8 microbatches with accum_steps=8 → last returns True
- GradientAccumulator: save/load round-trips state correctly
- GradientAccumulator: idempotent after optimizer_step_taken
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.idle.accumulator import GradientAccumulator
from dojo.idle.detector import IdleDetector
from dojo.idle.guardian import PowerGuard, ThermalGuard
from dojo.idle.resource import ResourceGuard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ---------------------------------------------------------------------------
# IdleDetector
# ---------------------------------------------------------------------------

class TestIdleDetector:
    def test_is_idle_returns_bool(self):
        detector = IdleDetector(threshold_seconds=300)
        result = detector.is_idle()
        assert isinstance(result, bool)

    def test_is_idle_true_when_seconds_above_threshold(self):
        detector = IdleDetector(threshold_seconds=60)
        with patch.object(detector, "seconds_since_last_input", return_value=120.0):
            assert detector.is_idle() is True

    def test_is_idle_false_when_seconds_below_threshold(self):
        detector = IdleDetector(threshold_seconds=300)
        with patch.object(detector, "seconds_since_last_input", return_value=10.0):
            assert detector.is_idle() is False

    def test_is_idle_exactly_at_threshold_is_idle(self):
        detector = IdleDetector(threshold_seconds=60)
        with patch.object(detector, "seconds_since_last_input", return_value=60.0):
            assert detector.is_idle() is True

    def test_seconds_since_last_input_returns_float(self):
        detector = IdleDetector()
        result = detector.seconds_since_last_input()
        assert isinstance(result, float)
        assert result >= 0.0

    def test_idle_window_detected_evidence_emitted_on_transition(self):
        """Evidence emitted when state transitions from active to idle."""
        detector = IdleDetector(threshold_seconds=60)
        # First call: active (below threshold)
        with patch.object(detector, "seconds_since_last_input", return_value=10.0):
            detector.is_idle()
        # Second call: idle (above threshold) → edge-triggered event
        with patch.object(detector, "seconds_since_last_input", return_value=120.0):
            detector.is_idle()

        from dojo.core.evidence import EvidenceType
        events = EvidenceLogger.get_events()
        idle_events = [
            e for e in events
            if isinstance(e.get("type"), str)
            and EvidenceType.IDLE_WINDOW_DETECTED.value in e.get("type", "")
        ]
        assert len(idle_events) >= 1

    def test_no_evidence_when_stays_idle(self):
        """Staying idle should not re-emit IDLE_WINDOW_DETECTED on each poll."""
        detector = IdleDetector(threshold_seconds=60)
        # First call: transition active → idle
        with patch.object(detector, "seconds_since_last_input", return_value=10.0):
            detector.is_idle()
        with patch.object(detector, "seconds_since_last_input", return_value=120.0):
            detector.is_idle()

        EvidenceLogger.clear()

        # Subsequent idle calls — no new events expected
        with patch.object(detector, "seconds_since_last_input", return_value=180.0):
            detector.is_idle()
        with patch.object(detector, "seconds_since_last_input", return_value=240.0):
            detector.is_idle()

        from dojo.core.evidence import EvidenceType
        events = EvidenceLogger.get_events()
        idle_events = [
            e for e in events
            if isinstance(e.get("type"), str)
            and EvidenceType.IDLE_WINDOW_DETECTED.value in e.get("type", "")
        ]
        assert len(idle_events) == 0

    def test_custom_threshold_respected(self):
        detector = IdleDetector(threshold_seconds=120)
        with patch.object(detector, "seconds_since_last_input", return_value=119.9):
            assert detector.is_idle() is False
        with patch.object(detector, "seconds_since_last_input", return_value=120.0):
            assert detector.is_idle() is True


# ---------------------------------------------------------------------------
# ThermalGuard
# ---------------------------------------------------------------------------

class TestThermalGuard:
    def test_should_pause_false_when_pynvml_unavailable(self):
        guard = ThermalGuard()
        # pynvml raises ImportError → get_gpu_temp_celsius returns 0.0
        with patch.object(guard, "get_gpu_temp_celsius", return_value=0.0):
            assert guard.should_pause() is False

    def test_should_pause_true_when_temp_above_threshold(self):
        guard = ThermalGuard()
        with patch.object(guard, "get_gpu_temp_celsius", return_value=85.0):
            assert guard.should_pause() is True

    def test_should_pause_false_when_temp_at_threshold(self):
        """At exactly 80°C the guard should NOT pause (> not >=)."""
        guard = ThermalGuard()
        with patch.object(guard, "get_gpu_temp_celsius", return_value=80.0):
            assert guard.should_pause() is False

    def test_can_resume_true_when_temp_zero(self):
        guard = ThermalGuard()
        with patch.object(guard, "get_gpu_temp_celsius", return_value=0.0):
            assert guard.can_resume() is True

    def test_can_resume_true_below_resume_threshold(self):
        guard = ThermalGuard()
        # resume threshold = 80 - 5 = 75°C; 74°C should allow resume
        with patch.object(guard, "get_gpu_temp_celsius", return_value=74.0):
            assert guard.can_resume() is True

    def test_can_resume_false_when_temp_above_resume_threshold(self):
        guard = ThermalGuard()
        with patch.object(guard, "get_gpu_temp_celsius", return_value=76.0):
            assert guard.can_resume() is False

    def test_get_gpu_temp_celsius_returns_float_when_pynvml_missing(self):
        guard = ThermalGuard()
        temp = guard.get_gpu_temp_celsius()
        assert isinstance(temp, float)

    def test_hysteresis_constants(self):
        assert ThermalGuard.PAUSE_THRESHOLD_CELSIUS == 80.0
        assert ThermalGuard.HYSTERESIS_CELSIUS == 5.0


# ---------------------------------------------------------------------------
# PowerGuard
# ---------------------------------------------------------------------------

class TestPowerGuard:
    def test_is_on_battery_returns_bool(self):
        guard = PowerGuard()
        result = guard.is_on_battery()
        assert isinstance(result, bool)

    def test_is_on_battery_does_not_raise(self):
        guard = PowerGuard()
        # Must not raise regardless of psutil availability
        try:
            guard.is_on_battery()
        except Exception as e:
            pytest.fail(f"is_on_battery() raised unexpectedly: {e}")

    def test_should_pause_false_when_plugged_in(self):
        guard = PowerGuard()
        with patch.object(guard, "is_on_battery", return_value=False):
            assert guard.should_pause() is False

    def test_should_pause_true_when_on_battery(self):
        guard = PowerGuard()
        with patch.object(guard, "is_on_battery", return_value=True):
            assert guard.should_pause() is True

    def test_is_on_battery_false_when_psutil_returns_none(self):
        """When psutil returns no battery info (desktop), assume plugged in."""
        guard = PowerGuard()
        with patch("dojo.idle.guardian.PowerGuard.is_on_battery", return_value=False):
            assert guard.is_on_battery() is False

    def test_is_on_battery_returns_false_when_psutil_unavailable(self):
        """If psutil import fails, should return False (safe default)."""
        guard = PowerGuard()
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "psutil":
                raise ImportError("psutil not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = guard.is_on_battery()
        assert result is False


# ---------------------------------------------------------------------------
# ResourceGuard
# ---------------------------------------------------------------------------

class TestResourceGuard:
    def test_all_guards_clear_returns_tuple(self):
        guard = ResourceGuard()
        result = guard.all_guards_clear(required_vram_bytes=1024 * 1024 * 1024)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_all_guards_clear_returns_true_when_all_safe(self):
        guard = ResourceGuard()
        with (
            patch.object(guard._thermal, "should_pause", return_value=False),
            patch.object(guard._power, "should_pause", return_value=False),
            patch.object(guard, "check_vram_available", return_value=True),
        ):
            can_train, reasons = guard.all_guards_clear(required_vram_bytes=512)
        assert can_train is True
        assert reasons == []

    def test_all_guards_clear_returns_false_on_thermal(self):
        guard = ResourceGuard()
        with (
            patch.object(guard._thermal, "should_pause", return_value=True),
            patch.object(guard._thermal, "get_gpu_temp_celsius", return_value=85.0),
            patch.object(guard._power, "should_pause", return_value=False),
            patch.object(guard, "check_vram_available", return_value=True),
        ):
            can_train, reasons = guard.all_guards_clear(required_vram_bytes=512)
        assert can_train is False
        assert len(reasons) >= 1

    def test_all_guards_clear_returns_false_on_battery(self):
        guard = ResourceGuard()
        with (
            patch.object(guard._thermal, "should_pause", return_value=False),
            patch.object(guard._power, "should_pause", return_value=True),
            patch.object(guard, "check_vram_available", return_value=True),
        ):
            can_train, reasons = guard.all_guards_clear(required_vram_bytes=512)
        assert can_train is False
        assert any("battery" in r.lower() for r in reasons)

    def test_all_guards_clear_returns_false_on_vram(self):
        guard = ResourceGuard()
        with (
            patch.object(guard._thermal, "should_pause", return_value=False),
            patch.object(guard._power, "should_pause", return_value=False),
            patch.object(guard, "check_vram_available", return_value=False),
        ):
            can_train, reasons = guard.all_guards_clear(required_vram_bytes=10 * 1024 ** 3)
        assert can_train is False
        assert any("vram" in r.lower() for r in reasons)

    def test_check_vram_available_true_when_pynvml_missing(self):
        """When pynvml is not installed, optimistically return True."""
        guard = ResourceGuard()
        result = guard.check_vram_available(required_bytes=1)
        assert isinstance(result, bool)

    def test_vram_reserve_bytes_constant(self):
        assert ResourceGuard.VRAM_RESERVE_BYTES == 512 * 1024 * 1024

    def test_multiple_blocking_reasons_collected(self):
        guard = ResourceGuard()
        with (
            patch.object(guard._thermal, "should_pause", return_value=True),
            patch.object(guard._thermal, "get_gpu_temp_celsius", return_value=90.0),
            patch.object(guard._power, "should_pause", return_value=True),
            patch.object(guard, "check_vram_available", return_value=False),
        ):
            can_train, reasons = guard.all_guards_clear(required_vram_bytes=512)
        assert can_train is False
        assert len(reasons) == 3


# ---------------------------------------------------------------------------
# GradientAccumulator
# ---------------------------------------------------------------------------

class TestGradientAccumulator:
    def test_8_microbatches_last_returns_true(self):
        accum = GradientAccumulator(accum_steps=8)
        results = [accum.record_microbatch() for _ in range(8)]
        # Only the 8th call should return True
        assert results[-1] is True
        assert all(r is False for r in results[:-1])

    def test_first_7_microbatches_return_false(self):
        accum = GradientAccumulator(accum_steps=8)
        for _ in range(7):
            assert accum.record_microbatch() is False

    def test_accumulated_count_increments(self):
        accum = GradientAccumulator(accum_steps=8)
        accum.record_microbatch()
        accum.record_microbatch()
        assert accum.accumulated == 2

    def test_optimizer_step_taken_resets_accumulated(self):
        accum = GradientAccumulator(accum_steps=4)
        for _ in range(4):
            accum.record_microbatch()
        accum.optimizer_step_taken()
        assert accum.accumulated == 0

    def test_optimizer_step_taken_increments_current_step(self):
        accum = GradientAccumulator(accum_steps=4)
        for _ in range(4):
            accum.record_microbatch()
        accum.optimizer_step_taken()
        assert accum.current_step == 1

    def test_idempotent_after_optimizer_step(self):
        """After optimizer_step_taken(), accumulated resets; next 4 micro-batches work."""
        accum = GradientAccumulator(accum_steps=4)
        for _ in range(4):
            accum.record_microbatch()
        accum.optimizer_step_taken()

        # Second round
        results = [accum.record_microbatch() for _ in range(4)]
        assert results[-1] is True
        assert all(r is False for r in results[:-1])
        accum.optimizer_step_taken()
        assert accum.current_step == 2
        assert accum.accumulated == 0

    def test_save_creates_file(self, tmp_path):
        accum = GradientAccumulator(accum_steps=8)
        accum.record_microbatch()
        accum.record_microbatch()
        path = tmp_path / "accum_state.json"
        accum.save_state(path)
        assert path.exists()

    def test_save_load_round_trip(self, tmp_path):
        accum = GradientAccumulator(accum_steps=8)
        for _ in range(3):
            accum.record_microbatch()
        accum_steps_before_save = accum.accumulated
        current_step_before_save = accum.current_step

        path = tmp_path / "accum_state.json"
        accum.save_state(path)

        # Restore into a fresh accumulator
        accum2 = GradientAccumulator(accum_steps=8)
        accum2.load_state(path)
        assert accum2.accumulated == accum_steps_before_save
        assert accum2.current_step == current_step_before_save

    def test_load_state_rejects_mismatched_accum_steps(self, tmp_path):
        accum = GradientAccumulator(accum_steps=4)
        path = tmp_path / "state.json"
        accum.save_state(path)

        accum2 = GradientAccumulator(accum_steps=8)
        with pytest.raises(ValueError):
            accum2.load_state(path)

    def test_save_state_json_contains_expected_keys(self, tmp_path):
        accum = GradientAccumulator(accum_steps=8)
        path = tmp_path / "state.json"
        accum.save_state(path)
        data = json.loads(path.read_text())
        assert "accum_steps" in data
        assert "accumulated" in data
        assert "current_step" in data

    def test_initial_accumulated_is_zero(self):
        accum = GradientAccumulator(accum_steps=4)
        assert accum.accumulated == 0

    def test_initial_current_step_is_zero(self):
        accum = GradientAccumulator(accum_steps=4)
        assert accum.current_step == 0

    def test_accum_steps_1_returns_true_immediately(self):
        accum = GradientAccumulator(accum_steps=1)
        assert accum.record_microbatch() is True

    def test_invalid_accum_steps_raises(self):
        with pytest.raises(ValueError):
            GradientAccumulator(accum_steps=0)

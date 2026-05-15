"""Unit tests for IdleTrainingDaemon GPU wiring — d20.

Covers:
1. test_idle_daemon_uses_simulated_runner_without_real_gpu
2. test_make_real_gpu_step_runner_calls_finetune
3. test_real_gpu_step_runner_returns_loss
4. test_idle_daemon_fsm_transitions_to_training
5. test_idle_daemon_checkpoint_evidence_emitted
6. test_idle_daemon_auto_selects_real_runner_when_gpu_set
7. test_idle_training_result_has_merkle_root
8. test_idle_daemon_pauses_on_resource_contention
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.idle.daemon import (
    DaemonState,
    IdleTrainConfig,
    IdleTrainingDaemon,
    _default_step_runner,
    make_real_gpu_step_runner,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path) -> IdleTrainConfig:
    return IdleTrainConfig(
        base_model="dummy/qwen-7b",
        pack_id="test-pack",
        checkpoint_dir=tmp_path / "checkpoints",
        grad_accum_steps=1,
        idle_threshold_seconds=1,
        required_vram_bytes=1024,
        poll_interval_seconds=0.0,
    )


def _always_idle_detector():
    """Return an IdleDetector mock that always reports idle."""
    det = MagicMock()
    det.is_idle.return_value = True
    return det


def _clear_guards_resource_guard():
    """Return a ResourceGuard mock that always reports all-clear."""
    guard = MagicMock()
    guard.all_guards_clear.return_value = (True, [])
    return guard


def _blocking_resource_guard(second_call_clear: bool = True):
    """Return a ResourceGuard mock that blocks on first call.

    If second_call_clear is True (default), the second call returns (True, []).
    """
    guard = MagicMock()
    guard.all_guards_clear.side_effect = [
        (False, ["Thermal pause: GPU temp 90.0°C exceeds 80.0°C"]),
        (True, []) if second_call_clear else (False, ["still blocked"]),
    ]
    return guard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ---------------------------------------------------------------------------
# 1. Simulated runner is selected by default (no DOJO_REAL_GPU)
# ---------------------------------------------------------------------------

class TestSimulatedRunner:
    def test_idle_daemon_uses_simulated_runner_without_real_gpu(self, tmp_path):
        """With DOJO_REAL_GPU unset/0, step result should have mode='simulated'."""
        config = _make_config(tmp_path)

        with patch.dict("os.environ", {"DOJO_REAL_GPU": "0"}):
            daemon = IdleTrainingDaemon(
                config,
                idle_detector=_always_idle_detector(),
                resource_guard=_clear_guards_resource_guard(),
            )

        result_dict = daemon._step_runner(0, 0)
        assert result_dict["mode"] == "simulated"

    def test_default_step_runner_returns_expected_keys(self):
        result = _default_step_runner(0, 0)
        assert "loss" in result
        assert "mode" in result
        assert result["mode"] == "simulated"

    def test_default_step_runner_loss_decreases_with_steps(self):
        loss_0 = _default_step_runner(0, 0)["loss"]
        loss_1 = _default_step_runner(1, 0)["loss"]
        assert loss_1 < loss_0


# ---------------------------------------------------------------------------
# 2 & 3. Real GPU step runner calls finetune() and returns loss
# ---------------------------------------------------------------------------

class TestRealGpuStepRunner:
    def _mock_finetune_result(self, final_loss: float = 0.5):
        return SimpleNamespace(
            final_loss=final_loss,
            adapter_path=Path("/tmp/adapter"),
            training_receipt="abc123",
            steps_completed=1,
            wall_time_seconds=0.1,
            backend_used="native",
        )

    def test_make_real_gpu_step_runner_calls_finetune(self, tmp_path):
        """DOJO_REAL_GPU=1: finetune() must be called exactly once."""
        config = _make_config(tmp_path)
        runner = make_real_gpu_step_runner(config)

        mock_result = self._mock_finetune_result()

        with (
            patch.dict("os.environ", {"DOJO_REAL_GPU": "1"}),
            patch("dojo.train.finetune.finetune", return_value=mock_result) as mock_ft,
            patch("dojo.hal.detector.detect") as mock_detect,
        ):
            mock_detect.return_value = MagicMock()
            runner(0, 0)

        mock_ft.assert_called_once()

    def test_real_gpu_step_runner_returns_loss(self, tmp_path):
        """DOJO_REAL_GPU=1: step result loss should equal finetune() final_loss."""
        config = _make_config(tmp_path)
        runner = make_real_gpu_step_runner(config)

        mock_result = self._mock_finetune_result(final_loss=0.314)

        with (
            patch.dict("os.environ", {"DOJO_REAL_GPU": "1"}),
            patch("dojo.train.finetune.finetune", return_value=mock_result),
            patch("dojo.hal.detector.detect") as mock_detect,
        ):
            mock_detect.return_value = MagicMock()
            result = runner(0, 0)

        assert result["loss"] == pytest.approx(0.314)
        assert result["mode"] == "real_gpu"

    def test_real_gpu_step_runner_falls_back_to_simulated_when_gpu_unset(
        self, tmp_path
    ):
        """When DOJO_REAL_GPU != '1', the real runner falls back to simulated."""
        config = _make_config(tmp_path)
        runner = make_real_gpu_step_runner(config)

        with patch.dict("os.environ", {"DOJO_REAL_GPU": "0"}):
            result = runner(0, 0)

        assert result["mode"] == "simulated"

    def test_real_gpu_step_runner_checkpoint_path_in_result(self, tmp_path):
        """DOJO_REAL_GPU=1: checkpoint_path should be non-empty in result."""
        config = _make_config(tmp_path)
        runner = make_real_gpu_step_runner(config)

        mock_result = self._mock_finetune_result()

        with (
            patch.dict("os.environ", {"DOJO_REAL_GPU": "1"}),
            patch("dojo.train.finetune.finetune", return_value=mock_result),
            patch("dojo.hal.detector.detect") as mock_detect,
        ):
            mock_detect.return_value = MagicMock()
            result = runner(0, 0)

        assert result["checkpoint_path"] != ""


# ---------------------------------------------------------------------------
# 4. FSM transitions to TRAINING
# ---------------------------------------------------------------------------

class TestFSMTransitions:
    def test_idle_daemon_fsm_transitions_to_training(self, tmp_path):
        """run() should drive the FSM through WAITING_FOR_IDLE → GUARDING → TRAINING."""
        config = _make_config(tmp_path)
        daemon = IdleTrainingDaemon(
            config,
            idle_detector=_always_idle_detector(),
            resource_guard=_clear_guards_resource_guard(),
        )

        result = daemon.run()

        # Check final state
        assert daemon.state == DaemonState.COMPLETED

        # Check FSM history includes TRAINING
        history_states = [to for (_, to) in daemon._fsm.history]
        assert DaemonState.TRAINING in history_states

    def test_idle_daemon_fsm_full_path(self, tmp_path):
        """FSM must visit all expected states in order."""
        config = _make_config(tmp_path)
        daemon = IdleTrainingDaemon(
            config,
            idle_detector=_always_idle_detector(),
            resource_guard=_clear_guards_resource_guard(),
        )

        daemon.run()

        history_states = [to for (_, to) in daemon._fsm.history]
        expected_in_order = [
            DaemonState.WAITING_FOR_IDLE,
            DaemonState.GUARDING,
            DaemonState.TRAINING,
            DaemonState.CHECKPOINTING,
            DaemonState.COMPLETED,
        ]
        # All expected states appear, in order
        idx = 0
        for expected in expected_in_order:
            while idx < len(history_states) and history_states[idx] != expected:
                idx += 1
            assert idx < len(history_states), (
                f"Expected state {expected} not found in history after position {idx}: "
                f"{history_states}"
            )
            idx += 1

    def test_idle_daemon_initial_state_is_idle(self, tmp_path):
        config = _make_config(tmp_path)
        daemon = IdleTrainingDaemon(
            config,
            idle_detector=_always_idle_detector(),
            resource_guard=_clear_guards_resource_guard(),
        )
        assert daemon.state == DaemonState.IDLE


# ---------------------------------------------------------------------------
# 5. Checkpoint evidence emitted
# ---------------------------------------------------------------------------

class TestCheckpointEvidence:
    def test_idle_daemon_checkpoint_evidence_emitted(self, tmp_path):
        """After run(), a TRAINING_STEP event with 'checkpoint_saved' must be in evidence."""
        config = _make_config(tmp_path)
        daemon = IdleTrainingDaemon(
            config,
            idle_detector=_always_idle_detector(),
            resource_guard=_clear_guards_resource_guard(),
        )

        daemon.run()

        events = EvidenceLogger.get_events()
        step_events = [
            e for e in events
            if isinstance(e.get("type"), str)
            and EvidenceType.TRAINING_STEP.value in e.get("type", "")
        ]
        assert len(step_events) >= 1
        # The event data must contain 'checkpoint_saved'
        for ev in step_events:
            data = ev.get("data", {})
            assert "checkpoint_saved" in data, (
                f"TRAINING_STEP event missing 'checkpoint_saved' key: {ev}"
            )

    def test_training_checkpoint_event_emitted(self, tmp_path):
        """TRAINING_CHECKPOINT evidence must also be emitted."""
        config = _make_config(tmp_path)
        daemon = IdleTrainingDaemon(
            config,
            idle_detector=_always_idle_detector(),
            resource_guard=_clear_guards_resource_guard(),
        )

        daemon.run()

        events = EvidenceLogger.get_events()
        ckpt_events = [
            e for e in events
            if isinstance(e.get("type"), str)
            and EvidenceType.TRAINING_CHECKPOINT.value in e.get("type", "")
        ]
        assert len(ckpt_events) >= 1


# ---------------------------------------------------------------------------
# 6. Auto-selects real runner when DOJO_REAL_GPU=1
# ---------------------------------------------------------------------------

class TestAutoRunnerSelection:
    def test_idle_daemon_auto_selects_real_runner_when_gpu_set(self, tmp_path):
        """With DOJO_REAL_GPU=1 and no step_runner argument, should use real runner."""
        config = _make_config(tmp_path)

        with patch.dict("os.environ", {"DOJO_REAL_GPU": "1"}):
            daemon = IdleTrainingDaemon(
                config,
                idle_detector=_always_idle_detector(),
                resource_guard=_clear_guards_resource_guard(),
            )

        # The step runner should NOT be the default simulated one when GPU is set.
        # Calling it with DOJO_REAL_GPU=0 in env should fall back, but in DOJO_REAL_GPU=1
        # context it should call finetune. We just check it's a different callable than
        # the plain _default_step_runner.
        assert daemon._step_runner is not _default_step_runner

    def test_idle_daemon_auto_selects_simulated_runner_when_gpu_not_set(self, tmp_path):
        """With DOJO_REAL_GPU=0 and no step_runner argument, should use simulated runner."""
        config = _make_config(tmp_path)

        with patch.dict("os.environ", {"DOJO_REAL_GPU": "0"}):
            daemon = IdleTrainingDaemon(
                config,
                idle_detector=_always_idle_detector(),
                resource_guard=_clear_guards_resource_guard(),
            )

        assert daemon._step_runner is _default_step_runner

    def test_idle_daemon_accepts_explicit_step_runner(self, tmp_path):
        """Explicit step_runner overrides auto-selection."""
        config = _make_config(tmp_path)
        custom_runner = MagicMock(return_value={
            "current_step": 0,
            "accumulated": 0,
            "loss": 0.1,
            "mode": "custom",
            "checkpoint_path": "",
        })

        with patch.dict("os.environ", {"DOJO_REAL_GPU": "1"}):
            daemon = IdleTrainingDaemon(
                config,
                step_runner=custom_runner,
                idle_detector=_always_idle_detector(),
                resource_guard=_clear_guards_resource_guard(),
            )

        assert daemon._step_runner is custom_runner


# ---------------------------------------------------------------------------
# 7. IdleTrainResult has non-empty merkle root
# ---------------------------------------------------------------------------

class TestIdleTrainResult:
    def test_idle_training_result_has_merkle_root(self, tmp_path):
        """IdleTrainResult.evidence_merkle_root must be a non-empty string."""
        config = _make_config(tmp_path)
        daemon = IdleTrainingDaemon(
            config,
            idle_detector=_always_idle_detector(),
            resource_guard=_clear_guards_resource_guard(),
        )

        result = daemon.run()

        assert isinstance(result.evidence_merkle_root, str)
        assert len(result.evidence_merkle_root) > 0

    def test_idle_training_result_fields(self, tmp_path):
        """IdleTrainResult has all expected fields."""
        config = _make_config(tmp_path)
        daemon = IdleTrainingDaemon(
            config,
            idle_detector=_always_idle_detector(),
            resource_guard=_clear_guards_resource_guard(),
        )

        result = daemon.run()

        assert hasattr(result, "steps_completed")
        assert hasattr(result, "final_loss")
        assert hasattr(result, "evidence_merkle_root")
        assert hasattr(result, "checkpoint_path")
        assert isinstance(result.final_loss, float)

    def test_idle_training_result_final_loss_is_finite(self, tmp_path):
        """final_loss should be a finite float."""
        import math

        config = _make_config(tmp_path)
        daemon = IdleTrainingDaemon(
            config,
            idle_detector=_always_idle_detector(),
            resource_guard=_clear_guards_resource_guard(),
        )

        result = daemon.run()

        assert not math.isnan(result.final_loss)
        assert not math.isinf(result.final_loss)


# ---------------------------------------------------------------------------
# 8. Paused state on resource contention
# ---------------------------------------------------------------------------

class TestResourceContentionPause:
    def test_idle_daemon_pauses_on_resource_contention(self, tmp_path):
        """When ResourceGuard blocks first call, daemon should enter PAUSED state.

        The FSM history must include PAUSED before eventually completing.
        """
        config = _make_config(tmp_path)
        # First call blocks, second call clears
        blocking_guard = _blocking_resource_guard(second_call_clear=True)

        daemon = IdleTrainingDaemon(
            config,
            idle_detector=_always_idle_detector(),
            resource_guard=blocking_guard,
        )

        result = daemon.run()

        history_states = [to for (_, to) in daemon._fsm.history]
        assert DaemonState.PAUSED in history_states, (
            f"Expected PAUSED in FSM history, got: {history_states}"
        )
        # Should eventually complete
        assert daemon.state == DaemonState.COMPLETED

    def test_idle_daemon_paused_evidence_emitted(self, tmp_path):
        """When ResourceGuard blocks, IDLE_THERMAL_PAUSED evidence must be emitted."""
        config = _make_config(tmp_path)
        blocking_guard = _blocking_resource_guard(second_call_clear=True)

        daemon = IdleTrainingDaemon(
            config,
            idle_detector=_always_idle_detector(),
            resource_guard=blocking_guard,
        )

        daemon.run()

        events = EvidenceLogger.get_events()
        paused_events = [
            e for e in events
            if isinstance(e.get("type"), str)
            and EvidenceType.IDLE_THERMAL_PAUSED.value in e.get("type", "")
        ]
        assert len(paused_events) >= 1

    def test_idle_daemon_fails_when_guards_never_clear(self, tmp_path):
        """When resource guards block on both calls, daemon must raise RuntimeError."""
        config = _make_config(tmp_path)
        # Both calls block
        always_blocking_guard = _blocking_resource_guard(second_call_clear=False)

        daemon = IdleTrainingDaemon(
            config,
            idle_detector=_always_idle_detector(),
            resource_guard=always_blocking_guard,
        )

        with pytest.raises(RuntimeError, match="Resource guards still blocking"):
            daemon.run()

        assert daemon.state == DaemonState.FAILED

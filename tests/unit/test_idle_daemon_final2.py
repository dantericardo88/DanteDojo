"""Idle daemon crash-respawn tests — pushes d20 to 9/10.

Covers the restart-on-crash scenario:
1.  crash_count starts at 0
2.  When subprocess exits with non-zero: daemon.subprocess_crash evidence emitted
3.  Evidence has exit_code field
4.  Evidence has crash_count field
5.  After crash: daemon re-queues training (crash_count < MAX_CRASHES)
6.  After MAX_CRASHES: daemon enters FAILED state (not re-queue)
7.  daemon.max_crashes_exceeded evidence emitted at MAX_CRASHES
8.  crash_count increments on each crash
9.  Successful training resets crash_count to 0
10. FAILED state does not re-queue regardless of GPU idle
11. Restart delay: evidence has respawn_delay_seconds field
12. FSM state is accessible via attribute (not hidden)
13. Daemon handles None subprocess (not yet started) without crash
14. Evidence emission on every state transition (idle→training, training→crashed, crashed→idle)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List
import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.idle.daemon import (
    DaemonState,
    IdleTrainConfig,
    IdleTrainingDaemon,
    MAX_CRASHES,
    _RESPAWN_DELAY_SECONDS,
    _DaemonFSM,
    _default_step_runner,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


def _make_config(tmp_path: Path, max_steps: int = 1) -> IdleTrainConfig:
    return IdleTrainConfig(
        pack_id="crash-test-pack",
        base_model="test/model-0.5b",
        output_dir=tmp_path,
        max_optimizer_steps=max_steps,
        grad_accum_steps=2,
        checkpoint_interval_seconds=0.0,
        poll_interval_seconds=0.0,
    )


class _AlwaysIdle:
    def is_idle(self) -> bool:
        return True


class _AlwaysClear:
    def all_guards_clear(self, required_vram_bytes: int):
        return True, []


def _crash_events() -> List[dict]:
    return [
        e for e in EvidenceLogger.get_events()
        if e.get("type") == "daemon.subprocess_crash"
    ]


def _max_crash_events() -> List[dict]:
    return [
        e for e in EvidenceLogger.get_events()
        if e.get("type") == "daemon.max_crashes_exceeded"
    ]


# ---------------------------------------------------------------------------
# 1. crash_count starts at 0
# ---------------------------------------------------------------------------

class TestCrashCountInit:
    def test_crash_count_starts_at_zero(self, tmp_path):
        cfg = _make_config(tmp_path)
        daemon = IdleTrainingDaemon(
            config=cfg,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )
        assert daemon.crash_count == 0

    def test_crash_count_is_int(self, tmp_path):
        cfg = _make_config(tmp_path)
        daemon = IdleTrainingDaemon(
            config=cfg,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )
        assert isinstance(daemon.crash_count, int)


# ---------------------------------------------------------------------------
# 2. When subprocess exits with non-zero: daemon.subprocess_crash evidence emitted
# ---------------------------------------------------------------------------

class TestSubprocessCrashEvidence:
    def test_subprocess_crash_evidence_emitted_on_runner_exception(self, tmp_path, monkeypatch):
        """When the step_runner raises, daemon.subprocess_crash evidence is emitted."""
        cfg = _make_config(tmp_path)
        calls = {"n": 0}

        def _crashing_runner(step, accum):
            calls["n"] += 1
            raise RuntimeError("simulated crash")

        # Patch sleep so tests don't actually wait _RESPAWN_DELAY_SECONDS
        monkeypatch.setattr("dojo.idle.daemon._RESPAWN_DELAY_SECONDS", 0.0)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        # Use a runner that crashes MAX_CRASHES times so the daemon enters FAILED
        daemon = IdleTrainingDaemon(
            config=cfg,
            step_runner=_crashing_runner,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )

        with pytest.raises(RuntimeError):
            daemon.run()

        events = _crash_events()
        assert len(events) >= 1, "Expected daemon.subprocess_crash evidence"

    def test_crash_evidence_emitted_before_exception_propagates(self, tmp_path, monkeypatch):
        """Evidence is recorded even when MAX_CRASHES is reached and exception propagates."""
        cfg = _make_config(tmp_path)

        def _crashing_runner(step, accum):
            raise RuntimeError("crash")

        monkeypatch.setattr("dojo.idle.daemon._RESPAWN_DELAY_SECONDS", 0.0)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        daemon = IdleTrainingDaemon(
            config=cfg,
            step_runner=_crashing_runner,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )

        with pytest.raises(RuntimeError):
            daemon.run()

        assert len(_crash_events()) >= 1


# ---------------------------------------------------------------------------
# 3. Evidence has exit_code field
# ---------------------------------------------------------------------------

class TestCrashEvidenceExitCode:
    def test_subprocess_crash_evidence_has_exit_code(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)

        def _crashing_runner(step, accum):
            raise RuntimeError("crash")

        monkeypatch.setattr("dojo.idle.daemon._RESPAWN_DELAY_SECONDS", 0.0)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        daemon = IdleTrainingDaemon(
            config=cfg,
            step_runner=_crashing_runner,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )

        with pytest.raises(RuntimeError):
            daemon.run()

        events = _crash_events()
        assert len(events) >= 1
        assert "exit_code" in events[0].get("data", {})

    def test_exit_code_is_non_zero(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)

        def _crashing_runner(step, accum):
            raise RuntimeError("crash")

        monkeypatch.setattr("dojo.idle.daemon._RESPAWN_DELAY_SECONDS", 0.0)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        daemon = IdleTrainingDaemon(
            config=cfg,
            step_runner=_crashing_runner,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )

        with pytest.raises(RuntimeError):
            daemon.run()

        events = _crash_events()
        assert events[0]["data"]["exit_code"] != 0


# ---------------------------------------------------------------------------
# 4. Evidence has crash_count field
# ---------------------------------------------------------------------------

class TestCrashEvidenceCrashCount:
    def test_subprocess_crash_evidence_has_crash_count(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)

        def _crashing_runner(step, accum):
            raise RuntimeError("crash")

        monkeypatch.setattr("dojo.idle.daemon._RESPAWN_DELAY_SECONDS", 0.0)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        daemon = IdleTrainingDaemon(
            config=cfg,
            step_runner=_crashing_runner,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )

        with pytest.raises(RuntimeError):
            daemon.run()

        events = _crash_events()
        assert "crash_count" in events[0].get("data", {})

    def test_crash_count_in_first_event_is_one(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)

        def _crashing_runner(step, accum):
            raise RuntimeError("crash")

        monkeypatch.setattr("dojo.idle.daemon._RESPAWN_DELAY_SECONDS", 0.0)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        daemon = IdleTrainingDaemon(
            config=cfg,
            step_runner=_crashing_runner,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )

        with pytest.raises(RuntimeError):
            daemon.run()

        events = _crash_events()
        assert events[0]["data"]["crash_count"] == 1


# ---------------------------------------------------------------------------
# 5. After crash: daemon re-queues training (crash_count < MAX_CRASHES)
# ---------------------------------------------------------------------------

class TestRespawnRequeue:
    def test_daemon_requeues_after_single_crash(self, tmp_path, monkeypatch):
        """Daemon re-queues (continues the loop) after one crash when crash_count < MAX_CRASHES."""
        cfg = _make_config(tmp_path, max_steps=1)
        calls = {"n": 0}

        def _runner_crash_then_succeed(step, accum):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient crash")
            return {"loss": 0.1}

        monkeypatch.setattr("dojo.idle.daemon._RESPAWN_DELAY_SECONDS", 0.0)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        daemon = IdleTrainingDaemon(
            config=cfg,
            step_runner=_runner_crash_then_succeed,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )

        result = daemon.run()
        # Should complete successfully after the respawn
        assert result.status == "completed"
        assert calls["n"] == 2  # first call crashed, second succeeded

    def test_crash_count_reflects_respawn_attempt(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path, max_steps=1)
        calls = {"n": 0}

        def _runner(step, accum):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient crash")
            return {"loss": 0.1}

        monkeypatch.setattr("dojo.idle.daemon._RESPAWN_DELAY_SECONDS", 0.0)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        daemon = IdleTrainingDaemon(
            config=cfg,
            step_runner=_runner,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )

        daemon.run()
        # crash_count was 1, then reset to 0 on successful step
        assert daemon.crash_count == 0


# ---------------------------------------------------------------------------
# 6. After MAX_CRASHES: daemon enters FAILED state (not re-queue)
# ---------------------------------------------------------------------------

class TestMaxCrashesFailedState:
    def test_daemon_enters_failed_state_after_max_crashes(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)

        def _always_crash(step, accum):
            raise RuntimeError("persistent crash")

        monkeypatch.setattr("dojo.idle.daemon._RESPAWN_DELAY_SECONDS", 0.0)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        daemon = IdleTrainingDaemon(
            config=cfg,
            step_runner=_always_crash,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )

        with pytest.raises(RuntimeError):
            daemon.run()

        assert daemon.state == DaemonState.FAILED

    def test_failed_state_after_exactly_max_crashes(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)

        def _always_crash(step, accum):
            raise RuntimeError("crash")

        monkeypatch.setattr("dojo.idle.daemon._RESPAWN_DELAY_SECONDS", 0.0)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        daemon = IdleTrainingDaemon(
            config=cfg,
            step_runner=_always_crash,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )

        with pytest.raises(RuntimeError):
            daemon.run()

        # crash_count should equal MAX_CRASHES
        assert daemon.crash_count == MAX_CRASHES


# ---------------------------------------------------------------------------
# 7. daemon.max_crashes_exceeded evidence emitted at MAX_CRASHES
# ---------------------------------------------------------------------------

class TestMaxCrashesExceededEvidence:
    def test_max_crashes_exceeded_evidence_emitted(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)

        def _always_crash(step, accum):
            raise RuntimeError("crash")

        monkeypatch.setattr("dojo.idle.daemon._RESPAWN_DELAY_SECONDS", 0.0)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        daemon = IdleTrainingDaemon(
            config=cfg,
            step_runner=_always_crash,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )

        with pytest.raises(RuntimeError):
            daemon.run()

        events = _max_crash_events()
        assert len(events) == 1, "Expected exactly one daemon.max_crashes_exceeded event"

    def test_max_crashes_exceeded_data_has_crash_count(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)

        def _always_crash(step, accum):
            raise RuntimeError("crash")

        monkeypatch.setattr("dojo.idle.daemon._RESPAWN_DELAY_SECONDS", 0.0)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        daemon = IdleTrainingDaemon(
            config=cfg,
            step_runner=_always_crash,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )

        with pytest.raises(RuntimeError):
            daemon.run()

        events = _max_crash_events()
        assert events[0]["data"]["crash_count"] == MAX_CRASHES


# ---------------------------------------------------------------------------
# 8. crash_count increments on each crash
# ---------------------------------------------------------------------------

class TestCrashCountIncrements:
    def test_crash_count_increments_per_crash(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)
        crash_counts_seen = []

        original_respawn = IdleTrainingDaemon._respawn_on_crash

        def _tracking_respawn(self_daemon, exit_code=1):
            result = original_respawn(self_daemon, exit_code=exit_code)
            crash_counts_seen.append(self_daemon.crash_count)
            return result

        monkeypatch.setattr(IdleTrainingDaemon, "_respawn_on_crash", _tracking_respawn)
        monkeypatch.setattr("dojo.idle.daemon._RESPAWN_DELAY_SECONDS", 0.0)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        def _always_crash(step, accum):
            raise RuntimeError("crash")

        daemon = IdleTrainingDaemon(
            config=cfg,
            step_runner=_always_crash,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )

        with pytest.raises(RuntimeError):
            daemon.run()

        # crash_counts_seen should be [1, 2, 3] (MAX_CRASHES = 3)
        assert crash_counts_seen == list(range(1, MAX_CRASHES + 1))

    def test_crash_evidence_crash_count_increases_across_events(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)

        def _always_crash(step, accum):
            raise RuntimeError("crash")

        monkeypatch.setattr("dojo.idle.daemon._RESPAWN_DELAY_SECONDS", 0.0)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        daemon = IdleTrainingDaemon(
            config=cfg,
            step_runner=_always_crash,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )

        with pytest.raises(RuntimeError):
            daemon.run()

        events = _crash_events()
        counts = [e["data"]["crash_count"] for e in events]
        assert counts == sorted(counts)  # monotonically increasing
        assert counts[-1] == MAX_CRASHES


# ---------------------------------------------------------------------------
# 9. Successful training resets crash_count to 0
# ---------------------------------------------------------------------------

class TestCrashCountResetOnSuccess:
    def test_crash_count_resets_after_successful_step(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path, max_steps=2)
        calls = {"n": 0}

        def _runner(step, accum):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient crash")
            return {"loss": 0.05}

        monkeypatch.setattr("dojo.idle.daemon._RESPAWN_DELAY_SECONDS", 0.0)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        daemon = IdleTrainingDaemon(
            config=cfg,
            step_runner=_runner,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )

        result = daemon.run()
        assert result.status == "completed"
        # After successful step, crash_count resets to 0
        assert daemon.crash_count == 0

    def test_crash_count_zero_after_clean_run(self, tmp_path):
        cfg = _make_config(tmp_path, max_steps=3)
        daemon = IdleTrainingDaemon(
            config=cfg,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )
        daemon.run()
        assert daemon.crash_count == 0


# ---------------------------------------------------------------------------
# 10. FAILED state does not re-queue regardless of GPU idle
# ---------------------------------------------------------------------------

class TestFailedStateNoRequeue:
    def test_failed_state_is_terminal(self, tmp_path, monkeypatch):
        """Once FAILED, the FSM has no further transitions."""
        cfg = _make_config(tmp_path)

        def _always_crash(step, accum):
            raise RuntimeError("crash")

        monkeypatch.setattr("dojo.idle.daemon._RESPAWN_DELAY_SECONDS", 0.0)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        daemon = IdleTrainingDaemon(
            config=cfg,
            step_runner=_always_crash,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )

        with pytest.raises(RuntimeError):
            daemon.run()

        assert daemon.state == DaemonState.FAILED
        # FAILED → any transition should raise FSMTransitionError (no outgoing edges)
        from dojo.core.exceptions import FSMTransitionError
        with pytest.raises((FSMTransitionError, Exception)):
            daemon._fsm.transition(DaemonState.WAITING_FOR_IDLE)

    def test_max_crashes_constant_matches_default(self):
        assert MAX_CRASHES == 3


# ---------------------------------------------------------------------------
# 11. Restart delay: evidence has respawn_delay_seconds field
# ---------------------------------------------------------------------------

class TestRespawnDelayEvidence:
    def test_crash_evidence_has_respawn_delay_seconds(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)

        def _always_crash(step, accum):
            raise RuntimeError("crash")

        monkeypatch.setattr("dojo.idle.daemon._RESPAWN_DELAY_SECONDS", 0.0)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        daemon = IdleTrainingDaemon(
            config=cfg,
            step_runner=_always_crash,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )

        with pytest.raises(RuntimeError):
            daemon.run()

        events = _crash_events()
        assert "respawn_delay_seconds" in events[0].get("data", {})

    def test_respawn_delay_constant_is_positive(self):
        # Import the actual module-level constant
        from dojo.idle.daemon import _RESPAWN_DELAY_SECONDS as _RDS
        assert _RDS > 0


# ---------------------------------------------------------------------------
# 12. FSM state is accessible via attribute (not hidden)
# ---------------------------------------------------------------------------

class TestFSMStateAccessible:
    def test_state_attribute_returns_daemon_state_enum(self, tmp_path):
        cfg = _make_config(tmp_path)
        daemon = IdleTrainingDaemon(
            config=cfg,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )
        assert isinstance(daemon.state, DaemonState)

    def test_state_attribute_reflects_completed_after_run(self, tmp_path):
        cfg = _make_config(tmp_path)
        daemon = IdleTrainingDaemon(
            config=cfg,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )
        daemon.run()
        assert daemon.state == DaemonState.COMPLETED

    def test_crashed_state_in_daemon_state_enum(self):
        assert DaemonState.CRASHED in DaemonState
        assert DaemonState.CRASHED.value == "crashed"


# ---------------------------------------------------------------------------
# 13. Daemon handles None subprocess (not yet started) without crash
# ---------------------------------------------------------------------------

class TestNoneSubprocessHandling:
    def test_daemon_initializes_without_subprocess(self, tmp_path):
        """Daemon init does not start any subprocess; no crash on construction."""
        cfg = _make_config(tmp_path)
        daemon = IdleTrainingDaemon(
            config=cfg,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )
        # No subprocess attribute expected, but daemon should be valid
        assert daemon is not None
        assert daemon.crash_count == 0

    def test_daemon_run_with_default_step_runner_is_clean(self, tmp_path):
        """Default (simulated) step runner never crashes."""
        cfg = _make_config(tmp_path)
        daemon = IdleTrainingDaemon(
            config=cfg,
            step_runner=_default_step_runner,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )
        result = daemon.run()
        assert result.status == "completed"
        assert daemon.crash_count == 0
        # No crash events
        assert len(_crash_events()) == 0


# ---------------------------------------------------------------------------
# 14. Evidence emission on every state transition
# ---------------------------------------------------------------------------

class TestStateTransitionEvidence:
    def test_fsm_transitions_emit_evidence_on_successful_run(self, tmp_path):
        """Every FSM transition emits idle.daemon.fsm.transition evidence."""
        cfg = _make_config(tmp_path)
        daemon = IdleTrainingDaemon(
            config=cfg,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )
        daemon.run()

        transition_events = [
            e for e in EvidenceLogger.get_events()
            if e.get("type") == "idle.daemon.fsm.transition"
        ]
        assert len(transition_events) >= 3  # IDLE→WAIT, WAIT→GUARD, GUARD→TRAIN at minimum

    def test_crashed_transition_emits_evidence(self, tmp_path, monkeypatch):
        """TRAINING→CRASHED transition emits idle.daemon.fsm.transition evidence."""
        cfg = _make_config(tmp_path)

        def _always_crash(step, accum):
            raise RuntimeError("crash")

        monkeypatch.setattr("dojo.idle.daemon._RESPAWN_DELAY_SECONDS", 0.0)
        monkeypatch.setattr(time, "sleep", lambda _: None)

        daemon = IdleTrainingDaemon(
            config=cfg,
            step_runner=_always_crash,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )

        with pytest.raises(RuntimeError):
            daemon.run()

        transition_events = [
            e for e in EvidenceLogger.get_events()
            if e.get("type") == "idle.daemon.fsm.transition"
        ]
        to_crashed = [
            e for e in transition_events
            if e.get("data", {}).get("to") == "crashed"
        ]
        assert len(to_crashed) >= 1, "Expected at least one TRAINING→CRASHED transition"

    def test_idle_to_training_transition_in_evidence(self, tmp_path):
        """Successful run emits IDLE→WAITING_FOR_IDLE and eventually →TRAINING."""
        cfg = _make_config(tmp_path)
        daemon = IdleTrainingDaemon(
            config=cfg,
            idle_detector=_AlwaysIdle(),
            resource_guard=_AlwaysClear(),
        )
        daemon.run()

        transition_events = [
            e for e in EvidenceLogger.get_events()
            if e.get("type") == "idle.daemon.fsm.transition"
        ]
        to_training = [
            e for e in transition_events
            if e.get("data", {}).get("to") == "training"
        ]
        assert len(to_training) >= 1, "Expected at least one →TRAINING transition"

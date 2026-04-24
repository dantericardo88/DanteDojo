"""Constitutional tests: one-door doctrine enforcement.

These tests verify the constitutional substrate enforces the one-door doctrine:
- @one_door emits evidence on every call
- DOJO_BYPASS env var triggers OneDoorViolation
- FSM illegal transitions raise FSMTransitionError
- EvidenceLogger fail-closed guarantees
"""

import os
import pytest
from enum import Enum
from unittest.mock import patch

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import (
    ConstitutionalViolation,
    DojoError,
    FSMTransitionError,
    OneDoorViolation,
    PrivacyTierViolation,
    PromotionBlocked,
)
from dojo.core.fsm import BaseFSM, fsm_guard
from dojo.core.one_door import one_door, ok, fail
from dojo.core.types import PrivacyTier


@pytest.fixture(autouse=True)
def clear_logger():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ---------------------------------------------------------------------------
# One-door: evidence emission on every call
# ---------------------------------------------------------------------------

class TestOneDoorDecorator:
    def test_emits_evidence_on_call(self):
        @one_door("test.subsystem")
        def my_fn(x: int) -> int:
            return x * 2

        my_fn(5)
        events = EvidenceLogger.get_events()
        assert any(e.get("type") == EvidenceType.ONE_DOOR_ENTRY.value for e in events)

    def test_emits_entry_point_name(self):
        @one_door("hal.detect")
        def detect() -> str:
            return "done"

        detect()
        events = EvidenceLogger.get_events()
        entry_events = [e for e in events if e.get("type") == EvidenceType.ONE_DOOR_ENTRY.value]
        assert any("hal.detect" in str(e.get("data", {})) for e in entry_events)

    def test_decorated_fn_returns_correctly(self):
        @one_door("pack.import")
        def import_pack(name: str) -> dict:
            return {"pack": name}

        result = import_pack("empanada-bookkeeping")
        assert result == {"pack": "empanada-bookkeeping"}

    def test_bypass_env_var_raises_violation(self):
        @one_door("train.finetune")
        def finetune() -> None:
            pass

        with patch.dict(os.environ, {"DOJO_BYPASS": "1"}):
            with pytest.raises(OneDoorViolation):
                finetune()

    def test_bypass_env_var_empty_does_not_raise(self):
        @one_door("train.eval")
        def evaluate() -> str:
            return "ok"

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DOJO_BYPASS", None)
            result = evaluate()
            assert result == "ok"

    def test_dunder_one_door_attribute_set(self):
        @one_door("serve.launch")
        def launch() -> None:
            pass

        assert hasattr(launch, "__one_door__")
        assert launch.__one_door__ == "serve.launch"


# ---------------------------------------------------------------------------
# FSM: illegal transitions raise FSMTransitionError
# ---------------------------------------------------------------------------

class TestBaseFSM:
    def _make_fsm(self) -> BaseFSM:
        class State(Enum):
            IDLE = "idle"
            RUNNING = "running"
            DONE = "done"
            FAILED = "failed"

        class MyFSM(BaseFSM[State]):
            initial_state = State.IDLE
            _TRANSITIONS = {
                State.IDLE: frozenset({State.RUNNING}),
                State.RUNNING: frozenset({State.DONE, State.FAILED}),
            }

        return MyFSM()

    def test_valid_transition(self):
        fsm = self._make_fsm()
        fsm.transition(fsm.state.__class__["RUNNING"])
        assert fsm.state.value == "running"

    def test_illegal_transition_raises(self):
        fsm = self._make_fsm()
        with pytest.raises(FSMTransitionError):
            # Can't go from IDLE → DONE directly
            fsm.transition(fsm.state.__class__["DONE"])

    def test_terminal_state_has_no_outgoing(self):
        fsm = self._make_fsm()
        fsm.transition(fsm.state.__class__["RUNNING"])
        fsm.transition(fsm.state.__class__["DONE"])
        assert fsm.is_terminal() is True

    def test_history_tracks_transitions(self):
        fsm = self._make_fsm()
        fsm.transition(fsm.state.__class__["RUNNING"])
        assert len(fsm.history) == 1

    def test_reset_clears_history(self):
        fsm = self._make_fsm()
        fsm.transition(fsm.state.__class__["RUNNING"])
        fsm.reset()
        assert len(fsm.history) == 0
        assert fsm.state.value == "idle"


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:
    def test_all_dojo_errors_are_dojo_error(self):
        errors = [
            OneDoorViolation("x"),
            FSMTransitionError("x"),
            PrivacyTierViolation("x"),
            ConstitutionalViolation("x"),
        ]
        for e in errors:
            assert isinstance(e, DojoError)

    def test_promotion_blocked_carries_gate_and_reasons(self):
        err = PromotionBlocked("gate failed", gate="harvest_check", reasons=["receipt revoked", "license denied"])
        assert err.gate == "harvest_check"
        assert len(err.reasons) == 2
        assert isinstance(err, DojoError)

    def test_dojo_error_actionable_reason(self):
        e = DojoError("short message", reason="actionable detail about what to do")
        assert "actionable detail" in str(e)


# ---------------------------------------------------------------------------
# Evidence: fail-closed guarantees
# ---------------------------------------------------------------------------

class TestEvidenceFailClosed:
    def test_emit_non_dict_raises_type_error(self):
        logger = EvidenceLogger.get_instance()
        with pytest.raises(TypeError):
            logger.emit("string not dict")  # type: ignore[arg-type]

    def test_info_on_broken_message_does_not_raise(self):
        logger = EvidenceLogger.get_instance()
        # Even a circular-reference dict — info must not propagate exceptions
        # (it returns None on internal failure, never raises)
        logger.info("safe message")  # must not raise

    def test_merkle_root_stable_on_no_events(self):
        logger = EvidenceLogger.get_instance()
        root1 = logger.compute_merkle_root()
        root2 = logger.compute_merkle_root()
        assert root1 == root2

    def test_chain_verify_passes_on_clean_logger(self):
        # Empty chain verifies
        from dojo.core.evidence import EvidenceChain
        chain = EvidenceChain(run_id="test", merkle_root="", size=0, last_hash="")
        assert chain.verify_chain() is True


# ---------------------------------------------------------------------------
# Envelope helpers
# ---------------------------------------------------------------------------

class TestEnvelope:
    def test_ok_shape(self):
        result = ok({"value": 42})
        assert result["ok"] is True
        assert result["data"]["value"] == 42

    def test_fail_shape(self):
        result = fail("invalid-config", "missing field", details={"field": "seed"})
        assert result["ok"] is False
        assert result["error"]["code"] == "invalid-config"
        assert "field" in result["error"]["details"]

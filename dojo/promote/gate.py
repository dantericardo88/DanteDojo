"""DanteDojo promotion gate — Three-way promotion FSM.

Single one-door entry: promote(). Runs HARVEST_CHECK → DOJO_CHECK →
FORGE_CHECK → PASSPORT_BUILD → SEALED.

Any gate failure transitions to BLOCKED and raises PromotionBlocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import PromotionBlocked
from dojo.core.fsm import BaseFSM
from dojo.core.one_door import one_door
from dojo.core.types import ConstitutionalPassport, ModelArtifact


# ---------------------------------------------------------------------------
# FSM state definitions
# ---------------------------------------------------------------------------

class GateState(Enum):
    INIT = "init"
    HARVEST_CHECK = "harvest_check"
    DOJO_CHECK = "dojo_check"
    FORGE_CHECK = "forge_check"
    PASSPORT_BUILD = "passport_build"
    SEALED = "sealed"
    BLOCKED = "blocked"


# ---------------------------------------------------------------------------
# Result type returned by each gate check
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateCheckResult:
    """Result of a single gate check."""

    gate: str
    passed: bool
    reasons: list[str]
    receipt: bytes


# ---------------------------------------------------------------------------
# Promotion FSM
# ---------------------------------------------------------------------------

class _PromotionFSM(BaseFSM[GateState]):
    """Internal FSM that drives the three-way promotion gate."""

    initial_state = GateState.INIT

    _TRANSITIONS: dict[GateState, frozenset[GateState]] = {  # type: ignore[assignment]
        GateState.INIT: frozenset({GateState.HARVEST_CHECK}),
        GateState.HARVEST_CHECK: frozenset({GateState.DOJO_CHECK, GateState.BLOCKED}),
        GateState.DOJO_CHECK: frozenset({GateState.FORGE_CHECK, GateState.BLOCKED}),
        GateState.FORGE_CHECK: frozenset({GateState.PASSPORT_BUILD, GateState.BLOCKED}),
        GateState.PASSPORT_BUILD: frozenset({GateState.SEALED, GateState.BLOCKED}),
        GateState.SEALED: frozenset(),
        GateState.BLOCKED: frozenset(),
    }


# ---------------------------------------------------------------------------
# One-door entry point
# ---------------------------------------------------------------------------

@one_door("promote.gate")
def promote(
    candidate: ModelArtifact,
    harvest_receipt_id: str,
    eval_receipt_merkle: str,
    *,
    harvest_client: Optional[Any] = None,
    forge_client: Optional[Any] = None,
    eval_passing: bool = True,
    baseline_f1: float = 0.70,
    candidate_f1: float = 0.80,
) -> ConstitutionalPassport:
    """Run the three-way gate: HARVEST_CHECK → DOJO_CHECK → FORGE_CHECK → PASSPORT_BUILD → SEALED.

    All three must return passed=True.
    ANY failure → transition to BLOCKED → raise PromotionBlocked.

    Emits:
    - PROMOTE_HARVEST_GREEN or PROMOTE_HARVEST_BLOCKED
    - PROMOTE_DOJO_GREEN or PROMOTE_DOJO_BLOCKED
    - PROMOTE_FORGE_GREEN or PROMOTE_FORGE_BLOCKED
    - PROMOTE_PASSPORT_ISSUED
    """
    from dojo.promote.dojo_check import check_dojo  # noqa: PLC0415
    from dojo.promote.forge_check import check_forge  # noqa: PLC0415
    from dojo.promote.harvest_check import check_harvest  # noqa: PLC0415
    from dojo.promote.passport import build_passport  # noqa: PLC0415

    logger = EvidenceLogger.get_instance()
    fsm = _PromotionFSM()

    # INIT → HARVEST_CHECK
    fsm.transition(GateState.HARVEST_CHECK)

    harvest_result = check_harvest(
        harvest_receipt_id,
        harvest_client=harvest_client,
    )

    if not harvest_result.passed:
        fsm.transition(GateState.BLOCKED)
        raise PromotionBlocked(
            f"Promotion blocked at harvest_check for artifact '{candidate.artifact_id}'",
            gate="harvest_check",
            reasons=harvest_result.reasons,
        )

    # HARVEST_CHECK → DOJO_CHECK
    fsm.transition(GateState.DOJO_CHECK)

    dojo_result = check_dojo(
        candidate_f1=candidate_f1,
        baseline_f1=baseline_f1,
        eval_receipt_merkle=eval_receipt_merkle,
    )

    if not dojo_result.passed:
        fsm.transition(GateState.BLOCKED)
        raise PromotionBlocked(
            f"Promotion blocked at dojo_check for artifact '{candidate.artifact_id}'",
            gate="dojo_check",
            reasons=dojo_result.reasons,
        )

    # DOJO_CHECK → FORGE_CHECK
    fsm.transition(GateState.FORGE_CHECK)

    forge_result = check_forge(
        candidate,
        forge_client=forge_client,
    )

    if not forge_result.passed:
        fsm.transition(GateState.BLOCKED)
        raise PromotionBlocked(
            f"Promotion blocked at forge_check for artifact '{candidate.artifact_id}'",
            gate="forge_check",
            reasons=forge_result.reasons,
        )

    # FORGE_CHECK → PASSPORT_BUILD
    fsm.transition(GateState.PASSPORT_BUILD)

    try:
        passport = build_passport(
            candidate=candidate,
            harvest_receipt=harvest_result,
            dojo_receipt=dojo_result,
            forge_receipt=forge_result,
            harvest_receipt_id=harvest_receipt_id,
        )
    except Exception as exc:
        fsm.transition(GateState.BLOCKED)
        raise PromotionBlocked(
            f"Passport build failed for artifact '{candidate.artifact_id}': {exc}",
            gate="passport_build",
            reasons=[str(exc)],
        ) from exc

    # PASSPORT_BUILD → SEALED
    fsm.transition(GateState.SEALED)

    logger.emit(
        {
            "type": EvidenceType.PROMOTE_PASSPORT_ISSUED.value,
            "data": {
                "passport_id": passport.passport_id,
                "model_artifact_id": passport.model_artifact_id,
                "harvest_receipt_id": passport.harvest_receipt_id,
                "merkle_root": passport.merkle_root,
                "issued_at": passport.issued_at.isoformat(),
                "expires_at": passport.expires_at.isoformat(),
                "fsm_history": [(a.value, b.value) for a, b in fsm.history],
            },
        },
        organ="promote",
    )

    return passport


__all__ = [
    "GateCheckResult",
    "GateState",
    "promote",
]

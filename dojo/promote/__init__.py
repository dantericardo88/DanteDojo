"""DanteDojo promotion gate — three-way constitutional gate."""

from __future__ import annotations

from dojo.promote.gate import GateCheckResult, GateState, promote
from dojo.promote.sealer import seal_passport, verify_seal

__all__ = [
    "GateCheckResult",
    "GateState",
    "promote",
    "seal_passport",
    "verify_seal",
]

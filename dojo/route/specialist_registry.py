"""Capability-aware local-specialist registry (d34).

The router's existing local-specialist path checks *availability* only.  This
module adds *capability* awareness so a specialist registered for "code"
tasks cannot accidentally serve a "creative-writing" request.

Capability strings are free-form (the registry doesn't enforce a vocabulary).
Conventionally: domain tags like "code", "summarization", "math", "qa".

Emits ``route.specialist_capability_blocked`` evidence when a specialist is
alive but filtered out for capability mismatch — so audits can see exactly
why a routing decision was made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dojo.core.evidence import EvidenceLogger


@dataclass(frozen=True)
class SpecialistCapability:
    """Declared capabilities of a single local specialist."""

    specialist_id: str
    supported_task_types: Tuple[str, ...]
    alive: bool = True

    def supports(self, task_type: str) -> bool:
        return task_type in self.supported_task_types


@dataclass
class SpecialistRegistry:
    """Registry mapping specialist_id → SpecialistCapability.

    Provides capability-aware filtering with auditable rejection evidence.
    """

    _capabilities: Dict[str, SpecialistCapability] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        specialist_id: str,
        supported_task_types: List[str],
        alive: bool = True,
    ) -> SpecialistCapability:
        """Register a specialist, returning its capability record.

        Re-registering an existing specialist replaces its capabilities.
        """
        cap = SpecialistCapability(
            specialist_id=specialist_id,
            supported_task_types=tuple(supported_task_types),
            alive=alive,
        )
        self._capabilities[specialist_id] = cap
        return cap

    def unregister(self, specialist_id: str) -> None:
        self._capabilities.pop(specialist_id, None)

    def mark_dead(self, specialist_id: str) -> None:
        cap = self._capabilities.get(specialist_id)
        if cap is not None:
            self._capabilities[specialist_id] = SpecialistCapability(
                specialist_id=cap.specialist_id,
                supported_task_types=cap.supported_task_types,
                alive=False,
            )

    def mark_alive(self, specialist_id: str) -> None:
        cap = self._capabilities.get(specialist_id)
        if cap is not None:
            self._capabilities[specialist_id] = SpecialistCapability(
                specialist_id=cap.specialist_id,
                supported_task_types=cap.supported_task_types,
                alive=True,
            )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, specialist_id: str) -> Optional[SpecialistCapability]:
        return self._capabilities.get(specialist_id)

    def all(self) -> List[SpecialistCapability]:
        return list(self._capabilities.values())

    def all_alive(self) -> List[SpecialistCapability]:
        return [c for c in self._capabilities.values() if c.alive]

    def find_capable(self, task_type: str) -> List[SpecialistCapability]:
        """Return alive specialists that support ``task_type``."""
        return [c for c in self._capabilities.values()
                if c.alive and c.supports(task_type)]

    def can_handle(self, task_type: str) -> bool:
        """True iff any alive specialist supports ``task_type``."""
        return len(self.find_capable(task_type)) > 0

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def filter_for_task(self, task_type: str) -> Tuple[List[SpecialistCapability], List[str]]:
        """Return (capable, rejected_reasons).

        Emits one ``route.specialist_capability_blocked`` evidence event per
        rejected specialist so audits can trace routing decisions.

        Returns:
            Tuple of:
            - List of SpecialistCapability that can handle the task
            - List of human-readable rejection reasons (specialist_id + why)
        """
        capable: List[SpecialistCapability] = []
        rejections: List[str] = []

        for cap in self._capabilities.values():
            if not cap.alive:
                rejections.append(f"{cap.specialist_id}: dead")
                continue
            if not cap.supports(task_type):
                rejections.append(
                    f"{cap.specialist_id}: cannot handle '{task_type}' "
                    f"(supports {list(cap.supported_task_types)})"
                )
                EvidenceLogger.get_instance().emit(
                    {
                        "type": "route.specialist_capability_blocked",
                        "data": {
                            "specialist_id": cap.specialist_id,
                            "requested_task": task_type,
                            "supported_tasks": list(cap.supported_task_types),
                        },
                    },
                    organ="route",
                )
                continue
            capable.append(cap)

        return capable, rejections


__all__ = [
    "SpecialistCapability",
    "SpecialistRegistry",
]

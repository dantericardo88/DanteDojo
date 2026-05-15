"""DanteDojo per-rank evidence for FSDP2 multi-GPU training.

In FSDP2, each GPU rank runs an independent forward/backward pass. This module
provides per-rank evidence collection and a cross-rank Merkle root aggregation —
making DanteDojo the only training framework with cryptographic per-rank audit trails.

Novel property: aggregate_rank_merkle_roots() produces a single Merkle root that
commits to every training event on every GPU rank. Tampering with any rank's
loss or step data invalidates the root used in FinetuneResult.training_receipt.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from dojo.core.evidence import EvidenceLogger, EvidenceType


# ---------------------------------------------------------------------------
# Internal Merkle helper — no external dependency
# ---------------------------------------------------------------------------

def _merkle_root(leaves: list[str]) -> str:
    """Compute a Merkle root over a list of string leaves.

    Leaves are sorted before hashing so the root is deterministic regardless
    of emission order. Returns the SHA-256 hash of an empty string if no leaves.
    """
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    current = sorted(leaves)
    while len(current) > 1:
        next_level: list[str] = []
        for i in range(0, len(current), 2):
            left = current[i]
            right = current[i + 1] if i + 1 < len(current) else current[i]
            combined = hashlib.sha256((left + right).encode()).hexdigest()
            next_level.append(combined)
        current = next_level
    return current[0]


# ---------------------------------------------------------------------------
# RankEvidenceBundle — immutable per-rank receipt
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RankEvidenceBundle:
    """Immutable snapshot of one rank's evidence chain after training completes."""

    rank: int
    world_size: int
    steps_completed: int
    final_loss: float
    merkle_root: str
    event_count: int
    sharding_strategy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "steps_completed": self.steps_completed,
            "final_loss": self.final_loss,
            "merkle_root": self.merkle_root,
            "event_count": self.event_count,
            "sharding_strategy": self.sharding_strategy,
        }


# ---------------------------------------------------------------------------
# RankEvidenceCollector — mutable accumulator for one rank
# ---------------------------------------------------------------------------

class RankEvidenceCollector:
    """Accumulates evidence events for one distributed training rank.

    Args:
        rank: This rank's integer index (0 .. world_size-1).
        world_size: Total number of ranks in the job.
        mock_mode: If True, skips torch.distributed calls.

    Raises:
        DojoError: If rank is out of range for the given world_size.
    """

    def __init__(self, rank: int, world_size: int, mock_mode: bool = False) -> None:
        if world_size < 1 or not (0 <= rank < world_size):
            from dojo.core.exceptions import DojoError
            raise DojoError(
                f"Invalid rank={rank} for world_size={world_size}",
                reason="rank must satisfy 0 <= rank < world_size and world_size >= 1",
            )
        self.rank = rank
        self.world_size = world_size
        self.mock_mode = mock_mode
        self._events: list[dict[str, Any]] = []
        self._logger: EvidenceLogger | None = None

    def _get_logger(self) -> EvidenceLogger:
        if self._logger is None:
            self._logger = EvidenceLogger.get_instance()
        return self._logger

    def record(self, event: dict[str, Any]) -> None:
        """Record a training evidence event for this rank and emit to global chain."""
        stamped = {**event, "rank": self.rank, "world_size": self.world_size}
        self._events.append(stamped)
        self._get_logger().emit(
            {
                "type": EvidenceType.FSDP2_TRAINING_STEP.value,
                "data": stamped,
            },
            organ="fsdp2",
        )

    def compute_rank_merkle_root(self) -> str:
        """Return the Merkle root over all events recorded by this rank."""
        leaves = [json.dumps(e, sort_keys=True, separators=(",", ":")) for e in self._events]
        return _merkle_root(leaves)

    def seal(
        self,
        steps_completed: int,
        final_loss: float,
        sharding_strategy: str,
    ) -> RankEvidenceBundle:
        """Seal this rank's evidence chain and return an immutable bundle.

        Emits FSDP2_RANK_MERKLE_ROOT to the global evidence chain so the
        per-rank root is itself part of the tamper-evident log.
        """
        root = self.compute_rank_merkle_root()
        self._get_logger().emit(
            {
                "type": EvidenceType.FSDP2_RANK_MERKLE_ROOT.value,
                "data": {
                    "rank": self.rank,
                    "world_size": self.world_size,
                    "merkle_root": root,
                    "event_count": len(self._events),
                    "steps_completed": steps_completed,
                },
            },
            organ="fsdp2",
        )
        return RankEvidenceBundle(
            rank=self.rank,
            world_size=self.world_size,
            steps_completed=steps_completed,
            final_loss=final_loss,
            merkle_root=root,
            event_count=len(self._events),
            sharding_strategy=sharding_strategy,
        )


# ---------------------------------------------------------------------------
# Cross-rank aggregation
# ---------------------------------------------------------------------------

def aggregate_rank_merkle_roots(bundles: list[RankEvidenceBundle]) -> str:
    """Combine per-rank Merkle roots into a single global Merkle root.

    Ranks are sorted by index before combining, so the result is stable
    regardless of the order bundles are passed. The global root is what gets
    stored in FinetuneResult.training_receipt — committing to every GPU rank's
    training history in a single 64-character hex digest.

    No competitor produces this. It is the constitutional proof of distributed
    training integrity.
    """
    if not bundles:
        return hashlib.sha256(b"").hexdigest()
    sorted_bundles = sorted(bundles, key=lambda b: b.rank)
    leaves = [
        json.dumps(b.to_dict(), sort_keys=True, separators=(",", ":"))
        for b in sorted_bundles
    ]
    return _merkle_root(leaves)


__all__ = [
    "RankEvidenceBundle",
    "RankEvidenceCollector",
    "aggregate_rank_merkle_roots",
]

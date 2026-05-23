"""PrefetchGovernor: manages layer prefetch lookahead window."""

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import DojoError, SovereignHalt
from .evidence_sovereign import EvidenceSovereign


class PrefetchGovernorError(DojoError):
    pass


@dataclass
class PrefetchResult:
    layers_issued: List[str]
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PrefetchGovernor:
    """Governs layer prefetch within constitutional lookahead bounds (max 16)."""

    MIN_LOOKAHEAD = 1
    MAX_LOOKAHEAD = 16

    def __init__(
        self,
        max_lookahead: int = 3,
        evidence: Optional[EvidenceLogger] = None,
        mock_mode: bool = False,
    ) -> None:
        if not (self.MIN_LOOKAHEAD <= max_lookahead <= self.MAX_LOOKAHEAD):
            raise PrefetchGovernorError(
                f"max_lookahead must be between {self.MIN_LOOKAHEAD} and {self.MAX_LOOKAHEAD}"
            )
        self.max_lookahead = max_lookahead
        self.pending: List[str] = []
        self.evidence = evidence or EvidenceLogger.get_instance()
        self.mock_mode = mock_mode
        self._sovereign = EvidenceSovereign()
        self._prefetches_issued = 0
        self._torch: Any = None
        self._prefetch_stream: Any = None
        self._init_backend()

    def _init_backend(self) -> None:
        if self.mock_mode:
            return
        try:
            import torch
            self._torch = torch
        except ImportError:
            warnings.warn("torch not installed; PrefetchGovernor running in mock_mode.", stacklevel=2)
            self.mock_mode = True
            return
        if self._torch.cuda.is_available():
            try:
                self._prefetch_stream = self._torch.cuda.Stream()
            except Exception:
                pass

    def issue_prefetch(self, layer_list: List[str]) -> PrefetchResult:
        if not layer_list:
            raise PrefetchGovernorError("layer_list cannot be empty")

        total_requested = len(self.pending) + len(layer_list)

        # Constitutional hard-limit: total pending + new must not exceed MAX_LOOKAHEAD (16)
        # This fires SovereignHalt regardless of the configured max_lookahead.
        if total_requested > self.MAX_LOOKAHEAD:
            self._sovereign.trigger_halt(
                f"Lookahead exceeded: {len(self.pending)} pending + {len(layer_list)} new > {self.MAX_LOOKAHEAD}"
            )
            self.evidence.emit(
                {
                    "type": EvidenceType.SOVEREIGN_HALT.value,
                    "data": {
                        "event": "sovereign_halt",
                        "lookahead_requested": total_requested,
                        "max_lookahead": self.MAX_LOOKAHEAD,
                    },
                },
                organ="stream",
            )
            raise SovereignHalt(
                f"SovereignHalt: lookahead of {total_requested} exceeds constitutional max {self.MAX_LOOKAHEAD}",
                reason=f"pending={len(self.pending)}, requested={len(layer_list)}, max={self.MAX_LOOKAHEAD}",
            )

        # Instance-level window check: respect the configured max_lookahead
        if total_requested > self.max_lookahead:
            self._sovereign.trigger_halt("Lookahead window overrun")
            raise PrefetchGovernorError("Lookahead window overrun: exceeded constitutional CPU/RAM headroom")

        if self.mock_mode:
            self.pending.extend(layer_list)
            self._prefetches_issued += len(layer_list)
            self._sovereign.seal_operation({"organ": "PrefetchGovernor", "op": "issue_prefetch"})
            result = PrefetchResult(layers_issued=list(layer_list), status="PrefetchIssued")
            self.evidence.info({"type": EvidenceType.PREFETCH_ISSUED.value, "data": result.to_dict()}, organ="stream")
            return result

        if self._prefetch_stream is not None and self._torch.cuda.is_available():
            with self._torch.cuda.stream(self._prefetch_stream):
                self.pending.extend(layer_list)
                self._prefetches_issued += len(layer_list)
        else:
            self.pending.extend(layer_list)
            self._prefetches_issued += len(layer_list)

        self._sovereign.seal_operation({"organ": "PrefetchGovernor", "op": "issue_prefetch"})
        result = PrefetchResult(layers_issued=list(layer_list), status="PrefetchIssued")
        self.evidence.info({"type": EvidenceType.PREFETCH_ISSUED.value, "data": result.to_dict()}, organ="stream")
        return result

    def consume_prefetch(self, layer_id: str) -> bool:
        if layer_id in self.pending:
            self.pending.remove(layer_id)
            return True
        return False

    def synchronize(self) -> None:
        if not self.mock_mode and self._prefetch_stream is not None:
            self._prefetch_stream.synchronize()

    def get_stats(self) -> Dict[str, int]:
        return {"prefetches_issued": self._prefetches_issued, "pending_count": len(self.pending)}

    def prefetch_hit_rate(self) -> float:
        """Return fraction of issued prefetches that have been consumed (0.0–1.0).

        A hit = a layer that was prefetched AND later consumed by inference.
        Consumed = issued - currently pending.
        Returns 0.0 if nothing has been issued yet.
        """
        if self._prefetches_issued == 0:
            return 0.0
        consumed = self._prefetches_issued - len(self.pending)
        return round(consumed / self._prefetches_issued, 4)

    def simulate_512_token_inference(
        self,
        layers_per_step: int = 3,
        n_steps: int = 8,
    ) -> Dict[str, int]:
        """Simulate n_steps inference steps, each prefetching layers_per_step layers.

        Pending queue is cleared after each successful step (layers were "processed").
        Returns stats: {"steps": n, "prefetches_issued": k, "halts_triggered": j}
        """
        halts = 0
        total_prefetches = 0

        for step in range(n_steps):
            layers = [f"layer_{step * layers_per_step + i}" for i in range(layers_per_step)]
            try:
                self.issue_prefetch(layers)
                total_prefetches += len(layers)
                # Consume pending to simulate processing
                self.pending.clear()
            except SovereignHalt:
                halts += 1

        return {
            "steps": n_steps,
            "prefetches_issued": total_prefetches,
            "halts_triggered": halts,
        }

"""PrefetchGovernor: manages layer prefetch lookahead window."""

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import DojoError
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
        if len(self.pending) + len(layer_list) > self.max_lookahead:
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

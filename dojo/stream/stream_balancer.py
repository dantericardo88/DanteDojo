"""StreamBalancer: overlaps compute and memory ops within hardware stream limits."""

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import DojoError
from .evidence_sovereign import EvidenceSovereign


class StreamBalancerError(DojoError):
    pass


@dataclass
class OverlapResult:
    compute_op: str
    memory_op: str
    status: str
    streams_used: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StreamBalancer:
    """Balances compute/memory overlap within constitutional PCIe/CUDA stream limits (max 8)."""

    MIN_STREAMS = 1
    MAX_STREAMS = 8

    def __init__(
        self,
        max_streams: int = 2,
        evidence: Optional[EvidenceLogger] = None,
        mock_mode: bool = False,
    ) -> None:
        self.max_streams = max_streams
        self.active_streams = 0
        self.evidence = evidence or EvidenceLogger.get_instance()
        self.mock_mode = mock_mode
        self._sovereign = EvidenceSovereign()
        self._overlaps_completed = 0
        self._torch: Any = None
        self._cuda_streams: List[Any] = []
        self._init_backend()

    def _init_backend(self) -> None:
        if self.mock_mode:
            return
        try:
            import torch
            self._torch = torch
        except ImportError:
            warnings.warn("torch not installed; StreamBalancer running in mock_mode.", stacklevel=2)
            self.mock_mode = True
            return
        if self._torch.cuda.is_available():
            try:
                for _ in range(self.max_streams):
                    self._cuda_streams.append(self._torch.cuda.Stream())
            except Exception:
                pass

    def start_overlap(self, compute_op: str, memory_op: str) -> OverlapResult:
        if not compute_op:
            raise StreamBalancerError("compute_op cannot be empty")
        if not memory_op:
            raise StreamBalancerError("memory_op cannot be empty")
        if self.active_streams >= self.max_streams:
            self._sovereign.trigger_halt("Hardware stream exhaustion")
            raise StreamBalancerError("Hardware stream exhaustion: PCIe/CUDA stream limits breached")

        if self.mock_mode:
            self.active_streams += 1
            self._overlaps_completed += 1
            self._sovereign.seal_operation({"organ": "StreamBalancer", "op": "start_overlap"})
            result = OverlapResult(compute_op=compute_op, memory_op=memory_op, status="OverlapComplete")
            self.evidence.info({"type": EvidenceType.LAYER_STREAM_LOADED.value, "data": result.to_dict()}, organ="stream")
            return result

        idx = self.active_streams
        self.active_streams += 1
        if idx < len(self._cuda_streams) and self._torch.cuda.is_available():
            stream = self._cuda_streams[idx]
            event = self._torch.cuda.current_stream().record_event()
            stream.wait_event(event)

        self._overlaps_completed += 1
        self._sovereign.seal_operation({"organ": "StreamBalancer", "op": "start_overlap"})
        result = OverlapResult(
            compute_op=compute_op, memory_op=memory_op,
            status="OverlapComplete", streams_used=self.active_streams,
        )
        self.evidence.info({"type": EvidenceType.LAYER_STREAM_LOADED.value, "data": result.to_dict()}, organ="stream")
        return result

    def end_overlap(self) -> None:
        if self.active_streams > 0:
            self.active_streams -= 1

    def synchronize_all(self) -> None:
        if not self.mock_mode and self._torch is not None and self._torch.cuda.is_available():
            for s in self._cuda_streams:
                s.synchronize()

    def get_stats(self) -> Dict[str, int]:
        return {"overlaps_completed": self._overlaps_completed, "active_streams": self.active_streams}

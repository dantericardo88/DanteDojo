"""KVCacheStreamer: pages KV cache entries within sovereign size limits."""

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import DojoError
from .evidence_sovereign import EvidenceSovereign


class KVCacheStreamerError(DojoError):
    pass


@dataclass
class KVPageResult:
    key: str
    size: int
    status: str
    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KVCacheStreamer:
    """Pages KV cache entries with constitutional size limit enforcement."""

    MIN_CACHE_SIZE = 1
    MAX_CACHE_SIZE = 1_000_000  # sovereign memory ceiling (tokens)

    def __init__(
        self,
        max_cache_size: int = 1000,
        evidence: Optional[EvidenceLogger] = None,
        mock_mode: bool = False,
    ) -> None:
        self.max_cache_size = max_cache_size
        self.current_size = 0
        self.evidence = evidence or EvidenceLogger.get_instance()
        self.mock_mode = mock_mode
        self._sovereign = EvidenceSovereign()
        self._pages_stored = 0
        self._torch: Any = None
        self._cache_tensors: Dict[str, Any] = {}
        self._cache_devices: Dict[str, str] = {}
        self._init_backend()

    def _init_backend(self) -> None:
        if self.mock_mode:
            return
        try:
            import torch
            self._torch = torch
        except ImportError:
            warnings.warn("torch not installed; KVCacheStreamer running in mock_mode.", stacklevel=2)
            self.mock_mode = True

    def page_kv_entry(self, key: str, value: str, size: int) -> KVPageResult:
        if not key:
            raise KVCacheStreamerError("key cannot be empty")
        if size <= 0:
            raise KVCacheStreamerError("size must be positive")
        if self.current_size + size > self.max_cache_size:
            self._sovereign.trigger_halt("Sovereign size exceeded")
            raise KVCacheStreamerError("Sovereign size exceeded: KV Cache limit breached")

        if self.mock_mode:
            self.current_size += size
            self._pages_stored += 1
            self._cache_devices[key] = "cpu"
            self._sovereign.seal_operation({"organ": "KVCacheStreamer", "op": "page_kv_entry"})
            result = KVPageResult(key=key, size=size, status="KVReady")
            self.evidence.info({"type": EvidenceType.KV_PAGE_STORED.value, "data": result.to_dict()}, organ="stream")
            return result

        try:
            self._cache_tensors[key] = self._torch.zeros(size, dtype=self._torch.float16)
            self._cache_devices[key] = "cpu"
            self.current_size += size
            self._pages_stored += 1
            self._sovereign.seal_operation({"organ": "KVCacheStreamer", "op": "page_kv_entry"})
            result = KVPageResult(key=key, size=size, status="KVReady", device="cpu")
            self.evidence.info({"type": EvidenceType.KV_PAGE_STORED.value, "data": result.to_dict()}, organ="stream")
            return result
        except KVCacheStreamerError:
            raise
        except Exception as e:
            raise KVCacheStreamerError(f"page_kv_entry failed: {e}") from e

    def cache_to_cpu(self, key: Optional[str] = None) -> List[str]:
        if self.mock_mode:
            keys = [key] if key else list(self._cache_devices.keys())
            for k in keys:
                self._cache_devices[k] = "cpu"
            return keys
        keys_to_move = [key] if key else list(self._cache_tensors.keys())
        moved = []
        cpu = self._torch.device("cpu")
        for k in keys_to_move:
            if k in self._cache_tensors and self._cache_tensors[k] is not None:
                self._cache_tensors[k] = self._cache_tensors[k].to(cpu)
                self._cache_devices[k] = "cpu"
                moved.append(k)
        return moved

    def cache_to_gpu(self, key: Optional[str] = None, device: str = "cuda:0") -> List[str]:
        if self.mock_mode:
            keys = [key] if key else list(self._cache_devices.keys())
            for k in keys:
                self._cache_devices[k] = device
            return keys
        if not self._torch.cuda.is_available():
            return []
        keys_to_move = [key] if key else list(self._cache_tensors.keys())
        moved = []
        gpu = self._torch.device(device)
        for k in keys_to_move:
            if k in self._cache_tensors and self._cache_tensors[k] is not None:
                self._cache_tensors[k] = self._cache_tensors[k].to(gpu)
                self._cache_devices[k] = device
                moved.append(k)
        return moved

    def get_cache_devices(self) -> Dict[str, str]:
        return dict(self._cache_devices)

    def get_stats(self) -> Dict[str, int]:
        return {"pages_stored": self._pages_stored, "current_size": self.current_size}

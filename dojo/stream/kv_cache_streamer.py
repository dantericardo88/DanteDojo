"""KVCacheStreamer: pages KV cache entries within sovereign size limits."""

import warnings
from collections import OrderedDict
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
        # Ordered dict tracks insertion order for eviction (oldest-first)
        self._cache_sizes: OrderedDict[str, int] = OrderedDict()
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
            self._cache_sizes[key] = size
            self._sovereign.seal_operation({"organ": "KVCacheStreamer", "op": "page_kv_entry"})
            result = KVPageResult(key=key, size=size, status="KVReady")
            self.evidence.info({"type": EvidenceType.KV_PAGE_STORED.value, "data": result.to_dict()}, organ="stream")
            return result

        try:
            self._cache_tensors[key] = self._torch.zeros(size, dtype=self._torch.float16)
            self._cache_devices[key] = "cpu"
            self._cache_sizes[key] = size
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

    def evict_oldest(self, n: int = 1) -> int:
        """Evict the n oldest KV cache entries (by insertion order).

        Returns the number of entries actually evicted.  If fewer than n
        entries exist, all are evicted and the actual count is returned.
        """
        evicted = 0
        while evicted < n and self._cache_sizes:
            oldest_key, oldest_size = next(iter(self._cache_sizes.items()))
            del self._cache_sizes[oldest_key]
            self._cache_devices.pop(oldest_key, None)
            if not self.mock_mode:
                self._cache_tensors.pop(oldest_key, None)
            self.current_size = max(0, self.current_size - oldest_size)
            self.evidence.info(
                {
                    "type": EvidenceType.LAYER_STREAM_EVICTED.value,
                    "data": {"evicted_key": oldest_key, "freed_size": oldest_size, "current_size": self.current_size},
                },
                organ="stream",
            )
            evicted += 1
        return evicted

    def simulate_generation(
        self,
        n_tokens: int = 512,
        tokens_per_page: int = 1,
        evict_when_full: bool = True,
    ) -> Dict[str, int]:
        """Simulate n_tokens of autoregressive generation with KV paging.

        For each generated token:
        1. Add its KV entry to the cache.
        2. If the cache is full and evict_when_full is True, evict the oldest
           entries to make room, then retry.
        3. Emit an evidence event every 64 tokens.

        Returns:
            {"tokens_generated": n, "evictions_fired": k, "final_cache_size": m}
        """
        evictions_fired = 0

        for i in range(n_tokens):
            key = f"token_{i:05d}"
            value = f"kv_val_{i}"
            size = tokens_per_page

            try:
                self.page_kv_entry(key, value, size)
            except KVCacheStreamerError:
                if evict_when_full:
                    evicted_count = self.evict_oldest(n=size)
                    evictions_fired += evicted_count
                    # Retry after eviction
                    self.page_kv_entry(key, value, size)
                else:
                    raise

            if i > 0 and i % 64 == 0:
                self.evidence.emit(
                    {
                        "type": EvidenceType.LAYER_STREAM_EVICTED.value,
                        "data": {
                            "tokens_processed": i + 1,
                            "cache_size": self.current_size,
                            "evictions_so_far": evictions_fired,
                        },
                    },
                    organ="stream",
                )

        return {
            "tokens_generated": n_tokens,
            "evictions_fired": evictions_fired,
            "final_cache_size": self.current_size,
        }

    def get_stats(self) -> Dict[str, int]:
        return {"pages_stored": self._pages_stored, "current_size": self.current_size}

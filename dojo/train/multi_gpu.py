"""DanteDojo multi-GPU profile detection and FSDP2 strategy recommendation.

Detects available GPU count via torch.cuda (or falls back gracefully) and
recommends the appropriate FSDP2 sharding strategy based on device count and
per-device VRAM. Fully mock_mode=True compatible — no real GPU required.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from dojo.core.evidence import EvidenceLogger, EvidenceType

# Strategy string constants mirror torch.distributed.fsdp.ShardingStrategy names
# so they can be passed directly to the mapping in _run_real_fsdp2.
STRATEGY_FULL_SHARD    = "FULL_SHARD"     # ZeRO-3: params + grads + optim sharded
STRATEGY_SHARD_GRAD_OP = "SHARD_GRAD_OP"  # ZeRO-2: grads + optim sharded, params replicated
STRATEGY_NO_SHARD      = "NO_SHARD"       # DDP: fully replicated (no sharding)
STRATEGY_HYBRID_SHARD  = "HYBRID_SHARD"   # ZeRO-3 within node, replicated across nodes

_GIB = 1024 ** 3


@dataclass(frozen=True)
class MultiGPUProfile:
    """Immutable snapshot of detected multi-GPU capabilities."""

    device_count: int
    device_names: tuple[str, ...]
    total_vram_bytes: int
    per_device_vram_bytes: int
    has_nvlink: bool
    recommended_strategy: str
    backend: str        # "nccl" for CUDA, "gloo" for CPU/test
    mock_mode: bool

    @property
    def total_vram_gb(self) -> float:
        return self.total_vram_bytes / _GIB

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_count": self.device_count,
            "device_names": list(self.device_names),
            "total_vram_bytes": self.total_vram_bytes,
            "per_device_vram_bytes": self.per_device_vram_bytes,
            "has_nvlink": self.has_nvlink,
            "recommended_strategy": self.recommended_strategy,
            "backend": self.backend,
            "mock_mode": self.mock_mode,
        }

    def profile_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _select_strategy(device_count: int, per_device_vram_bytes: int) -> str:
    """Return the recommended FSDP2 sharding strategy for given hardware."""
    if device_count <= 1:
        return STRATEGY_NO_SHARD
    if device_count >= 4 and per_device_vram_bytes >= 40 * _GIB:
        return STRATEGY_HYBRID_SHARD
    if device_count >= 2 and per_device_vram_bytes >= 24 * _GIB:
        return STRATEGY_FULL_SHARD
    return STRATEGY_SHARD_GRAD_OP


def detect_multi_gpu(mock_mode: bool = False) -> MultiGPUProfile:
    """Detect available GPUs and recommend an FSDP2 sharding strategy.

    Args:
        mock_mode: If True, returns a deterministic 2-GPU mock profile
                   without touching torch.cuda. Use in tests and CI.

    Returns:
        MultiGPUProfile — immutable snapshot of detected capabilities.
    """
    if mock_mode:
        per_vram = 24 * _GIB
        profile = MultiGPUProfile(
            device_count=2,
            device_names=("Mock-GPU-0", "Mock-GPU-1"),
            total_vram_bytes=2 * per_vram,
            per_device_vram_bytes=per_vram,
            has_nvlink=False,
            recommended_strategy=STRATEGY_FULL_SHARD,
            backend="gloo",
            mock_mode=True,
        )
        EvidenceLogger.get_instance().emit(
            {
                "type": EvidenceType.FSDP2_PROFILE_DETECTED.value,
                "data": profile.to_dict(),
            },
            organ="train",
        )
        return profile

    # --- Real GPU detection path ---
    device_count = 0
    device_names: list[str] = []
    per_device_vram = 0
    has_nvlink = False

    try:
        import torch
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            device_names.append(props.name)
            if i == 0:
                per_device_vram = props.total_memory
            # NVLink heuristic: multi-GPU with identical VRAM on server hardware
            if device_count > 1 and props.total_memory == per_device_vram:
                has_nvlink = props.total_memory >= 40 * _GIB
    except Exception:
        pass

    strategy = _select_strategy(device_count, per_device_vram)
    backend = "nccl" if device_count >= 1 else "gloo"

    profile = MultiGPUProfile(
        device_count=device_count,
        device_names=tuple(device_names),
        total_vram_bytes=device_count * per_device_vram,
        per_device_vram_bytes=per_device_vram,
        has_nvlink=has_nvlink,
        recommended_strategy=strategy,
        backend=backend,
        mock_mode=False,
    )

    EvidenceLogger.get_instance().emit(
        {
            "type": EvidenceType.FSDP2_PROFILE_DETECTED.value,
            "data": profile.to_dict(),
        },
        organ="train",
    )
    return profile


__all__ = [
    "MultiGPUProfile",
    "STRATEGY_FULL_SHARD",
    "STRATEGY_HYBRID_SHARD",
    "STRATEGY_NO_SHARD",
    "STRATEGY_SHARD_GRAD_OP",
    "detect_multi_gpu",
]

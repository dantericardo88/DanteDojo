"""DanteDojo training FSM — BaseTrainer, FinetuneConfig, FinetuneResult.

Defines the canonical training state machine, frozen config dataclass, and
immutable result dataclass used by all training backends.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.fsm import BaseFSM


# ---------------------------------------------------------------------------
# FSM states
# ---------------------------------------------------------------------------

class TrainingState(Enum):
    IDLE = "idle"
    PROFILING = "profiling"
    ROUTING = "routing"
    PREPARING = "preparing"
    TRAINING = "training"
    CHECKPOINTING = "checkpointing"
    EVALUATING = "evaluating"
    PROMOTING = "promoting"
    FAILED = "failed"
    COMPLETED = "completed"


# ---------------------------------------------------------------------------
# BaseTrainer FSM
# ---------------------------------------------------------------------------

class BaseTrainer(BaseFSM[TrainingState]):
    """FSM that governs the lifecycle of a fine-tuning run.

    Every state transition emits an evidence event so the run is fully
    auditable even if it fails mid-way.
    """

    initial_state = TrainingState.IDLE

    _TRANSITIONS: dict[TrainingState, frozenset[TrainingState]] = {
        TrainingState.IDLE: frozenset({TrainingState.PROFILING}),
        TrainingState.PROFILING: frozenset({TrainingState.ROUTING, TrainingState.FAILED}),
        TrainingState.ROUTING: frozenset({TrainingState.PREPARING, TrainingState.FAILED}),
        TrainingState.PREPARING: frozenset({TrainingState.TRAINING, TrainingState.FAILED}),
        TrainingState.TRAINING: frozenset({TrainingState.CHECKPOINTING, TrainingState.FAILED}),
        TrainingState.CHECKPOINTING: frozenset({TrainingState.EVALUATING, TrainingState.FAILED}),
        TrainingState.EVALUATING: frozenset({
            TrainingState.PROMOTING,
            TrainingState.COMPLETED,
            TrainingState.FAILED,
        }),
        TrainingState.PROMOTING: frozenset({TrainingState.COMPLETED, TrainingState.FAILED}),
    }

    def __init__(self) -> None:
        super().__init__()
        self._logger = EvidenceLogger.get_instance()

    def on_transition(self, previous: TrainingState, current: TrainingState) -> None:
        """Emit an evidence event on every state transition."""
        self._logger.emit(
            {
                "type": "train.fsm.transition",
                "data": {
                    "from": previous.value,
                    "to": current.value,
                },
            },
            organ="train",
        )


# ---------------------------------------------------------------------------
# FinetuneConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FinetuneConfig:
    """Immutable fine-tuning configuration.

    Defaults are safe for a 4060-laptop (8 GB VRAM).  Larger hardware should
    override batch_size, max_seq_length, and num_epochs via keyword arguments.
    """

    pack_id: str
    base_model: str
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    batch_size: int = 1
    grad_accum_steps: int = 8
    max_seq_length: int = 2048
    num_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("artifacts/train_run"))
    # Precision / training mode: "sft", "qlora", "fp8", "bf16"
    training_mode: str = "sft"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "base_model": self.base_model,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "batch_size": self.batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "max_seq_length": self.max_seq_length,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "seed": self.seed,
            "output_dir": str(self.output_dir),
            "training_mode": self.training_mode,
        }

    def config_hash(self) -> str:
        """Stable SHA-256 hash of the config dict (sorted keys)."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# FinetuneResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FinetuneResult:
    """Immutable result returned after a fine-tuning run completes."""

    adapter_path: Path
    base_model: str
    training_receipt: str    # Merkle root from evidence chain
    final_loss: float
    steps_completed: int
    wall_time_seconds: float
    backend_used: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_path": str(self.adapter_path),
            "base_model": self.base_model,
            "training_receipt": self.training_receipt,
            "final_loss": self.final_loss,
            "steps_completed": self.steps_completed,
            "wall_time_seconds": self.wall_time_seconds,
            "backend_used": self.backend_used,
        }


@dataclass(frozen=True)
class FSDP2Config:
    """Configuration overlay for FSDP2 multi-GPU training."""

    sharding_strategy: str = "FULL_SHARD"
    world_size: int = 0           # 0 = auto-detect from torch.cuda.device_count()
    backend: str = "nccl"
    cpu_offload_params: bool = False
    cpu_offload_activations: bool = False
    transformer_layer_cls: str = "LlamaDecoderLayer"
    use_orig_params: bool = True
    activation_checkpointing: bool = False

    def config_hash(self) -> str:
        payload = json.dumps(
            {
                "sharding_strategy": self.sharding_strategy,
                "world_size": self.world_size,
                "backend": self.backend,
                "cpu_offload_params": self.cpu_offload_params,
                "cpu_offload_activations": self.cpu_offload_activations,
                "transformer_layer_cls": self.transformer_layer_cls,
                "use_orig_params": self.use_orig_params,
                "activation_checkpointing": self.activation_checkpointing,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "BaseTrainer",
    "FSDP2Config",
    "FinetuneConfig",
    "FinetuneResult",
    "TrainingState",
]

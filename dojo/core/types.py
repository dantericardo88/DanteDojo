"""DanteDojo core type definitions.

All dataclasses are frozen=True and JSON-serializable with stable key ordering.
Pydantic v2 is used for validation at system boundaries; internal structures
use stdlib dataclasses for zero-overhead construction.

Transplanted and extended from STEAK V3 steak/core/types.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class HardwareTier(str, Enum):
    TIER_4060_LAPTOP = "tier_4060_laptop"    # Ada SM 8.9, 8GB, NF4 QLoRA capable
    TIER_3090 = "tier_3090"                  # Ampere SM 8.6, 24GB, NO native FP8
    TIER_3090_DUAL = "tier_3090_dual"        # 2×3090, 48GB, vLLM TP=2
    TIER_H100 = "tier_h100"                  # Hopper SM 9.0, 80GB, native FP8
    TIER_APPLE_M4 = "tier_apple_m4"          # Apple MLX, unified memory
    TIER_AMD_MI300 = "tier_amd_mi300"        # AMD ROCm, native FP8
    TIER_INTEL_B580 = "tier_intel_b580"      # Intel Arc, 12GB
    TIER_CPU = "tier_cpu"                    # Fallback — inference only


class TrainingMode(str, Enum):
    LORA = "lora"
    QLORA = "qlora"
    FULL_FT = "full_ft"
    PRETRAIN = "pretrain"


class PrecisionType(str, Enum):
    BF16 = "bf16"
    FP8 = "fp8"        # Requires SM >= 8.9 (Ada/Hopper/Blackwell only)
    FP4 = "fp4"        # Requires SM >= 10.0 (Blackwell only)
    NF4 = "nf4"        # bitsandbytes NF4 for QLoRA
    INT4 = "int4"      # llama.cpp GGUF INT4
    W8A16 = "w8a16"    # Weight-only INT8 / Marlin kernel (Ampere safe)


class TrainingBackend(str, Enum):
    UNSLOTH = "unsloth"    # Single-GPU, Ada/Ampere, speed champion
    AXOLOTL = "axolotl"    # Multi-GPU, config-driven, FSDP2
    MLX = "mlx"            # Apple Silicon
    NATIVE = "native"      # HF Transformers directly (H100 / custom)


class Backend(str, Enum):
    VLLM = "vllm"
    SGLANG = "sglang"
    LLAMACPP = "llamacpp"
    MLX_LM = "mlx_lm"


class PackFormat(str, Enum):
    INSTRUCTION = "instruction"
    CONVERSATION = "conversation"
    COMPLETION = "completion"
    ALPACA = "alpaca"
    SHAREGPT = "sharegpt"


class PrivacyTier(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"    # never route to external API
    SECRETS = "secrets"              # never route to external API


class RoutingOption(str, Enum):
    LOCAL_SPECIALIST = "local_specialist"
    LOCAL_GENERALIST = "local_generalist"
    API_HAIKU = "api_haiku"
    API_SONNET = "api_sonnet"
    API_DEEPSEEK = "api_deepseek"
    API_GPT5_MINI = "api_gpt5_mini"


# ---------------------------------------------------------------------------
# Hardware profile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HALProfile:
    """Signed hardware capability profile emitted by dojo/hal/detector.py."""

    tier: HardwareTier
    gpu_name: str
    vram_bytes: int
    compute_capability: str          # e.g. "8.6" for Ampere, "8.9" for Ada
    has_native_fp8: bool             # True iff CC >= 8.9
    has_native_fp4: bool             # True iff CC >= 10.0 (Blackwell)
    has_nvlink: bool
    pcie_gen: int
    interconnect_bandwidth_gb_s: float
    memory_bandwidth_gb_s: float
    driver_version: str
    profile_sha256: str
    signed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def vram_gb(self) -> float:
        return self.vram_bytes / (1024 ** 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "gpu_name": self.gpu_name,
            "vram_bytes": self.vram_bytes,
            "compute_capability": self.compute_capability,
            "has_native_fp8": self.has_native_fp8,
            "has_native_fp4": self.has_native_fp4,
            "has_nvlink": self.has_nvlink,
            "pcie_gen": self.pcie_gen,
            "memory_bandwidth_gb_s": self.memory_bandwidth_gb_s,
            "driver_version": self.driver_version,
            "profile_sha256": self.profile_sha256,
            "signed_at": self.signed_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Dojo configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DojoConfig:
    """Top-level Dojo runtime configuration. Frozen for deterministic hashing."""

    hardware_tier: HardwareTier
    training_mode: TrainingMode
    precision: PrecisionType
    backend: TrainingBackend
    seed: int
    evidence_dir: Path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hardware_tier": self.hardware_tier.value,
            "training_mode": self.training_mode.value,
            "precision": self.precision.value,
            "backend": self.backend.value,
            "seed": self.seed,
            "evidence_dir": str(self.evidence_dir),
        }

    def stable_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Pack types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LicenseSpec:
    spdx_id: str
    allows_derived_ft: bool       # permits fine-tuned model redistribution
    allows_commercial_use: bool
    requires_attribution: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spdx_id": self.spdx_id,
            "allows_derived_ft": self.allows_derived_ft,
            "allows_commercial_use": self.allows_commercial_use,
            "requires_attribution": self.requires_attribution,
        }


@dataclass(frozen=True)
class ProvenanceManifest:
    harvest_organ_version: str
    sources: tuple[str, ...]       # list of source URIs / identifiers
    collection_date_utc: str
    dedup_method: str
    human_reviewed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "harvest_organ_version": self.harvest_organ_version,
            "sources": list(self.sources),
            "collection_date_utc": self.collection_date_utc,
            "dedup_method": self.dedup_method,
            "human_reviewed": self.human_reviewed,
        }


@dataclass(frozen=True)
class PackReference:
    """Pointer to a DanteHarvest specialization pack."""

    uri: str                       # e.g. "harvest://empanada-bookkeeping/v3"
    sha256: str
    signature: bytes
    harvest_receipt_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uri": self.uri,
            "sha256": self.sha256,
            "signature": self.signature.hex(),
            "harvest_receipt_id": self.harvest_receipt_id,
        }


# ---------------------------------------------------------------------------
# Quantization spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantizationSpec:
    method: str                    # "gguf_q4_k_m", "gptq_int4", "awq_int4", "nf4"
    bits: int
    group_size: Optional[int] = None
    calibration_dataset: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "bits": self.bits,
            "group_size": self.group_size,
            "calibration_dataset": self.calibration_dataset,
        }


# ---------------------------------------------------------------------------
# Model artifact
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelArtifact:
    """OCI-compatible model artifact registered in the Dojo model registry."""

    artifact_id: str               # e.g. "dojo://qwen25-7b-empanada-bookkeeping/v1"
    base_model: str                # HF Hub model ID or local path
    lora_weights: Optional[Path]
    quantization: Optional[QuantizationSpec]
    training_receipt: str          # Merkle root from evidence chain
    passport: Optional["ConstitutionalPassport"]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "base_model": self.base_model,
            "lora_weights": str(self.lora_weights) if self.lora_weights else None,
            "quantization": self.quantization.to_dict() if self.quantization else None,
            "training_receipt": self.training_receipt,
            "has_passport": self.passport is not None,
            "created_at": self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Constitutional passport
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConstitutionalPassport:
    """Three-way sealed passport required before DanteAgents loads a model."""

    passport_id: str
    model_artifact_id: str
    harvest_receipt_id: str
    harvest_check_receipt: bytes     # signed by Harvest key
    dojo_check_receipt: bytes        # signed by Dojo key
    forge_check_receipt: bytes       # signed by Forge key
    merkle_root: str                 # combines all three receipts + artifact fingerprint
    issued_at: datetime
    expires_at: datetime             # default: issued_at + 90 days
    revocation_url: str              # Forge endpoint to check revocation

    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passport_id": self.passport_id,
            "model_artifact_id": self.model_artifact_id,
            "harvest_receipt_id": self.harvest_receipt_id,
            "merkle_root": self.merkle_root,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "revocation_url": self.revocation_url,
            "is_expired": self.is_expired(),
        }


# ---------------------------------------------------------------------------
# Deployment manifest
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeploymentManifest:
    """Describes a running inference server instance."""

    model_artifact_id: str
    backend: Backend
    endpoint_url: str
    context_length: int
    quantization: Optional[str]
    tp_size: int
    dtype: str
    passport_id: str
    started_at: datetime
    pid: int
    manifest_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_artifact_id": self.model_artifact_id,
            "backend": self.backend.value,
            "endpoint_url": self.endpoint_url,
            "context_length": self.context_length,
            "quantization": self.quantization,
            "tp_size": self.tp_size,
            "dtype": self.dtype,
            "passport_id": self.passport_id,
            "started_at": self.started_at.isoformat(),
            "pid": self.pid,
            "manifest_sha256": self.manifest_sha256,
        }


# ---------------------------------------------------------------------------
# Routing types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoutingPolicy:
    task_name: str
    preference_order: tuple[RoutingOption, ...]
    max_latency_ms: int
    max_cost_usd: float
    privacy_tier: PrivacyTier
    fallback_on_unavailable: bool

    def allows_external_api(self) -> bool:
        return self.privacy_tier not in (PrivacyTier.CONFIDENTIAL, PrivacyTier.SECRETS)


@dataclass(frozen=True)
class CostReceipt:
    task: str
    route_taken: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    local_displaced_estimate_usd: float
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "route_taken": self.route_taken,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "local_displaced_estimate_usd": self.local_displaced_estimate_usd,
            "recorded_at": self.recorded_at.isoformat(),
        }


__all__ = [
    "Backend",
    "ConstitutionalPassport",
    "CostReceipt",
    "DeploymentManifest",
    "DojoConfig",
    "HALProfile",
    "HardwareTier",
    "LicenseSpec",
    "ModelArtifact",
    "PackFormat",
    "PackReference",
    "PrecisionType",
    "PrivacyTier",
    "ProvenanceManifest",
    "QuantizationSpec",
    "RoutingOption",
    "RoutingPolicy",
    "TrainingBackend",
    "TrainingMode",
]

"""DanteDojo exception hierarchy.

Every subsystem raises a subclass of DojoError so callers can catch at any
granularity. All errors carry an actionable `reason` string — no silent failures.
"""

from __future__ import annotations

from typing import Any, Optional


class DojoError(Exception):
    """Base exception for all DanteDojo errors."""

    def __init__(self, message: str, reason: str = "", details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.reason = reason or message
        self.details = details or {}

    def __str__(self) -> str:
        if self.reason and self.reason != self.args[0]:
            return f"{self.args[0]} — {self.reason}"
        return self.args[0]


# --- Constitutional violations ---

class OneDoorViolation(DojoError):
    """Raised when a caller bypasses the one-door entry point of a subsystem."""


class SovereignHalt(DojoError):
    """Raised when a constitutional hard limit is breached during inference.

    Examples: lookahead > 16 layers, VRAM eviction failure.
    """


class EvidenceChainError(DojoError):
    """Raised when the evidence chain is corrupt or cannot be written."""


class FSMTransitionError(DojoError):
    """Raised on illegal FSM state transitions."""


class ConstitutionalViolation(DojoError):
    """Raised when any constitutional precondition is not met."""


# --- Hardware / precision ---

class HALDetectionError(DojoError):
    """Raised when hardware autodetection fails with no viable backend."""


class PrecisionUnsupported(DojoError):
    """Raised when a requested precision is not supported on the detected hardware.

    Example: FP8 compute on Ampere (SM 8.6) raises this — caller must route to BF16.
    """


class VRAMExceeded(DojoError):
    """Raised when the requested configuration would exceed available VRAM."""


# --- Pack ingest ---

class PackError(DojoError):
    """Base exception for pack ingest failures."""


class PackSignatureInvalid(PackError):
    """Raised when an Ed25519 pack signature does not verify."""


class PackSchemaInvalid(PackError):
    """Raised when a pack is missing required schema fields."""


class PackContaminationDetected(PackError):
    """Raised when a pack sample appears in the eval gold set."""


class PackLicenseRejected(PackError):
    """Raised when a pack's license prohibits derived fine-tuned redistribution."""


# --- Training ---

class TrainingError(DojoError):
    """Base exception for training failures."""


class LossDivergenceError(TrainingError):
    """Raised on NaN/Inf loss detected during training."""


class CheckpointError(TrainingError):
    """Raised on checkpoint save/restore failures."""


class BackendUnavailable(TrainingError):
    """Raised when the requested training backend (Unsloth/Axolotl) is not installed."""


class FSDP2ConfigError(TrainingError):
    """Raised when FSDP2 sharding strategy or config parameters are invalid."""


class RLTrainingError(TrainingError):
    """Base exception for reinforcement-learning training failures (GRPO, DPO, PPO)."""


class KLExplosionError(RLTrainingError):
    """Raised when KL divergence exceeds the constitutional max_kl ceiling during GRPO training."""


class GRPODivergenceError(RLTrainingError):
    """Raised when GRPO reward or policy diverges beyond recoverable bounds."""


class DPORewardCollapse(RLTrainingError):
    """Raised when chosen_reward ≤ rejected_reward for too many steps (reward margin collapse)."""


# --- Quantization ---

class QuantizationError(DojoError):
    """Base exception for quantization failures."""


class QuantQualityRejected(QuantizationError):
    """Raised when post-quantization perplexity or F1 regression exceeds threshold."""


# --- Serving ---

class ServeError(DojoError):
    """Base exception for serving failures."""


class ServerStartError(ServeError):
    """Raised when the inference server subprocess fails to start."""


class ServerHealthTimeout(ServeError):
    """Raised when the server does not pass health checks within the warm-up window."""


# --- Routing ---

class RouteError(DojoError):
    """Base exception for routing failures."""


class PrivacyTierViolation(RouteError):
    """Raised when a CONFIDENTIAL/SECRETS task would be sent to an external API."""


class NoRouteAvailable(RouteError):
    """Raised when no valid route exists for the given task and privacy tier."""


class BudgetExceeded(RouteError):
    """Raised when a routing request would exceed the monthly budget limit."""


class RoutingProviderUnavailable(RouteError):
    """Raised when all preferred providers are unavailable and no fallback exists."""


# --- Evaluation ---

class EvalError(DojoError):
    """Base exception for evaluation failures."""


class EvalContaminationError(EvalError):
    """Raised when task contamination is detected against the training pack."""


class JudgeDriftError(EvalError):
    """Raised when the judge-vs-gold agreement drops below the minimum threshold."""


# --- Promotion ---

class PromotionError(DojoError):
    """Base exception for promotion gate failures."""


class PromotionBlocked(PromotionError):
    """Raised when the three-way gate blocks a candidate. Includes gate-level reasons."""

    def __init__(self, message: str, gate: str, reasons: list[str]) -> None:
        super().__init__(message, reason="; ".join(reasons))
        self.gate = gate
        self.reasons = reasons


class PassportExpired(PromotionError):
    """Raised when a constitutional passport has passed its expiry date."""


class PassportRevoked(PromotionError):
    """Raised when a passport revocation check returns True."""


# --- Ecosystem ---

class EcosystemError(DojoError):
    """Base exception for cross-organ HTTP/artifact failures."""


class HarvestClientError(EcosystemError):
    """Raised when DanteHarvest is unreachable or returns an error."""


class ForgeClientError(EcosystemError):
    """Raised when DanteForge is unreachable or returns an error. Always fail-closed."""


class RegistryError(DojoError):
    """Raised on model registry read/write failures."""


__all__ = [
    "DojoError",
    "OneDoorViolation",
    "SovereignHalt",
    "EvidenceChainError",
    "FSMTransitionError",
    "ConstitutionalViolation",
    "HALDetectionError",
    "PrecisionUnsupported",
    "VRAMExceeded",
    "PackError",
    "PackSignatureInvalid",
    "PackSchemaInvalid",
    "PackContaminationDetected",
    "PackLicenseRejected",
    "TrainingError",
    "LossDivergenceError",
    "CheckpointError",
    "BackendUnavailable",
    "FSDP2ConfigError",
    "RLTrainingError",
    "KLExplosionError",
    "GRPODivergenceError",
    "DPORewardCollapse",
    "QuantizationError",
    "QuantQualityRejected",
    "ServeError",
    "ServerStartError",
    "ServerHealthTimeout",
    "RouteError",
    "PrivacyTierViolation",
    "NoRouteAvailable",
    "BudgetExceeded",
    "RoutingProviderUnavailable",
    "EvalError",
    "EvalContaminationError",
    "JudgeDriftError",
    "PromotionError",
    "PromotionBlocked",
    "PassportExpired",
    "PassportRevoked",
    "EcosystemError",
    "HarvestClientError",
    "ForgeClientError",
    "RegistryError",
]

"""DanteDojo context-length manager.

Provides safe context-length caps per hardware tier and a VRAM estimation
utility used for pre-flight checks before a training run starts.
"""

from __future__ import annotations

from dojo.core.types import HALProfile, HardwareTier, PrecisionType


# Bytes per parameter for each precision type (used in VRAM estimation).
_BYTES_PER_PARAM: dict[PrecisionType, float] = {
    PrecisionType.BF16: 2.0,
    PrecisionType.FP8: 1.0,
    PrecisionType.FP4: 0.5,
    PrecisionType.NF4: 0.5,   # bitsandbytes NF4 ≈ 4-bit
    PrecisionType.INT4: 0.5,
    PrecisionType.W8A16: 1.0,  # weight-only int8, activations in fp16
}

# Maximum safe context lengths per hardware tier.
_MAX_CONTEXT: dict[HardwareTier, int] = {
    HardwareTier.TIER_4060_LAPTOP: 2048,
    HardwareTier.TIER_3090: 8192,
    HardwareTier.TIER_3090_DUAL: 8192,
    HardwareTier.TIER_H100: 32768,
    HardwareTier.TIER_APPLE_M4: 8192,
    HardwareTier.TIER_AMD_MI300: 32768,
    HardwareTier.TIER_INTEL_B580: 4096,
    HardwareTier.TIER_CPU: 512,
}


class ContextLengthManager:
    """Hardware-aware context length capping and VRAM estimation."""

    MAX_SAFE_4060: int = 2048
    MAX_SAFE_3090: int = 8192
    MAX_SAFE_H100: int = 32768

    @staticmethod
    def get_safe_context(profile: HALProfile, requested: int = 4096) -> int:
        """Return the safe context length for *profile*, capped at *requested*.

        Never exceeds the hardware-specific limit defined in ``_MAX_CONTEXT``.
        """
        hw_max = _MAX_CONTEXT.get(profile.tier, 512)
        return min(hw_max, requested)

    @staticmethod
    def estimate_vram_for_config(
        model_params_b: float,
        batch_size: int,
        seq_len: int,
        precision: PrecisionType,
        lora_r: int = 16,
    ) -> int:
        """Estimate VRAM bytes required for a training configuration.

        Estimation formula (conservative):
        - Model weights:   params_b * 1e9 * bytes_per_param
        - Gradients:       same as weights for BF16; half for 4-bit (gradients in BF16)
        - Optimizer states (AdamW): 2× model weights in BF16
        - LoRA parameters: lora_r * hidden_dim_estimate * 2 * 2B (two adapters, BF16)
        - Activations:     batch_size * seq_len * hidden_dim * 4B (rough)

        Returns integer bytes.
        """
        params = model_params_b * 1e9
        bpp = _BYTES_PER_PARAM.get(precision, 2.0)

        # Rough hidden dim estimate: empirical ~256 per billion params (capped)
        hidden_dim = min(int(model_params_b * 256), 8192)

        # LoRA adapter parameters (two matrices per layer, ≈32 layers estimate)
        lora_layers = 32
        lora_params = lora_r * hidden_dim * 2 * lora_layers

        # QLoRA / NF4 / INT4: base model weights are frozen 4-bit; only LoRA
        # adapters are trainable in BF16 with full optimizer states.
        is_qlora = precision in (PrecisionType.NF4, PrecisionType.INT4, PrecisionType.FP4)

        # Model weights
        weight_bytes = params * bpp

        if is_qlora:
            # Frozen base weights in NF4; gradients and optimizer states only for LoRA
            grad_bytes = lora_params * 2.0          # LoRA grads in BF16
            optim_bytes = lora_params * 2.0 * 2.0   # AdamW: 2 moments in BF16
        else:
            # Full-precision training: gradients + AdamW states for all params
            grad_bytes = params * 2.0
            optim_bytes = params * 2.0 * 2.0

        lora_bytes = lora_params * 2.0  # BF16

        # Activations (batch * seq * hidden * fp32)
        activation_bytes = batch_size * seq_len * hidden_dim * 4.0

        total = weight_bytes + grad_bytes + optim_bytes + lora_bytes + activation_bytes
        return int(total)


__all__ = ["ContextLengthManager"]

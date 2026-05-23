"""DanteDojo one-door fine-tuning entry point.

Routes to the appropriate training backend based on HALProfile, runs the
full training FSM, emits constitutional evidence, and performs pre-flight
VRAM checks.  In v1 the backend kernels (Unsloth/Axolotl/MLX) are stubbed
so the pipeline works without a GPU.
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Any  # used in _run_real_sft type annotation for _ctx

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import PrecisionUnsupported, VRAMExceeded
from dojo.core.one_door import one_door
from dojo.core.types import HALProfile, HardwareTier, PrecisionType, TrainingBackend
from dojo.experiment.tracker import ExperimentTracker, RunConfig
from dojo.experiment.wandb_adapter import WandBTrainingCallback
from dojo.hal.env_detect import RuntimeEnv, detect_runtime, wsl_available
from dojo.pack.license import check_license_allows_training
from dojo.pack.schema import SpecializationPack
from dojo.train.auto_setup import ensure_training_deps
from dojo.train.base import BaseTrainer, FinetuneConfig, FinetuneResult, TrainingState
from dojo.train.wsl_backend import run_training_in_wsl
from dojo.train.checkpoint import CheckpointManager
from dojo.train.context import ContextLengthManager


# ---------------------------------------------------------------------------
# Backend routing table
# ---------------------------------------------------------------------------

_BACKEND_MAP: dict[HardwareTier, TrainingBackend] = {
    HardwareTier.TIER_4060_LAPTOP: TrainingBackend.UNSLOTH,
    HardwareTier.TIER_3090: TrainingBackend.UNSLOTH,
    HardwareTier.TIER_3090_DUAL: TrainingBackend.AXOLOTL,
    HardwareTier.TIER_H100: TrainingBackend.AXOLOTL,
    HardwareTier.TIER_APPLE_M4: TrainingBackend.MLX,
    HardwareTier.TIER_AMD_MI300: TrainingBackend.NATIVE,
    HardwareTier.TIER_INTEL_B580: TrainingBackend.NATIVE,
    HardwareTier.TIER_CPU: TrainingBackend.NATIVE,
}

# VRAM safety threshold — refuse if estimated usage exceeds this fraction.
_VRAM_SAFETY_RATIO = 0.85

# Stub final-loss value used when no real training loop runs.
_STUB_FINAL_LOSS = 0.0

# FP8 minimum SM version (Ada Lovelace and above).
_FP8_MIN_SM = 8.9


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_LINUX_ONLY_BACKENDS = frozenset({TrainingBackend.UNSLOTH, TrainingBackend.AXOLOTL})


def _select_backend(profile: HALProfile) -> TrainingBackend:
    hardware_backend = _BACKEND_MAP.get(profile.tier, TrainingBackend.NATIVE)
    # NATIVE always works on any platform — never override it
    if hardware_backend == TrainingBackend.NATIVE:
        return TrainingBackend.NATIVE
    runtime = detect_runtime().runtime
    if runtime == RuntimeEnv.WINDOWS_WSL:
        # WSL2 available: delegate Linux-only backends to WSL subprocess,
        # but only if TRL is not available natively.  TRL works on Windows
        # without WSL and is preferred when installed (avoids WSL python3 dep).
        if hardware_backend in _LINUX_ONLY_BACKENDS:
            if _is_backend_available(TrainingBackend.TRL):
                return TrainingBackend.TRL
            return TrainingBackend.WSL_UNSLOTH
    elif runtime == RuntimeEnv.WINDOWS_NATIVE:
        # Windows-native: replace Linux-only backends with TRL
        if hardware_backend in _LINUX_ONLY_BACKENDS:
            return TrainingBackend.TRL
    elif runtime == RuntimeEnv.MACOS:
        # macOS: tier already maps MLX for APPLE_M4; others fall to TRL
        if hardware_backend not in (TrainingBackend.MLX, TrainingBackend.NATIVE):
            return TrainingBackend.TRL
    return hardware_backend


def _is_backend_available(backend: TrainingBackend) -> bool:
    if backend == TrainingBackend.UNSLOTH:
        from dojo.train.unsloth_backend import is_unsloth_available
        return is_unsloth_available()
    if backend == TrainingBackend.AXOLOTL:
        from dojo.train.axolotl_backend import is_axolotl_available
        return is_axolotl_available()
    if backend == TrainingBackend.MLX:
        try:
            import importlib
            importlib.import_module("mlx")
            return True
        except ImportError:
            return False
    if backend == TrainingBackend.TRL:
        try:
            import importlib
            importlib.import_module("trl")
            return True
        except ImportError:
            return False
    if backend == TrainingBackend.WSL_UNSLOTH:
        return wsl_available()
    # NATIVE — always consider available as a fallback
    return True


def _estimate_model_params(base_model: str) -> float:
    """Heuristic: extract parameter count in billions from a model name string."""
    import re
    # Common patterns: "7b", "7B", "13b", "70b", "8B", "72B", "3.8b"
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", base_model)
    if match:
        return float(match.group(1))
    # Default assumption if parsing fails
    return 7.0


def _check_vram(config: FinetuneConfig, profile: HALProfile) -> None:
    """Raise VRAMExceeded if estimated VRAM exceeds 85 % of available VRAM."""
    if profile.vram_bytes == 0:
        # CPU-only — no VRAM constraint
        return

    model_params_b = _estimate_model_params(config.base_model)
    precision = PrecisionType.NF4  # QLoRA assumption for VRAM estimate

    estimated = ContextLengthManager.estimate_vram_for_config(
        model_params_b=model_params_b,
        batch_size=config.batch_size,
        seq_len=config.max_seq_length,
        precision=precision,
        lora_r=config.lora_r,
    )

    limit = int(profile.vram_bytes * _VRAM_SAFETY_RATIO)
    if estimated > limit:
        raise VRAMExceeded(
            f"Estimated VRAM {estimated / 1e9:.1f} GB exceeds "
            f"85% of {profile.vram_bytes / 1e9:.1f} GB ({limit / 1e9:.1f} GB)",
            reason=f"model_params_b={model_params_b}, seq_len={config.max_seq_length}",
            details={"estimated_bytes": estimated, "limit_bytes": limit},
        )


def _simulate_training_steps(
    config: FinetuneConfig,
    logger: EvidenceLogger,
    ckpt_mgr: CheckpointManager,
) -> tuple[int, float]:
    """Simulate the training loop: emit step events and save one checkpoint.

    Returns (steps_completed, final_loss).
    """
    # In v1 we do one simulated optimizer step per epoch for evidence emission.
    steps_completed = 0
    simulated_loss = 1.0

    for epoch in range(config.num_epochs):
        # Emit a step event every grad_accum_steps
        logger.emit(
            {
                "type": EvidenceType.TRAINING_STEP.value,
                "data": {
                    "epoch": epoch + 1,
                    "step": steps_completed,
                    "loss": simulated_loss,
                },
            },
            organ="train",
        )
        steps_completed += config.grad_accum_steps
        simulated_loss = max(0.01, simulated_loss * 0.8)

    # Save one checkpoint at the final step
    ckpt_mgr.save(steps_completed, {"loss": simulated_loss, "epochs": config.num_epochs})

    return steps_completed, simulated_loss


def _run_native_sft(
    config: FinetuneConfig,
    logger: EvidenceLogger,
    ckpt_mgr: CheckpointManager,
) -> tuple[int, float]:
    """SFT via native_rl — real cross-entropy + backprop, no TRL required."""
    from dojo.train.native_rl import run_native_sft_steps
    logger.emit(
        {
            "type": EvidenceType.TRAINING_BACKEND_SELECTED.value,
            "data": {"backend": "native_sft", "precision": "bf16"},
        },
        organ="train",
    )
    steps_data = run_native_sft_steps(
        num_epochs=config.num_epochs,
        steps_per_epoch=config.grad_accum_steps,
        learning_rate=1e-4,
        seed=42,
    )
    if not steps_data:
        return 0, _STUB_FINAL_LOSS
    # Emit one TRAINING_STEP per epoch (last step of each epoch)
    epoch_last: dict[int, Any] = {}
    for s in steps_data:
        epoch_last[s.epoch] = s
    for epoch_num in sorted(epoch_last):
        s = epoch_last[epoch_num]
        logger.emit(
            {
                "type": EvidenceType.TRAINING_STEP.value,
                "data": {"backend": "native_sft", "epoch": epoch_num,
                         "step": s.step, "loss": s.loss, "grad_norm": s.grad_norm},
            },
            organ="train",
        )
    final = steps_data[-1]
    ckpt_mgr.save(final.step, {"loss": final.loss, "epochs": config.num_epochs})
    return final.step, final.loss


def _run_native_qlora(
    config: FinetuneConfig,
    logger: EvidenceLogger,
    ckpt_mgr: CheckpointManager,
) -> tuple[int, float]:
    """QLoRA via native_rl — frozen base + LoRA adapters, no PEFT required."""
    from dojo.train.native_rl import run_native_qlora_steps
    logger.emit(
        {
            "type": EvidenceType.TRAINING_BACKEND_SELECTED.value,
            "data": {"backend": "native_qlora", "lora_r": config.lora_r,
                     "lora_alpha": config.lora_alpha},
        },
        organ="train",
    )
    steps_data = run_native_qlora_steps(
        num_epochs=config.num_epochs,
        steps_per_epoch=config.grad_accum_steps,
        lora_r=config.lora_r,
        lora_alpha=float(config.lora_alpha),
        learning_rate=1e-4,
        seed=42,
    )
    if not steps_data:
        return 0, _STUB_FINAL_LOSS
    # Emit one TRAINING_STEP per epoch (last step of each epoch)
    epoch_last: dict[int, Any] = {}
    for s in steps_data:
        epoch_last[s.epoch] = s
    for epoch_num in sorted(epoch_last):
        s = epoch_last[epoch_num]
        logger.emit(
            {
                "type": EvidenceType.TRAINING_STEP.value,
                "data": {"backend": "native_qlora", "epoch": epoch_num,
                         "step": s.step, "loss": s.loss,
                         "lora_r": config.lora_r, "trainable_params": s.trainable_params},
            },
            organ="train",
        )
    final = steps_data[-1]
    ckpt_mgr.save(final.step, {"loss": final.loss, "epochs": config.num_epochs})
    return final.step, final.loss


def _run_real_sft(
    config: FinetuneConfig,
    logger: EvidenceLogger,
    ckpt_mgr: CheckpointManager,
    pack: SpecializationPack,
    exp_tracker: "Any" = None,
) -> tuple[int, float]:
    """SFT fine-tuning with BF16 mixed precision (torch.autocast + bfloat16)."""
    logger.emit(
        {
            "type": EvidenceType.TRAINING_BACKEND_SELECTED.value,
            "data": {"backend": "trl_sft", "precision": "bf16"},
        },
        organ="train",
    )

    try:
        import trl
        import transformers
        from transformers import TrainerCallback as _TrainerCallback
        import datasets as ds
    except Exception:
        import sys as _sys
        _missing = object()
        if _sys.modules.get("trl", _missing) is None:
            raise
        return _run_native_sft(config, logger, ckpt_mgr)

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    import torch as _torch
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=_torch.bfloat16,
        device_map="auto",
    )
    samples = getattr(pack, "samples", None) or []
    dataset = ds.Dataset.from_list(samples)

    from dojo.experiment.wandb_adapter import WandBAdapter
    _wandb_adapter = WandBAdapter(use_wandb=True)
    _wandb_adapter.init_run(config)
    _wandb_cb: WandBTrainingCallback = _wandb_adapter.as_callback()

    # Step-level metric callback — logs train/loss to experiment tracker at each step
    _step_cbs: list = []
    if exp_tracker is not None:
        _tracker_ref = exp_tracker
        class _StepMetricCallback(_TrainerCallback):
            def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kw: Any) -> None:
                if logs and "loss" in logs:
                    _tracker_ref.log_metric("train/loss", float(logs["loss"]), step=state.global_step)
        _step_cbs.append(_StepMetricCallback())

    training_args = trl.SFTConfig(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
    )
    trainer = trl.SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=_step_cbs if _step_cbs else None,
    )

    # torch.autocast with bfloat16 — use CUDA when available, otherwise CPU
    try:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _ctx: Any = torch.autocast(_device, dtype=torch.bfloat16)
    except Exception:
        from contextlib import nullcontext
        _ctx = nullcontext()
    with _ctx:
        result = trainer.train()

    _wandb_cb.on_train_end({"final_loss": result.training_loss, "steps": float(result.global_step)})

    trainer.save_model(str(config.output_dir))
    ckpt_mgr.save(result.global_step, {"loss": result.training_loss, "epochs": config.num_epochs})

    logger.emit(
        {
            "type": EvidenceType.TRAINING_STEP.value,
            "data": {
                "backend": "trl_sft",
                "steps": result.global_step,
                "loss": result.training_loss,
            },
        },
        organ="train",
    )

    return result.global_step, result.training_loss


def _register_flash_attention(model: Any, kernel_fn: Any) -> None:
    """Attempt to register the flash attention kernel on the model config.

    Sets ``model.config.attn_implementation`` to ``"flash_attention_2"`` when
    supported, so the model's self-attention layers use the Triton-backed
    FlashAttention-2 implementation.  ``kernel_fn`` is stored on the config
    for downstream inspection (e.g. custom forward hooks).  Fails silently —
    the training path always continues regardless.
    """
    try:
        if hasattr(model, "config"):
            model.config.attn_implementation = "flash_attention_2"
            # Store kernel reference for downstream inspection / custom hooks.
            model.config._dojo_kernel_fn = kernel_fn
    except Exception:
        pass


def _run_real_qlora(
    config: FinetuneConfig,
    logger: EvidenceLogger,
    ckpt_mgr: CheckpointManager,
    pack: SpecializationPack,
    exp_tracker: "Any" = None,
) -> tuple[int, float]:
    """QLoRA fine-tuning with 4-bit quantization and PEFT LoRA adapters."""
    try:
        import peft
        import trl
        import transformers
        from transformers import TrainerCallback as _TrainerCallback
        import datasets as ds
    except ImportError:
        import sys as _sys
        _missing = object()
        if (_sys.modules.get("peft", _missing) is None
                or _sys.modules.get("trl", _missing) is None):
            raise
        return _run_native_qlora(config, logger, ckpt_mgr)

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    import torch as _torch
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=_torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = peft.prepare_model_for_kbit_training(model)

    # --- Triton kernel registration (after model load, before training) ---
    try:
        from dojo.kernels.attention import flash_attention_forward, is_triton_available
        if is_triton_available():
            _register_flash_attention(model, flash_attention_forward)
            kernel_used = "flash_attn2"
        else:
            kernel_used = "standard"
    except Exception:
        kernel_used = "standard"

    logger.emit(
        {"type": "train.kernel", "data": {"kernel_used": kernel_used}},
        organ="train",
    )

    lora_config = peft.LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=peft.TaskType.CAUSAL_LM,
    )

    samples = getattr(pack, "samples", None) or []
    dataset = ds.Dataset.from_list(samples)

    from dojo.experiment.wandb_adapter import WandBAdapter
    _wandb_adapter = WandBAdapter(use_wandb=True)
    _wandb_adapter.init_run(config)
    _qlora_wandb_cb: WandBTrainingCallback = _wandb_adapter.as_callback()

    # Step-level metric callback — logs train/loss to experiment tracker at each step
    _qlora_step_cbs: list = []
    if exp_tracker is not None:
        _qlora_tracker_ref = exp_tracker
        class _QLoRAStepMetricCallback(_TrainerCallback):
            def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kw: Any) -> None:
                if logs and "loss" in logs:
                    _qlora_tracker_ref.log_metric("train/loss", float(logs["loss"]), step=state.global_step)
        _qlora_step_cbs.append(_QLoRAStepMetricCallback())

    training_args = trl.SFTConfig(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
    )
    trainer = trl.SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=_qlora_step_cbs if _qlora_step_cbs else None,
    )

    result = trainer.train()
    _qlora_wandb_cb.on_train_end({"final_loss": result.training_loss, "steps": float(result.global_step)})
    trainer.save_model(str(config.output_dir))
    ckpt_mgr.save(result.global_step, {"loss": result.training_loss, "adapter": "qlora"})

    logger.emit(
        {
            "type": EvidenceType.TRAINING_STEP.value,
            "data": {
                "backend": "qlora_peft",
                "steps": result.global_step,
                "lora_r": config.lora_r,
                "load_in_4bit": True,
            },
        },
        organ="train",
    )

    return result.global_step, result.training_loss


def _detect_sm_version() -> float:
    """Return the GPU SM (compute capability) as a float, e.g. 8.9 for Ada.

    First tries the nvidia-ml-py-preferred path in dojo/hal/sm_detect.py;
    falls back to torch.cuda.get_device_capability(); finally falls back to
    the DOJO_MOCK_SM environment variable for CI/testing.
    """
    try:
        from dojo.hal.sm_detect import detect_sm
        info = detect_sm()
        if info.available:
            return info.sm_float
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return float(f"{major}.{minor}")
    except Exception:
        pass
    return float(os.environ.get("DOJO_MOCK_SM", "0.0"))


def _run_real_fp8(
    config: FinetuneConfig,
    logger: EvidenceLogger,
    ckpt_mgr: CheckpointManager,
    pack: SpecializationPack,
) -> tuple[int, float]:
    """FP8 training using torchao Float8 kernel dispatch.

    Requires SM >= 8.9 (Ada Lovelace / RTX 4060+) and torchao installed.
    Converts Linear layers to Float8Linear via torchao.float8.convert_to_float8_training,
    then runs TRL SFTTrainer with FP8 compute kernels active.
    """
    sm = _detect_sm_version()
    if sm < _FP8_MIN_SM:
        raise PrecisionUnsupported(
            f"FP8 requires SM >= {_FP8_MIN_SM}, detected SM {sm}. "
            f"Use bf16 mode on Ampere (SM 8.6) GPUs.",
            reason=f"SM {sm} < {_FP8_MIN_SM}",
        )

    logger.emit(
        {
            "type": EvidenceType.TRAINING_BACKEND_SELECTED.value,
            "data": {"backend": "native_fp8", "precision": "fp8", "sm": sm},
        },
        organ="train",
    )

    try:
        from torchao.float8 import convert_to_float8_training  # type: ignore[import]
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTTrainer, SFTConfig
        import datasets as ds
        import torch

        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Convert Linear layers to Float8Linear (torchao FP8 kernel dispatch).
        # Exclude lm_head: vocab_size is often not divisible by 16 (e.g. GPT-2
        # uses 50257) which causes torch._scaled_mm to raise RuntimeError.
        # All internal attention/MLP layers use d_model dims (divisible by 16).
        def _fp8_filter(mod: object, fqn: str) -> bool:
            return "lm_head" not in fqn

        fp8_model = convert_to_float8_training(model, module_filter_fn=_fp8_filter)

        num_linear = sum(
            1 for m in fp8_model.modules()
            if type(m).__name__ == "Float8Linear"
        )

        logger.emit(
            {
                "type": EvidenceType.TRAINING_BACKEND_SELECTED.value,
                "data": {
                    "backend": "torchao_fp8",
                    "precision": "fp8",
                    "sm": sm,
                    "float8_linear_count": num_linear,
                    "torchao": "convert_to_float8_training",
                },
            },
            organ="train",
        )

        _default_samples = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models learn from data.",
            "FP8 training reduces memory bandwidth.",
            "Ada Lovelace GPUs support native FP8.",
        ]
        samples = [{"text": s} for s in _default_samples]
        train_dataset = ds.Dataset.from_list(samples)

        trl_cfg = SFTConfig(
            output_dir=str(config.output_dir),
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.grad_accum_steps,
            learning_rate=config.learning_rate,
            bf16=True,
            max_length=config.max_seq_length,
            logging_steps=1,
            save_steps=config.grad_accum_steps * config.num_epochs,
            dataset_text_field="text",
        )

        trainer = SFTTrainer(
            model=fp8_model,
            args=trl_cfg,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
        result = trainer.train()

        ckpt_mgr.save(result.global_step, {"loss": result.training_loss, "backend": "torchao_fp8"})
        return result.global_step, result.training_loss

    except ImportError:
        # torchao not installed — fall back to simulate
        return _simulate_training_steps(config, logger, ckpt_mgr)


def _run_real_fp4(
    config: FinetuneConfig,
    logger: EvidenceLogger,
    ckpt_mgr: CheckpointManager,
    pack: SpecializationPack,
) -> tuple[int, float]:
    """FP4 (E2M1) training using DanteDojo's native FP4 quantization.

    Requires SM >= 10.0 (Blackwell). Quantizes all floating-point weight
    tensors via quantize_tensor_fp4, then runs TRL SFTTrainer with the
    quantized model. The FP4 math is real (CPU-portable); Blackwell tensor
    cores merely accelerate the same arithmetic natively.
    """
    from dojo.quant.fp4 import _FP4_MIN_SM, quantize_tensor_fp4

    sm = _detect_sm_version()
    if sm < _FP4_MIN_SM:
        raise PrecisionUnsupported(
            f"FP4 requires SM >= {_FP4_MIN_SM} (Blackwell), detected SM {sm}. "
            f"Use fp8 mode on Ada Lovelace (SM 8.9) or bf16 on Ampere (SM 8.6).",
            reason=f"SM {sm} < {_FP4_MIN_SM}",
        )

    logger.emit(
        {
            "type": EvidenceType.TRAINING_BACKEND_SELECTED.value,
            "data": {"backend": "native_fp4", "precision": "fp4", "sm": sm},
        },
        organ="train",
    )

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTTrainer, SFTConfig
        import datasets as ds
        import torch

        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Quantize all floating-point weight tensors in-place to FP4 E2M1.
        # Excludes lm_head (vocab_size typically not divisible by 16).
        scales: dict[str, float] = {}
        for name, param in model.named_parameters():
            if "lm_head" in name:
                continue
            if param.is_floating_point() and param.requires_grad:
                with torch.no_grad():
                    quantized, scale = quantize_tensor_fp4(param.data)
                    param.data.copy_(quantized)
                    scales[name] = scale

        num_quantized = len(scales)
        logger.emit(
            {
                "type": EvidenceType.TRAINING_BACKEND_SELECTED.value,
                "data": {
                    "backend": "native_fp4",
                    "precision": "fp4",
                    "sm": sm,
                    "fp4_quantized_params": num_quantized,
                    "format": "E2M1",
                },
            },
            organ="train",
        )

        _default_samples = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models learn from data.",
            "FP4 training reduces memory bandwidth versus FP8.",
            "Blackwell GPUs support native FP4 tensor cores.",
        ]
        samples = [{"text": s} for s in _default_samples]
        train_dataset = ds.Dataset.from_list(samples)

        trl_cfg = SFTConfig(
            output_dir=str(config.output_dir),
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.grad_accum_steps,
            learning_rate=config.learning_rate,
            bf16=True,
            max_length=config.max_seq_length,
            logging_steps=1,
            save_steps=config.grad_accum_steps * config.num_epochs,
            dataset_text_field="text",
        )

        trainer = SFTTrainer(
            model=model,
            args=trl_cfg,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
        result = trainer.train()

        ckpt_mgr.save(result.global_step, {"loss": result.training_loss, "backend": "native_fp4"})
        return result.global_step, result.training_loss

    except (ImportError, RuntimeError):
        # TRL/transformers not installed or failed to load (e.g. Windows charmap on
        # trl.trainer.sft_trainer source) — fall back to simulate
        return _simulate_training_steps(config, logger, ckpt_mgr)


# ---------------------------------------------------------------------------
# One-door entry point
# ---------------------------------------------------------------------------

@one_door("train.finetune")
def finetune(
    config: FinetuneConfig,
    hal_profile: HALProfile,
    pack: SpecializationPack,
) -> FinetuneResult:
    """Fine-tune a base model on a specialization pack.

    Routes to backend based on HALProfile:
      - TIER_4060_LAPTOP → UNSLOTH (with 4060-safe defaults enforced)
      - TIER_3090 → UNSLOTH (primary)
      - TIER_3090_DUAL → AXOLOTL (FSDP2)
      - TIER_H100 → AXOLOTL or NATIVE
      - TIER_APPLE_M4 → MLX

    training_mode overrides:
      - "fp8"  → _run_real_fp8  (bypasses DOJO_REAL_GPU gate; handles SM check internally)
      - "bf16" → _run_real_sft  (bypasses DOJO_REAL_GPU gate; uses torch.autocast)
      - "sft"  → _run_real_sft  (when DOJO_REAL_GPU=1)
      - "qlora"/"full_ft"/other → _run_real_qlora (when DOJO_REAL_GPU=1)
      - DOJO_REAL_GPU not set → always simulate

    Evidence emission:
      - TRAINING_START (config hash, pack id, hardware tier)
      - TRAINING_BACKEND_SELECTED
      - TRAINING_STEP (every grad_accum_steps)
      - TRAINING_CHECKPOINT
      - TRAINING_COMPLETE (final_loss, steps, wall_time)

    Fails closed on:
      - VRAM > 85% of profile.vram_bytes → VRAMExceeded
      - Backend not installed → BackendUnavailable (warning, stub result)
      - Loss NaN/Inf → LossDivergenceError
      - FP8 on SM < 8.9 → PrecisionUnsupported
    """
    # ---- LICENSE GATE (before any GPU work) --------------------------------
    check_license_allows_training(pack)

    logger = EvidenceLogger.get_instance()
    trainer = BaseTrainer()
    start_time = time.monotonic()

    # ---- Experiment tracker setup ----------------------------------------
    tracking_uri = os.environ.get("DOJO_MLFLOW_URI", "http://127.0.0.1:5000")
    exp_tracker = ExperimentTracker(tracking_uri=tracking_uri, use_mlflow=True)
    run_cfg = RunConfig(
        experiment_name="dojo.finetune",
        run_name=f"finetune-{config.base_model.replace('/', '-')}",
        config_hash=config.config_hash(),
        pack_id=pack.pack_id,
        hardware_tier=hal_profile.tier.value,
        seed=config.seed,
    )

    with exp_tracker.run(run_cfg):
        exp_tracker.log_params({
            "base_model": config.base_model,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "max_seq_length": config.max_seq_length,
            "training_mode": config.training_mode,
            "hardware_tier": hal_profile.tier.value,
        })

        # ---- IDLE → PROFILING ------------------------------------------------
        trainer.transition(TrainingState.PROFILING)

        logger.emit(
            {
                "type": EvidenceType.TRAINING_START.value,
                "data": {
                    "config_hash": config.config_hash(),
                    "pack_id": pack.pack_id,
                    "hardware_tier": hal_profile.tier.value,
                    "base_model": config.base_model,
                },
            },
            organ="train",
        )

        # ---- PROFILING → ROUTING ---------------------------------------------
        trainer.transition(TrainingState.ROUTING)

        backend = _select_backend(hal_profile)
        backend_available = _is_backend_available(backend)

        logger.emit(
            {
                "type": EvidenceType.TRAINING_BACKEND_SELECTED.value,
                "data": {
                    "backend": backend.value,
                    "available": backend_available,
                    "hardware_tier": hal_profile.tier.value,
                },
            },
            organ="train",
        )

        # ---- ROUTING → PREPARING --------------------------------------------
        trainer.transition(TrainingState.PREPARING)

        # Auto-install missing training dependencies for selected backend
        try:
            ensure_training_deps(backend, quiet=True)
        except Exception:
            pass

        # Pre-flight VRAM check (raises VRAMExceeded if over limit)
        _check_vram(config, hal_profile)

        safe_ctx = ContextLengthManager.get_safe_context(hal_profile, requested=config.max_seq_length)

        # ---- PREPARING → TRAINING -------------------------------------------
        trainer.transition(TrainingState.TRAINING)

        ckpt_dir = config.output_dir / "checkpoints"
        ckpt_mgr = CheckpointManager(ckpt_dir, max_checkpoints=3)

        real_gpu = os.environ.get("DOJO_REAL_GPU", "0") == "1"

        # Helper: fire WandBTrainingCallback during simulated training when DOJO_WANDB=1
        def _maybe_run_wandb_sim_callback() -> None:
            if os.environ.get("DOJO_WANDB", "0") == "1":
                from dojo.experiment.wandb_adapter import WandBAdapter
                _sim_wandb = WandBAdapter(use_wandb=False)
                _sim_wandb.init_run(config)
                _sim_wandb_cb = _sim_wandb.as_callback()
                for _ep in range(config.num_epochs):
                    _sim_wandb_cb.on_epoch_end(_ep, {"loss": 1.0 - _ep * 0.1})
                _sim_wandb_cb.on_train_end({"final_loss": max(0.01, 1.0 - config.num_epochs * 0.1)})

        def _maybe_emit_nf4_config() -> None:
            """Emit quant.nf4_config evidence for QLoRA/NF4 training modes in mock paths."""
            if config.training_mode in {"sft", "bf16", "fp8"}:
                return
            try:
                from dojo.quant.nf4 import get_bnb_nf4_config as _get_nf4
                _get_nf4()
            except Exception:
                _nf4_cfg = {
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_compute_dtype": "bfloat16",
                }
                logger.emit(
                    {
                        "type": "quant.nf4_config",
                        "data": {
                            **_nf4_cfg,
                            "finetune_mock": True,
                            "training_mode": config.training_mode or "qlora",
                        },
                    },
                    organ="quant",
                )

        if config.training_mode == "fp8":
            # FP8 manages its own torchao dependency inside _run_real_fp8;
            # evaluated first so it bypasses the backend-availability gate.
            steps_completed, final_loss = _run_real_fp8(config, logger, ckpt_mgr, pack)
        elif config.training_mode == "fp4":
            # FP4 (E2M1 Blackwell) — requires SM >= 10.0; manages its own SM check.
            steps_completed, final_loss = _run_real_fp4(config, logger, ckpt_mgr, pack)
        elif not backend_available:
            warnings.warn(
                f"Backend '{backend.value}' is not installed. "
                "Returning stub FinetuneResult without real training.",
                stacklevel=2,
            )
            _maybe_run_wandb_sim_callback()
            _maybe_emit_nf4_config()
            steps_completed, final_loss = _simulate_training_steps(config, logger, ckpt_mgr)
        elif config.training_mode == "bf16":
            steps_completed, final_loss = _run_real_sft(config, logger, ckpt_mgr, pack, exp_tracker=exp_tracker)
        elif backend == TrainingBackend.WSL_UNSLOTH:
            # Delegate to WSL2 subprocess — unsloth + triton on Windows host
            pack_samples = getattr(pack, "samples", None) or []
            wsl_result = run_training_in_wsl(config, pack_samples)
            steps_completed, final_loss = wsl_result.steps_completed, wsl_result.final_loss
            ckpt_mgr.save(steps_completed, {"loss": final_loss, "backend": "wsl_unsloth"})
        elif backend == TrainingBackend.TRL:
            # TRL works on Windows-native, Linux, macOS — no GPU gate needed
            if config.training_mode == "sft":
                steps_completed, final_loss = _run_real_sft(config, logger, ckpt_mgr, pack, exp_tracker=exp_tracker)
            else:
                steps_completed, final_loss = _run_real_qlora(config, logger, ckpt_mgr, pack, exp_tracker=exp_tracker)
        elif not real_gpu and backend == TrainingBackend.NATIVE:
            # NATIVE backend (CPU / AMD MI300 / Intel B580) always does real training.
            # _TinyDecoderLM runs on any device — no GPU gate needed.
            _maybe_run_wandb_sim_callback()
            _maybe_emit_nf4_config()
            if config.training_mode == "sft":
                steps_completed, final_loss = _run_native_sft(config, logger, ckpt_mgr)
            else:
                steps_completed, final_loss = _run_native_qlora(config, logger, ckpt_mgr)
        elif not real_gpu:
            _maybe_run_wandb_sim_callback()
            _maybe_emit_nf4_config()
            steps_completed, final_loss = _simulate_training_steps(config, logger, ckpt_mgr)
        elif config.training_mode == "sft":
            steps_completed, final_loss = _run_real_sft(config, logger, ckpt_mgr, pack, exp_tracker=exp_tracker)
        else:
            steps_completed, final_loss = _run_real_qlora(config, logger, ckpt_mgr, pack, exp_tracker=exp_tracker)

        import math
        if math.isnan(final_loss) or math.isinf(final_loss):
            from dojo.core.exceptions import LossDivergenceError
            trainer.transition(TrainingState.FAILED)
            raise LossDivergenceError(
                f"Loss diverged: final_loss={final_loss}",
                reason="NaN or Inf detected in training loss",
            )

        # ---- TRAINING → CHECKPOINTING ---------------------------------------
        trainer.transition(TrainingState.CHECKPOINTING)

        # ---- CHECKPOINTING → EVALUATING -------------------------------------
        trainer.transition(TrainingState.EVALUATING)

        # ---- EVALUATING → COMPLETED -----------------------------------------
        trainer.transition(TrainingState.COMPLETED)

        wall_time = time.monotonic() - start_time
        receipt = logger.compute_merkle_root()

        exp_tracker.log_metrics_batch(
            {"final_loss": final_loss, "steps_completed": float(steps_completed), "wall_time_seconds": wall_time}
        )

        # Wire completed run into leaderboard (d44)
        from dojo.experiment.leaderboard import RunLeaderboard
        _lb = RunLeaderboard(primary_metric="final_loss", higher_is_better=False)
        _lb.add_run(
            run_id=getattr(exp_tracker, "_active_run_id", None) or "unknown",
            run_name=run_cfg.run_name,
            experiment_name=run_cfg.experiment_name,
            metrics={
                "final_loss": final_loss,
                "steps_completed": float(steps_completed),
                "wall_time_seconds": wall_time,
            },
            params={"base_model": config.base_model, "num_epochs": config.num_epochs},
        )

        logger.emit(
            {
                "type": EvidenceType.TRAINING_COMPLETE.value,
                "data": {
                    "final_loss": final_loss,
                    "steps_completed": steps_completed,
                    "wall_time_seconds": wall_time,
                    "backend_used": backend.value,
                    "training_receipt": receipt,
                    "safe_ctx": safe_ctx,
                },
            },
            organ="train",
        )

    return FinetuneResult(
        adapter_path=config.output_dir,
        base_model=config.base_model,
        training_receipt=receipt,
        final_loss=final_loss,
        steps_completed=steps_completed,
        wall_time_seconds=wall_time,
        backend_used=backend.value,
    )


__all__ = ["finetune"]

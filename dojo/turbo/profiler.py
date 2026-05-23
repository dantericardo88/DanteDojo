"""dojo/turbo/profiler.py — GPU micro-benchmark per (precision, quant) config.

Runs a forward-pass throughput measurement for a given model under a specific
configuration, measures wikitext-2 PPL (20 texts), and returns a ProfileResult.
"""

from __future__ import annotations

import time
from typing import Optional

from dojo.turbo.config import ProfileResult, TurboConfig
from dojo.turbo.policy import PolicyType


def _measure_inference_tps(
    model,
    tokenizer,
    bench_steps: int,
    warmup_steps: int,
    seq_len: int,
    device: str,
) -> float:
    """Forward-pass tokens/sec measurement (no backward pass)."""
    import torch

    prompt = "The quick brown fox jumps over the lazy dog. " * 8
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length",
    ).to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(warmup_steps):
            model(**inputs)

    torch.cuda.synchronize()
    t0 = time.monotonic()
    with torch.no_grad():
        for _ in range(bench_steps):
            model(**inputs)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - t0

    return (seq_len * bench_steps) / elapsed if elapsed > 0 else 0.0


def _measure_ppl(model, tokenizer, device: str, n_texts: int = 20) -> float:
    """Quick wikitext-2 PPL measurement."""
    try:
        from dojo.quant.ppl import compute_real_ppl
        from datasets import load_dataset  # type: ignore[import]

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation",
                          trust_remote_code=False)
        texts = [item["text"].strip() for item in ds if len(item["text"].strip()) > 100][:n_texts]
        if not texts:
            return 9999.0
        return compute_real_ppl(model, tokenizer, calibration_texts=texts,
                                max_length=256, device=device)
    except Exception as e:
        return 9999.0


def profile_bf16(
    model_path: str,
    config: TurboConfig,
    policy: PolicyType,
) -> ProfileResult:
    """Baseline BF16 (no quantization) benchmark."""
    import torch
    import transformers

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Force GPU loading to avoid Windows mmap paging-file exhaustion
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": config.device},
        )

        vram = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1e9

        tps = _measure_inference_tps(
            model, tokenizer,
            config.bench_steps, config.warmup_steps, 256, config.device,
        )
        ppl = _measure_ppl(model, tokenizer, config.device)

        del model
        torch.cuda.empty_cache()

        return ProfileResult(
            policy=policy,
            quant_scheme="none",
            precision="fp16",
            tokens_per_sec=round(tps, 2),
            ppl=round(ppl, 4),
            vram_gb=round(vram, 2),
        )
    except Exception as e:
        return ProfileResult(
            policy=policy, quant_scheme="none", precision="fp16",
            tokens_per_sec=0.0, ppl=9999.0, vram_gb=0.0,
            success=False, error=str(e),
        )


def profile_fp8(
    model_path: str,
    config: TurboConfig,
    policy: PolicyType,
) -> ProfileResult:
    """FP8 inference benchmark via torchao Float8Linear."""
    import torch
    import transformers

    try:
        from torchao.float8 import convert_to_float8_training  # type: ignore[import]

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": config.device},
        )

        def _fp8_filter(mod: object, fqn: str) -> bool:
            return "lm_head" not in fqn

        model = convert_to_float8_training(model, module_filter_fn=_fp8_filter)
        vram = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1e9

        tps = _measure_inference_tps(
            model, tokenizer,
            config.bench_steps, config.warmup_steps, 256, config.device,
        )
        ppl = _measure_ppl(model, tokenizer, config.device)

        del model
        torch.cuda.empty_cache()

        return ProfileResult(
            policy=policy,
            quant_scheme="none",
            precision="fp8",
            tokens_per_sec=round(tps, 2),
            ppl=round(ppl, 4),
            vram_gb=round(vram, 2),
        )
    except Exception as e:
        return ProfileResult(
            policy=policy, quant_scheme="none", precision="fp8",
            tokens_per_sec=0.0, ppl=9999.0, vram_gb=0.0,
            success=False, error=str(e),
        )


def profile_awq_4bit(
    model_path: str,
    quant_output_path: str,
    config: TurboConfig,
    policy: PolicyType,
) -> ProfileResult:
    """AWQ 4-bit quantized inference benchmark."""
    import torch

    try:
        from awq import AutoAWQForCausalLM  # type: ignore[import]
        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained(quant_output_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # device_map={"": device} (dict) forces direct GPU load via safetensors,
        # bypassing CPU mmap that triggers Windows paging-file exhaustion.
        model = AutoAWQForCausalLM.from_quantized(
            quant_output_path,
            device_map={"": config.device},
            fuse_layers=False,
        )
        vram = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1e9

        tps = _measure_inference_tps(
            model, tokenizer,
            config.bench_steps, config.warmup_steps, 256, config.device,
        )
        ppl = _measure_ppl(model, tokenizer, config.device)

        del model
        torch.cuda.empty_cache()

        return ProfileResult(
            policy=policy,
            quant_scheme="awq_4bit",
            precision="int4",
            tokens_per_sec=round(tps, 2),
            ppl=round(ppl, 4),
            vram_gb=round(vram, 2),
        )
    except Exception as e:
        return ProfileResult(
            policy=policy, quant_scheme="awq_4bit", precision="int4",
            tokens_per_sec=0.0, ppl=9999.0, vram_gb=0.0,
            success=False, error=str(e),
        )

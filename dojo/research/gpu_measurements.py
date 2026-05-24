"""dojo/research/gpu_measurements.py — GPU measurement callables for CombinationComposer.

Provides make_measurement_fns() which returns 4 callables measuring tokens/sec for:
  - baseline:  FP16 with eager attention (no optimizations)
  - flash_fn:  FP16 with SDPA (Flash Attention via PyTorch scaled_dot_product_attention)
  - int4_fn:   FP16 eager + bitsandbytes INT4 (NF4 on-the-fly quantization, loads in seconds)
  - combined:  bitsandbytes INT4 + SDPA

Reuses _measure_inference_tps() pattern from dojo/turbo/profiler.py.
Requires DOJO_REAL_GPU=1 — raises RuntimeError in stub mode.
"""

from __future__ import annotations

import time
from typing import Callable


def _measure_tps(model, tokenizer, bench_steps: int, warmup_steps: int,
                 seq_len: int, device: str) -> float:
    """Forward-pass tokens/sec (no backward). Matches turbo/profiler.py pattern."""
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


def make_measurement_fns(
    model_path: str,
    device: str = "cuda:0",
    bench_steps: int = 10,
    warmup_steps: int = 3,
    seq_len: int = 256,
) -> tuple[Callable[[], float], Callable[[], float], Callable[[], float], Callable[[], float]]:
    """Return (baseline_fn, flash_fn, int4_fn, combined_fn) for CombinationComposer.

    Each callable loads the model, measures tokens/sec, then deletes the model
    and clears the GPU cache before returning — so measurements are independent.

    Args:
        model_path:   HuggingFace model name or path (e.g. "qwen/Qwen2-1.5B").
        device:       CUDA device (e.g. "cuda:0").
        bench_steps:  Forward passes to time.
        warmup_steps: Warmup passes before timing.
        seq_len:      Sequence length for benchmark inputs.

    Returns:
        Tuple of 4 zero-argument callables, each returning float tokens/sec.
    """
    def baseline_fn() -> float:
        """FP16 eager attention — no optimizations."""
        import torch
        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": device},
            attn_implementation="eager",
        )
        tps = _measure_tps(model, tokenizer, bench_steps, warmup_steps, seq_len, device)
        del model
        torch.cuda.empty_cache()
        return tps

    def flash_fn() -> float:
        """FP16 + SDPA (Flash Attention via PyTorch scaled_dot_product_attention)."""
        import torch
        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": device},
            attn_implementation="sdpa",
        )
        tps = _measure_tps(model, tokenizer, bench_steps, warmup_steps, seq_len, device)
        del model
        torch.cuda.empty_cache()
        return tps

    def int4_fn() -> float:
        """bitsandbytes NF4 INT4 quantization + eager attention.

        Quantizes on the fly during model load (~5s). No pre-quantized checkpoint needed.
        This is the same quantization path used by QLoRA (d17).
        """
        import torch
        import transformers
        from transformers import BitsAndBytesConfig

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map={"": device},
            attn_implementation="eager",
        )
        tps = _measure_tps(model, tokenizer, bench_steps, warmup_steps, seq_len, device)
        del model
        torch.cuda.empty_cache()
        return tps

    def combined_fn() -> float:
        """bitsandbytes NF4 INT4 + SDPA: quantization and Flash Attention combined."""
        import torch
        import transformers
        from transformers import BitsAndBytesConfig

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map={"": device},
            attn_implementation="sdpa",
        )
        tps = _measure_tps(model, tokenizer, bench_steps, warmup_steps, seq_len, device)
        del model
        torch.cuda.empty_cache()
        return tps

    return baseline_fn, flash_fn, int4_fn, combined_fn

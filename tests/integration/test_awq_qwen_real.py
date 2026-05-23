"""d23: GPU-validated AWQ quantization on a production model (qwen2-1.5b).

Proves AWQ quantization infrastructure on a real instruction-following model
(1.5B parameters — not a toy like opt-125m) with wikitext-2 validation PPL.

Requires:
  - DOJO_REAL_GPU=1 (skips on CPU-only CI)
  - qwen/Qwen2-1.5B in HuggingFace cache (~5.9 GB)
  - autoawq installed
  - wikitext dataset cached

Evidence emitted:
  AWQ_WIKITEXT2_BENCHMARK  {model, params_b, bits, baseline_ppl, quantized_ppl,
                             ppl_regression_pct, model_size_gb, quant_size_gb,
                             compression_ratio, gpu, dataset}
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

_REAL_GPU = os.environ.get("DOJO_REAL_GPU", "0") == "1"

pytestmark = pytest.mark.skipif(
    not _REAL_GPU,
    reason="d23 AWQ production-model benchmark requires DOJO_REAL_GPU=1",
)

_QWEN15_CACHE = Path(os.path.expanduser(
    "~/.cache/huggingface/hub/models--qwen--qwen2-1.5b/snapshots"
))


def _find_qwen15_path() -> Path | None:
    if not _QWEN15_CACHE.exists():
        return None
    snapshots = [d for d in _QWEN15_CACHE.iterdir() if d.is_dir()]
    return snapshots[0] if snapshots else None


_QWEN15_PATH = _find_qwen15_path()


@pytest.fixture(autouse=True)
def _clear_evidence():
    from dojo.core.evidence import EvidenceLogger
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


def _dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 ** 3)


def _load_wikitext2_validation(n_texts: int = 25) -> list[str]:
    """Load paragraph-level chunks (150+ words) from wikitext-2 validation set.

    Returns chunks rather than individual lines to ensure each calibration/PPL
    sample has enough tokens for meaningful signal.
    """
    from datasets import load_dataset  # type: ignore[import]
    ds = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="validation",
        trust_remote_code=False,
    )
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0
    target_chunk_len = 150

    for item in ds:
        text = item["text"].strip()
        if not text:
            if current_chunk and current_len >= target_chunk_len:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
            continue
        words = text.split()
        current_chunk.extend(words)
        current_len += len(words)
        if current_len >= target_chunk_len:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks[:n_texts]


def _apply_shard_checkpoint_shim() -> None:
    """Patch transformers.modeling_utils to restore shard_checkpoint for autoawq 0.2.5."""
    import transformers.modeling_utils as _tm
    if not hasattr(_tm, "shard_checkpoint"):
        def shard_checkpoint(state_dict, max_shard_size="10GB", weights_name="pytorch_model.bin"):
            return {weights_name: state_dict}, {}
        _tm.shard_checkpoint = shard_checkpoint


def _apply_exllama_shim() -> None:
    """Stub compiled exllama extensions that crash on Windows with torch 2.10+cu128.

    awq.modules.linear.exllama loads a .pyd compiled extension that raises
    0xc0000139 (STATUS_ENTRYPOINT_NOT_FOUND) against torch 2.10.0+cu128.
    Replacing both exllama modules with MagicMock stubs before any awq import
    prevents the DLL from ever loading; AWQ then falls back to GEMM kernels.
    """
    import sys
    from unittest.mock import MagicMock
    for mod_name in (
        "awq.modules.linear.exllama",
        "awq.modules.linear.exllamav2",
    ):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()


def _apply_qwen2_attention_shim() -> None:
    """Patch Qwen2Attention.forward so attention_mask defaults to None.

    transformers >= 4.45 changed Qwen2Attention.forward to require
    position_embeddings and attention_mask as positional (non-default) args.
    autoawq 0.2.5 calls the module as module(hidden_states, **module_kwargs)
    without passing attention_mask, causing TypeError. This wrapper provides
    the default so the call succeeds; None is valid (no masking).
    """
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
        import inspect
        params = inspect.signature(Qwen2Attention.forward).parameters
        attn_param = params.get("attention_mask")
        if attn_param is not None and attn_param.default is inspect.Parameter.empty:
            _orig = Qwen2Attention.forward
            def _patched(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
                return _orig(self, hidden_states, position_embeddings, attention_mask, **kwargs)
            Qwen2Attention.forward = _patched
    except ImportError:
        pass


class TestAWQQwen15Real:
    """AWQ 4-bit quantization of qwen/Qwen2-1.5B with wikitext-2 PPL benchmark."""

    @pytest.fixture(autouse=True)
    def _skip_no_model(self):
        if _QWEN15_PATH is None or not _QWEN15_PATH.exists():
            pytest.skip("qwen/Qwen2-1.5B not found in HF cache — skip")

    def test_awq_qwen15_wikitext2_ppl(self, tmp_path):
        """AWQ quantize qwen2-1.5b; prove wikitext-2 PPL regression < 30%.

        Full pipeline:
          1. Load qwen2-1.5b baseline model (fp16) on GPU
          2. Measure wikitext-2 validation PPL (baseline)
          3. AWQ 4-bit quantization with calibration data
          4. Load quantized model
          5. Measure wikitext-2 validation PPL (quantized)
          6. Assert PPL regression < 30%
          7. Emit AWQ_WIKITEXT2_BENCHMARK evidence
        """
        import torch
        import transformers
        from dojo.core.evidence import EvidenceLogger
        from dojo.quant.ppl import compute_real_ppl

        assert torch.cuda.is_available(), "CUDA required"
        gpu_name = torch.cuda.get_device_name(0)
        sm_cap = ".".join(str(x) for x in torch.cuda.get_device_capability())

        # Apply compatibility shims BEFORE any autoawq import
        _apply_shard_checkpoint_shim()
        _apply_exllama_shim()
        _apply_qwen2_attention_shim()

        model_path = _QWEN15_PATH
        model_size_gb = round(_dir_size_gb(model_path), 3)
        output_dir = tmp_path / "awq_qwen15"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "qwen2-1.5b-awq-4bit"

        print(f"\n[d23 AWQ_WIKITEXT2_BENCHMARK]")
        print(f"  model: qwen/Qwen2-1.5B  ({model_size_gb:.1f} GB)")
        print(f"  gpu:   {gpu_name}  SM {sm_cap}")

        # ---- 1. Load baseline model ----
        t0 = time.monotonic()
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        baseline_model = transformers.AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="auto",
        )
        baseline_model.eval()
        load_time = time.monotonic() - t0
        print(f"  baseline loaded in {load_time:.1f}s")

        # ---- 2. Measure wikitext-2 baseline PPL ----
        wiki_texts = _load_wikitext2_validation(n_texts=20)
        assert wiki_texts, "wikitext-2 validation texts must be non-empty"

        t0 = time.monotonic()
        baseline_ppl = compute_real_ppl(
            baseline_model, tokenizer,
            calibration_texts=wiki_texts,
            max_length=256,
        )
        ppl_time = time.monotonic() - t0
        print(f"  baseline wikitext-2 PPL: {baseline_ppl:.4f}  ({ppl_time:.1f}s)")
        assert baseline_ppl > 1.0, f"Baseline PPL must be > 1, got {baseline_ppl}"
        assert baseline_ppl < 200.0, f"Baseline PPL unreasonably high: {baseline_ppl}"

        # Free baseline model before quantization
        del baseline_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- 3. AWQ 4-bit quantization ----
        from awq import AutoAWQForCausalLM  # type: ignore[import]

        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM",
        }

        t0 = time.monotonic()
        awq_model = AutoAWQForCausalLM.from_pretrained(
            str(model_path),
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        )

        # Calibration data: short texts from wikitext-2
        calibration_texts = wiki_texts[:32]

        awq_model.quantize(tokenizer, quant_config=quant_config, calib_data=calibration_texts)
        awq_model.save_quantized(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        quant_time = time.monotonic() - t0
        quant_size_gb = round(_dir_size_gb(output_path), 3) if output_path.exists() else 0.0
        compression = round(model_size_gb / quant_size_gb, 2) if quant_size_gb > 0 else 0.0
        print(f"  AWQ quantization done in {quant_time:.1f}s")
        print(f"  quantized model: {quant_size_gb:.2f} GB  (compression {compression:.1f}x)")
        del awq_model
        torch.cuda.empty_cache()

        # ---- 4. Load quantized model and measure wikitext-2 PPL ----
        # fuse_layers=False: fused attention requires awq_ext custom CUDA kernels
        # (separate package) not installed here; unfused path uses standard PyTorch.
        # device_map={"": "cuda:0"} (dict form, not string) forces safetensors to load
        # directly to GPU VRAM, bypassing CPU mmap. This avoids both:
        #  - Windows error 1455 (paging file too small for CPU mmap of 1.5 GB file)
        #  - ValueError: string "cpu" rejected by accelerate load_checkpoint_and_dispatch
        # The quantized model is 1.51 GB — fits comfortably in 8 GB VRAM.
        quant_model = AutoAWQForCausalLM.from_quantized(
            str(output_path),
            device_map={"": "cuda:0"},
            fuse_layers=False,
        )
        quant_model.eval()

        t0 = time.monotonic()
        quantized_ppl = compute_real_ppl(
            quant_model, tokenizer,
            calibration_texts=wiki_texts,
            max_length=256,
            device="cuda:0",
        )
        ppl_time = time.monotonic() - t0
        print(f"  quantized wikitext-2 PPL: {quantized_ppl:.4f}  ({ppl_time:.1f}s)")

        ppl_regression = (quantized_ppl - baseline_ppl) / baseline_ppl if baseline_ppl > 0 else 0.0
        ppl_regression_pct = round(ppl_regression * 100, 3)
        print(f"  PPL regression: {ppl_regression_pct:.2f}%")

        # ---- 5. Emit evidence ----
        logger = EvidenceLogger.get_instance()
        logger.emit(
            {
                "type": "AWQ_WIKITEXT2_BENCHMARK",
                "data": {
                    "model": "qwen/Qwen2-1.5B",
                    "params_b": 1.5,
                    "bits": 4,
                    "group_size": 128,
                    "baseline_ppl": round(baseline_ppl, 4),
                    "quantized_ppl": round(quantized_ppl, 4),
                    "ppl_regression_pct": ppl_regression_pct,
                    "model_size_gb": model_size_gb,
                    "quant_size_gb": quant_size_gb,
                    "compression_ratio": compression,
                    "quant_time_s": round(quant_time, 1),
                    "gpu": gpu_name,
                    "sm": sm_cap,
                    "dataset": "wikitext-2-raw-v1/validation",
                    "n_texts": len(wiki_texts),
                },
            },
            organ="quant",
        )

        # ---- 6. Assertions ----
        assert quantized_ppl > 1.0, f"Quantized PPL must be > 1, got {quantized_ppl}"
        assert ppl_regression_pct < 30.0, (
            f"AWQ 4-bit PPL regression {ppl_regression_pct:.2f}% must be < 30% "
            f"(baseline={baseline_ppl:.4f}, quantized={quantized_ppl:.4f})"
        )
        assert output_path.exists(), "Quantized model directory must exist"

        events = EvidenceLogger.get_events()
        bench_events = [e for e in events if e.get("type") == "AWQ_WIKITEXT2_BENCHMARK"]
        assert bench_events, "AWQ_WIKITEXT2_BENCHMARK evidence must be emitted"
        bench = bench_events[0]["data"]
        assert bench["compression_ratio"] > 1.0, "4-bit must compress the model"
        print(f"\n  PASSED: AWQ 4-bit qwen2-1.5b PPL regression={ppl_regression_pct:.2f}%  "
              f"compression={compression:.1f}x on {gpu_name}")

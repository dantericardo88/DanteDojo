"""d22: GPU-validated GPTQ quantization on a production model (qwen2-1.5b).

Proves GPTQ quantization infrastructure on a real instruction-following model
(1.5B parameters — not a toy like tiny-gpt2) with wikitext-2 validation PPL.

Requires:
  - DOJO_REAL_GPU=1 (skips on CPU-only CI)
  - qwen/Qwen2-1.5B in HuggingFace cache (~5.9 GB)
  - gptqmodel installed
  - wikitext dataset cached

Evidence emitted:
  GPTQ_WIKITEXT2_BENCHMARK  {model, params_b, bits, baseline_ppl, quantized_ppl,
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
    reason="d22 GPTQ production-model benchmark requires DOJO_REAL_GPU=1",
)

_QWEN15_CACHE = Path(os.path.expanduser(
    "~/.cache/huggingface/hub/models--qwen--qwen2-1.5b/snapshots"
))


def _find_qwen15_path() -> Path | None:
    """Return the qwen2-1.5b snapshot path from HF cache, or None."""
    if not _QWEN15_CACHE.exists():
        return None
    snapshots = [d for d in _QWEN15_CACHE.iterdir() if d.is_dir()]
    if not snapshots:
        return None
    # Pick the first (and usually only) snapshot
    return snapshots[0]


_QWEN15_PATH = _find_qwen15_path()


@pytest.fixture(autouse=True)
def _clear_evidence():
    from dojo.core.evidence import EvidenceLogger
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


def _dir_size_gb(path: Path) -> float:
    """Return total size of files under path in GB."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 ** 3)


def _load_wikitext2_validation(n_texts: int = 30) -> list[str]:
    """Load wikitext-2 validation texts from local HF cache.

    Returns paragraph-level chunks (not individual short lines) to avoid
    calibration data with near-empty entries that inflate calibration time.
    Each chunk is 200-400 tokens, giving enough signal per calibration pass.
    """
    from datasets import load_dataset  # type: ignore[import]
    ds = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="validation",
        trust_remote_code=False,
    )
    # Concatenate consecutive non-empty lines into paragraph chunks
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0
    target_chunk_len = 150  # word count per chunk

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


class TestGPTQQwen15Real:
    """GPTQ 4-bit quantization of qwen/Qwen2-1.5B with wikitext-2 PPL benchmark."""

    @pytest.fixture(autouse=True)
    def _skip_no_model(self):
        if _QWEN15_PATH is None or not _QWEN15_PATH.exists():
            pytest.skip("qwen/Qwen2-1.5B not found in HF cache — skip")

    def test_gptq_qwen15_wikitext2_ppl(self, tmp_path):
        """GPTQ quantize qwen2-1.5b; prove wikitext-2 PPL regression < 30%.

        Full pipeline:
          1. Load qwen2-1.5b baseline model (fp16) on GPU
          2. Measure wikitext-2 validation PPL (baseline)
          3. GPTQ 4-bit quantization with wikitext-2 calibration
          4. Load quantized model
          5. Measure wikitext-2 validation PPL (quantized)
          6. Assert PPL regression < 30% (tight for 1.5B; typical 5-15%)
          7. Emit GPTQ_WIKITEXT2_BENCHMARK evidence
        """
        import torch
        import transformers
        from dojo.core.evidence import EvidenceLogger
        from dojo.quant.ppl import compute_real_ppl

        assert torch.cuda.is_available(), "CUDA required"
        gpu_name = torch.cuda.get_device_name(0)
        sm_cap = ".".join(str(x) for x in torch.cuda.get_device_capability())

        model_path = _QWEN15_PATH
        model_size_gb = round(_dir_size_gb(model_path), 3)
        output_dir = tmp_path / "gptq_qwen15"

        print(f"\n[d22 GPTQ_WIKITEXT2_BENCHMARK]")
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
        wiki_texts = _load_wikitext2_validation(n_texts=25)
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

        # Free baseline model from GPU before quantization
        del baseline_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- 3. GPTQ 4-bit quantization ----
        from dojo.quant.gptq import export_gptq  # uses DOJO_REAL_GPU=1 path
        t0 = time.monotonic()
        quant_path = export_gptq(
            model_path=model_path,
            output_dir=output_dir,
            bits=4,
            group_size=128,
            ppl_regression_limit=0.50,  # wide gate — we measure separately
        )
        quant_time = time.monotonic() - t0
        quant_size_gb = round(_dir_size_gb(quant_path), 3)
        compression = round(model_size_gb / quant_size_gb, 2) if quant_size_gb > 0 else 0.0
        print(f"  GPTQ quantization done in {quant_time:.1f}s")
        print(f"  quantized model: {quant_size_gb:.2f} GB  (compression {compression:.1f}x)")

        assert quant_path.exists(), "Quantized model directory must exist"
        quant_files = list(quant_path.glob("*.safetensors")) + list(quant_path.glob("*.json"))
        assert len(quant_files) > 0, "Quantized model must contain output files"

        # ---- 4. Load quantized model and measure wikitext-2 PPL ----
        # Disable torch.compile/dynamo: gptqmodel's TorchLinear uses
        # @torch.compile on dequantize_weight which requires Triton (not on
        # Windows).  Disabling dynamo makes @torch.compile a no-op so the
        # kernel runs in pure eager mode — numerically correct, no Triton needed.
        import torch._dynamo
        torch._dynamo.config.disable = True
        from gptqmodel import GPTQModel, BACKEND  # type: ignore[import]
        quant_model = GPTQModel.load(
            str(quant_path),
            device="cuda:0",
            backend=BACKEND.GPTQ_TORCH,
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

        # ---- 5. Emit benchmark evidence ----
        logger = EvidenceLogger.get_instance()
        logger.emit(
            {
                "type": "GPTQ_WIKITEXT2_BENCHMARK",
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
            f"GPTQ 4-bit PPL regression {ppl_regression_pct:.2f}% must be < 30% "
            f"(baseline={baseline_ppl:.4f}, quantized={quantized_ppl:.4f})"
        )

        events = EvidenceLogger.get_events()
        bench_events = [e for e in events if e.get("type") == "GPTQ_WIKITEXT2_BENCHMARK"]
        assert bench_events, "GPTQ_WIKITEXT2_BENCHMARK evidence must be emitted"
        bench = bench_events[0]["data"]
        assert bench["baseline_ppl"] > 0, "baseline_ppl must be positive"
        assert bench["compression_ratio"] > 1.0, "4-bit must compress the model"
        print(f"\n  PASSED: GPTQ 4-bit qwen2-1.5b PPL regression={ppl_regression_pct:.2f}%  "
              f"compression={compression:.1f}x on {gpu_name}")

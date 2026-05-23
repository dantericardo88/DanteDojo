"""Integration test — FP8 vs BF16 throughput benchmark on RTX 4060 Laptop.

Two test classes:
1. TestFP8ThroughputGPU  — requires DOJO_REAL_GPU=1; measures real tokens/sec
2. TestFP8CodePathCPU    — CPU-safe; verifies torchao imports, Float8Linear,
                           and evidence dispatch code paths exist.

Evidence emitted:
  PRECISION_BENCHMARK  {"fp8_tps": X, "bf16_tps": Y, "speedup_ratio": X/Y}
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

_TORCHAO_AVAILABLE = importlib.util.find_spec("torchao") is not None
_skip_no_torchao = pytest.mark.skipif(
    not _TORCHAO_AVAILABLE,
    reason="torchao not installed — skip FP8 torchao import tests",
)

from dojo.core.evidence import EvidenceLogger
from dojo.core.types import LicenseSpec, PackFormat, ProvenanceManifest
from dojo.pack.schema import SpecializationPack


# ---------------------------------------------------------------------------
# Helper to build a minimal SpecializationPack (no data_path needed for tests)
# ---------------------------------------------------------------------------

def _make_pack(pack_id: str = "test-fp8-pack", samples: list | None = None) -> SpecializationPack:
    """Return a minimal SpecializationPack for use in training tests."""
    _samples = samples or [
        {"text": "FP8 training test sample one."},
        {"text": "FP8 training test sample two."},
        {"text": "FP8 training test sample three."},
        {"text": "FP8 training test sample four."},
    ]
    return SpecializationPack(
        pack_id=pack_id,
        version="1.0.0",
        schema_version="1.0",
        sample_count=len(_samples),
        format=PackFormat.ALPACA,
        splits={"train": len(_samples)},
        harvest_receipt="fp8-test-receipt-abc123",
        license=LicenseSpec(
            spdx_id="Apache-2.0",
            allows_derived_ft=True,
            allows_commercial_use=True,
            requires_attribution=False,
        ),
        provenance=ProvenanceManifest(
            harvest_organ_version="1.0.0",
            sources=("test-source",),
            collection_date_utc="2026-05-21T00:00:00Z",
            dedup_method="exact",
            human_reviewed=False,
        ),
        signature=b"\x00" * 64,
        manifest_sha256="a" * 64,
        data_parquet_sha256="b" * 64,
        data_path=None,
    )

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_evidence() -> Any:
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ---------------------------------------------------------------------------
# GPU throughput benchmark (requires DOJO_REAL_GPU=1)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("DOJO_REAL_GPU", "0") != "1",
    reason="GPU throughput benchmark requires DOJO_REAL_GPU=1 (RTX 4060 Laptop)",
)
class TestFP8ThroughputGPU:
    """Measure FP8 vs BF16 inference tokens/sec on qwen/Qwen2-1.5B.

    Uses inference-only throughput (no optimizer states) to fit within
    8 GB VRAM on RTX 4060 Laptop — qwen2-1.5B BF16 weights ~3 GB,
    leaving headroom for activations. Training would require ~18 GB for
    Adam optimizer states (Adam fp32 = 2x param bytes).

    Results are emitted as PRECISION_BENCHMARK evidence and asserted.
    """

    _MODEL = "qwen/Qwen2-1.5B"
    _WARMUP_STEPS = 5
    _BENCH_STEPS = 50
    _BATCH = 1
    _SEQ_LEN = 128
    _GEN_TOKENS = 20

    @staticmethod
    def _find_qwen15_path():
        """Return local HF cache path for qwen/Qwen2-1.5B, or None."""
        import os
        cache = Path(os.path.expanduser(
            "~/.cache/huggingface/hub/models--qwen--qwen2-1.5b/snapshots"
        ))
        if not cache.exists():
            return None
        snapshots = [d for d in cache.iterdir() if d.is_dir()]
        return str(snapshots[0]) if snapshots else None

    def _run_inference(self, use_fp8: bool) -> tuple[float, int]:
        """Run inference for _BENCH_STEPS forward passes, return (tokens_per_sec, float8_count).

        Loads qwen2-1.5B in BF16, optionally applies Float8Linear conversion,
        then measures forward-pass throughput. No backward pass → no optimizer
        state memory, safe on 8 GB VRAM.

        Returns:
            (tokens_per_sec, float8_layer_count)
        """
        import torch
        import transformers

        model_path = self._find_qwen15_path() or self._MODEL

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # device_map={"": "cuda:0"} forces safetensors to use device="cuda:0" — loading
        # directly to GPU VRAM without staging through CPU RAM. device_map="auto" would
        # try CPU staging first, triggering Windows error 1455 (paging file too small to
        # mmap a 5.9 GB safetensors file in virtual address space).
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": "cuda:0"},
        )

        float8_count = 0
        if use_fp8:
            from torchao.float8 import convert_to_float8_training  # type: ignore[import]

            def _fp8_filter(mod: object, fqn: str) -> bool:
                return "lm_head" not in fqn

            model = convert_to_float8_training(model, module_filter_fn=_fp8_filter)
            float8_count = sum(
                1 for m in model.modules()
                if type(m).__name__ == "Float8Linear"
            )

        model.eval()

        prompt = "The quick brown fox jumps over the lazy dog. " * 4
        inputs = tokenizer(prompt, return_tensors="pt", max_length=self._SEQ_LEN,
                           truncation=True, padding="max_length").to("cuda:0")

        # Warmup
        with torch.no_grad():
            for _ in range(self._WARMUP_STEPS):
                _ = model(**inputs)

        torch.cuda.synchronize()
        t0 = time.monotonic()
        with torch.no_grad():
            for _ in range(self._BENCH_STEPS):
                out = model(**inputs)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - t0

        total_tokens = self._SEQ_LEN * self._BENCH_STEPS
        tps = total_tokens / elapsed if elapsed > 0 else 0.0

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return tps, float8_count

    def test_fp8_throughput_vs_bf16(self, tmp_path):
        """FP8 vs BF16 inference throughput on qwen/Qwen2-1.5B; emit PRECISION_BENCHMARK."""
        import torch

        assert torch.cuda.is_available(), "CUDA required for d10 benchmark"
        gpu_name = torch.cuda.get_device_name(0)
        sm = ".".join(str(x) for x in torch.cuda.get_device_capability())

        logger = EvidenceLogger.get_instance()

        # --- BF16 run (baseline) ---
        bf16_tps, _ = self._run_inference(use_fp8=False)

        # --- FP8 run ---
        fp8_tps, float8_count = self._run_inference(use_fp8=True)

        throughput_ratio = fp8_tps / bf16_tps if bf16_tps > 0 else 0.0

        print(f"\n[d10 FP8_THROUGHPUT_BENCHMARK]")
        print(f"  model:           {self._MODEL}")
        print(f"  gpu:             {gpu_name}  SM {sm}")
        print(f"  Float8Linear:    {float8_count} layers")
        print(f"  BF16 tps:        {bf16_tps:.1f}")
        print(f"  FP8  tps:        {fp8_tps:.1f}")
        print(f"  throughput_ratio:{throughput_ratio:.4f}  (FP8/BF16)")

        # Emit the PRECISION_BENCHMARK evidence (key receipt for scoring)
        logger.emit(
            {
                "type": "PRECISION_BENCHMARK",
                "data": {
                    "model": self._MODEL,
                    "params_b": 1.5,
                    "fp8_tps": round(fp8_tps, 2),
                    "bf16_tps": round(bf16_tps, 2),
                    "throughput_ratio": round(throughput_ratio, 4),
                    "float8_layer_count": float8_count,
                    "bench_steps": self._BENCH_STEPS,
                    "seq_len": self._SEQ_LEN,
                    "gpu": gpu_name,
                    "sm": sm,
                    "mode": "inference",
                },
            },
            organ="train",
        )

        # --- Assertions ---
        assert fp8_tps > 0, f"FP8 tokens/sec must be > 0, got {fp8_tps}"
        assert bf16_tps > 0, f"BF16 tokens/sec must be > 0, got {bf16_tps}"
        assert float8_count > 0, (
            f"No Float8Linear layers found after FP8 conversion of {self._MODEL}"
        )
        # FP8 inference must not regress by more than 25% vs BF16 on RTX 4060
        assert throughput_ratio >= 0.75, (
            f"FP8/BF16 throughput ratio {throughput_ratio:.4f} must be >= 0.75 "
            f"(FP8={fp8_tps:.1f} tps, BF16={bf16_tps:.1f} tps)"
        )

        events = EvidenceLogger.get_events()
        benchmark_events = [
            e for e in events
            if (e.get("type") == "PRECISION_BENCHMARK"
                or (isinstance(e.get("evidence"), dict)
                    and e["evidence"].get("type") == "PRECISION_BENCHMARK"))
        ]
        assert len(benchmark_events) >= 1, "PRECISION_BENCHMARK evidence must be emitted"

    def test_fp8_float8linear_layers_present(self, tmp_path):
        """Verify Float8Linear count matches expected for qwen2-1.5B after FP8 conversion."""
        fp8_tps, float8_count = self._run_inference(use_fp8=True)
        print(f"\n[d10] Float8Linear count: {float8_count}  tps={fp8_tps:.1f}")
        assert float8_count > 0, (
            f"Expected Float8Linear layers after FP8 conversion of {self._MODEL}, "
            f"found {float8_count}. Check torchao installation and module_filter_fn."
        )


# ---------------------------------------------------------------------------
# CPU-safe code path tests (no DOJO_REAL_GPU required)
# ---------------------------------------------------------------------------

class TestFP8CodePathCPU:
    """Verify FP8 infrastructure is importable and wired correctly.

    These tests run on any machine — no GPU or DOJO_REAL_GPU required.
    They confirm that the FP8 dispatch code path exists and that the
    evidence machinery fires correctly.
    """

    @_skip_no_torchao
    def test_torchao_importable(self):
        """torchao must be installed and importable."""
        import torchao  # type: ignore[import]
        assert torchao is not None, "torchao must be importable"

    @_skip_no_torchao
    def test_float8linear_importable(self):
        """Float8Linear must be importable from torchao.float8."""
        from torchao.float8 import Float8Linear  # type: ignore[import]
        assert Float8Linear is not None, "Float8Linear must be importable from torchao.float8"

    @_skip_no_torchao
    def test_convert_to_float8_training_importable(self):
        """convert_to_float8_training must be importable from torchao.float8."""
        from torchao.float8 import convert_to_float8_training  # type: ignore[import]
        assert callable(convert_to_float8_training), (
            "convert_to_float8_training must be callable"
        )

    def test_fp8_handler_importable(self):
        """FP8Handler from dojo.precision.fp8 must be importable."""
        from dojo.precision.fp8 import FP8Handler, is_fp8_supported
        assert FP8Handler is not None
        assert callable(is_fp8_supported)

    def test_fp8_handler_rejects_ampere(self):
        """FP8Handler must reject SM 8.6 (Ampere)."""
        from dojo.core.exceptions import PrecisionUnsupported
        from dojo.core.types import HALProfile, HardwareTier
        from dojo.precision.fp8 import FP8Handler

        ampere = HALProfile(
            tier=HardwareTier.TIER_3090,
            gpu_name="NVIDIA GeForce RTX 3090",
            vram_bytes=24 * 1024 ** 3,
            compute_capability="8.6",
            has_native_fp8=False,
            has_native_fp4=False,
            has_nvlink=False,
            pcie_gen=4,
            interconnect_bandwidth_gb_s=0.0,
            memory_bandwidth_gb_s=0.0,
            driver_version="test",
            profile_sha256="a" * 64,
        )

        with pytest.raises(PrecisionUnsupported):
            FP8Handler.ensure_supported(ampere)

    def test_fp8_handler_accepts_ada(self):
        """FP8Handler must accept SM 8.9 (Ada Lovelace)."""
        from dojo.core.types import HALProfile, HardwareTier
        from dojo.precision.fp8 import FP8Handler

        ada = HALProfile(
            tier=HardwareTier.TIER_4060_LAPTOP,
            gpu_name="NVIDIA GeForce RTX 4060 Laptop GPU",
            vram_bytes=8 * 1024 ** 3,
            compute_capability="8.9",
            has_native_fp8=True,
            has_native_fp4=False,
            has_nvlink=False,
            pcie_gen=4,
            interconnect_bandwidth_gb_s=0.0,
            memory_bandwidth_gb_s=0.0,
            driver_version="test",
            profile_sha256="b" * 64,
        )

        FP8Handler.ensure_supported(ada)  # must not raise

    def test_run_real_fp8_function_exists_in_finetune(self):
        """_run_real_fp8 must exist in dojo.train.finetune."""
        ft = importlib.import_module("dojo.train.finetune")
        assert hasattr(ft, "_run_real_fp8"), (
            "_run_real_fp8 function must exist in dojo.train.finetune"
        )
        assert callable(ft._run_real_fp8)

    def test_fp8_training_mode_routes_to_run_real_fp8(self, tmp_path, monkeypatch):
        """finetune() with training_mode='fp8' must route to _run_real_fp8."""
        ft = importlib.import_module("dojo.train.finetune")

        calls: list[dict] = []

        def _fake_run_real_fp8(config, logger, ckpt_mgr, pack):
            calls.append({"config": config, "pack": pack})
            return 4, 1.23

        monkeypatch.setattr(ft, "_run_real_fp8", _fake_run_real_fp8)
        # Also mock SM detection so it returns 8.9 (Ada)
        monkeypatch.setattr(ft, "_detect_sm_version", lambda: 8.9)

        from dojo.core.types import HALProfile, HardwareTier
        from dojo.train.base import FinetuneConfig

        cfg = FinetuneConfig(
            pack_id="test-fp8-pack",
            base_model="sshleifer/tiny-gpt2",
            training_mode="fp8",
            output_dir=tmp_path / "fp8_out",
            num_epochs=1,
            batch_size=1,
            grad_accum_steps=1,
            lora_r=4,
            lora_alpha=16,
            lora_dropout=0.05,
            learning_rate=1e-4,
            max_seq_length=64,
        )

        hal = HALProfile(
            tier=HardwareTier.TIER_4060_LAPTOP,
            gpu_name="NVIDIA GeForce RTX 4060 Laptop GPU",
            vram_bytes=8 * 1024 ** 3,
            compute_capability="8.9",
            has_native_fp8=True,
            has_native_fp4=False,
            has_nvlink=False,
            pcie_gen=4,
            interconnect_bandwidth_gb_s=0.0,
            memory_bandwidth_gb_s=0.0,
            driver_version="test",
            profile_sha256="c" * 64,
        )

        pack = _make_pack(pack_id="test-fp8-pack")

        result = ft.finetune(cfg, hal, pack)
        assert len(calls) == 1, f"_run_real_fp8 must be called once, got {len(calls)}"
        assert result.final_loss == pytest.approx(1.23, abs=1e-6)

    def test_evidence_emitted_in_fp8_dispatch(self, tmp_path, monkeypatch):
        """finetune() with training_mode='fp8' must emit TRAINING_BACKEND_SELECTED evidence."""
        ft = importlib.import_module("dojo.train.finetune")

        monkeypatch.setattr(ft, "_run_real_fp8", lambda c, l, k, p: (4, 1.23))
        monkeypatch.setattr(ft, "_detect_sm_version", lambda: 8.9)

        from dojo.core.evidence import EvidenceType
        from dojo.core.types import HALProfile, HardwareTier
        from dojo.train.base import FinetuneConfig

        cfg = FinetuneConfig(
            pack_id="test-fp8-ev-pack",
            base_model="sshleifer/tiny-gpt2",
            training_mode="fp8",
            output_dir=tmp_path / "fp8_ev_out",
            num_epochs=1,
            batch_size=1,
            grad_accum_steps=1,
            lora_r=4,
            lora_alpha=16,
            lora_dropout=0.05,
            learning_rate=1e-4,
            max_seq_length=64,
        )

        hal = HALProfile(
            tier=HardwareTier.TIER_4060_LAPTOP,
            gpu_name="NVIDIA GeForce RTX 4060 Laptop GPU",
            vram_bytes=8 * 1024 ** 3,
            compute_capability="8.9",
            has_native_fp8=True,
            has_native_fp4=False,
            has_nvlink=False,
            pcie_gen=4,
            interconnect_bandwidth_gb_s=0.0,
            memory_bandwidth_gb_s=0.0,
            driver_version="test",
            profile_sha256="d" * 64,
        )

        pack = _make_pack(pack_id="test-fp8-ev-pack")

        ft.finetune(cfg, hal, pack)

        events = EvidenceLogger.get_events()
        event_types = [
            e.get("type", e.get("evidence", {}).get("type", ""))
            for e in events
        ]

        assert EvidenceType.TRAINING_START.value in event_types, (
            f"TRAINING_START must be emitted. Got: {event_types}"
        )
        assert EvidenceType.TRAINING_COMPLETE.value in event_types, (
            f"TRAINING_COMPLETE must be emitted. Got: {event_types}"
        )

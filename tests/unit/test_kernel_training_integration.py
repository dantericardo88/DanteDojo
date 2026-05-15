"""Integration tests: Triton kernel wired into QLoRA training path (d15).

Three test classes:
- TestTritonKernelAvailability  (4 tests) — probe the kernel module itself
- TestTritonQLoRAIntegration    (6 tests) — mock full GPU stack, verify evidence
- TestKernelBenchmarkContracts  (4 tests) — correctness / shape guarantees
"""

from __future__ import annotations

import math
import os
import sys
import types
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest
import torch

from dojo.core.evidence import EvidenceLogger, EvidenceType


# ---------------------------------------------------------------------------
# Sentinel
# ---------------------------------------------------------------------------

_SENTINEL = object()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


@pytest.fixture(autouse=True)
def _no_real_gpu(monkeypatch):
    """Default: ensure DOJO_REAL_GPU is not set (CI mode)."""
    monkeypatch.delenv("DOJO_REAL_GPU", raising=False)


# ---------------------------------------------------------------------------
# Shared QKV helpers
# ---------------------------------------------------------------------------

def _qkv(B=1, H=2, S=8, D=16, dtype=torch.float32):
    q = torch.randn(B, H, S, D, dtype=dtype)
    k = torch.randn(B, H, S, D, dtype=dtype)
    v = torch.randn(B, H, S, D, dtype=dtype)
    return q, k, v


# ---------------------------------------------------------------------------
# Mock module factories (mirrors test_sft_qlora_pipeline.py)
# ---------------------------------------------------------------------------

def _make_trainer_result(global_step: int = 42, training_loss: float = 0.25):
    r = MagicMock()
    r.global_step = global_step
    r.training_loss = training_loss
    return r


def _make_mock_mlflow():
    m = types.ModuleType("mlflow")
    mock_run = MagicMock()
    mock_run.info.run_id = "mock-run-id"
    mock_run.info.run_name = "mock-run"
    mock_run.info.status = "RUNNING"
    mock_run.info.start_time = "0"
    mock_run.data.metrics = {}
    mock_run.data.params = {}
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)
    m.set_tracking_uri = MagicMock()
    m.set_experiment = MagicMock()
    m.start_run = MagicMock(return_value=mock_run)
    m.end_run = MagicMock()
    m.log_param = MagicMock()
    m.log_params = MagicMock()
    m.log_metric = MagicMock()
    m.log_artifact = MagicMock()
    m.get_run = MagicMock(return_value=mock_run)
    m.search_runs = MagicMock(return_value=[])
    return m


def _make_mock_trl(trainer_result=None):
    m = types.ModuleType("trl")
    if trainer_result is None:
        trainer_result = _make_trainer_result()
    trainer_mock = MagicMock()
    trainer_mock.train.return_value = trainer_result
    trainer_mock.save_model = MagicMock()
    m.SFTTrainer = MagicMock(return_value=trainer_mock)
    m.SFTConfig = MagicMock(return_value=MagicMock())
    m._trainer_instance = trainer_mock
    return m


def _make_mock_transformers():
    m = types.ModuleType("transformers")
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<eos>"
    m.AutoTokenizer = MagicMock()
    m.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
    mock_model = MagicMock()
    # Give the mock model a config with settable attributes
    mock_model.config = MagicMock()
    m.AutoModelForCausalLM = MagicMock()
    m.AutoModelForCausalLM.from_pretrained.return_value = mock_model
    m.TrainingArguments = MagicMock()
    m.Trainer = MagicMock()
    m.BitsAndBytesConfig = MagicMock(return_value=MagicMock())
    m.DataCollatorForSeq2Seq = MagicMock()
    return m


def _make_mock_peft():
    m = types.ModuleType("peft")
    m.LoraConfig = MagicMock(return_value=MagicMock())
    m.get_peft_model = MagicMock(side_effect=lambda model, cfg: model)
    m.prepare_model_for_kbit_training = MagicMock(side_effect=lambda model: model)
    m.TaskType = types.SimpleNamespace(CAUSAL_LM="CAUSAL_LM")
    return m


def _make_mock_datasets():
    m = types.ModuleType("datasets")
    m.Dataset = MagicMock()
    m.Dataset.from_list = MagicMock(return_value=MagicMock())
    return m


def _make_mock_unsloth():
    return types.ModuleType("unsloth")


# ---------------------------------------------------------------------------
# Pack / HAL builders
# ---------------------------------------------------------------------------

def _make_hal():
    from dojo.core.types import HALProfile, HardwareTier
    return HALProfile(
        tier=HardwareTier.TIER_3090,
        gpu_name="Mock GPU",
        vram_bytes=24 * 1024 ** 3,
        compute_capability="8.6",
        has_native_fp8=False,
        has_native_fp4=False,
        has_nvlink=False,
        pcie_gen=4,
        interconnect_bandwidth_gb_s=0.0,
        memory_bandwidth_gb_s=0.0,
        driver_version="550.0",
        profile_sha256="a" * 64,
        signed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _make_pack():
    from dojo.core.types import LicenseSpec, PackFormat, ProvenanceManifest
    from dojo.pack.schema import SpecializationPack
    return SpecializationPack(
        pack_id="test-pack",
        version="1.0.0",
        schema_version="1.0",
        sample_count=10,
        format=PackFormat.ALPACA,
        splits={"train": 8, "val": 2},
        harvest_receipt="abc123",
        license=LicenseSpec(
            spdx_id="MIT",
            allows_derived_ft=True,
            allows_commercial_use=True,
            requires_attribution=False,
        ),
        provenance=ProvenanceManifest(
            harvest_organ_version="1.0",
            sources=("https://example.com",),
            collection_date_utc="2026-01-01",
            dedup_method="minhash",
            human_reviewed=False,
        ),
        signature=b"\x00" * 64,
        manifest_sha256="d" * 64,
        data_parquet_sha256="e" * 64,
    )


# ---------------------------------------------------------------------------
# Mock GPU context manager
# ---------------------------------------------------------------------------

@contextmanager
def _mock_gpu_env(
    tmp_path: Path,
    triton_in_sys_modules: bool = False,
) -> Generator[dict[str, Any], None, None]:
    """Inject mock modules + DOJO_REAL_GPU=1 to exercise _run_real_qlora.

    Args:
        triton_in_sys_modules: when True, injects a fake ``triton`` module so
            ``is_triton_available()`` returns True; when False, ensures triton
            is absent from sys.modules.
    """
    trainer_result = _make_trainer_result(global_step=30, training_loss=0.20)
    mock_trl = _make_mock_trl(trainer_result)
    mock_transformers = _make_mock_transformers()
    mock_peft = _make_mock_peft()
    mock_datasets = _make_mock_datasets()
    mock_mlflow = _make_mock_mlflow()
    mock_unsloth = _make_mock_unsloth()

    output_dir = tmp_path / "qlora_kernel_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    from dojo.train.base import FinetuneConfig
    cfg = FinetuneConfig(
        pack_id="mock-pack",
        base_model="mock/model-7b",
        output_dir=output_dir,
        num_epochs=1,
        training_mode="qlora",
    )
    pack = _make_pack()
    hal = _make_hal()

    patch_targets = {
        "trl": mock_trl,
        "transformers": mock_transformers,
        "peft": mock_peft,
        "datasets": mock_datasets,
        "mlflow": mock_mlflow,
        "unsloth": mock_unsloth,
    }

    # Optionally inject / remove a fake triton module
    if triton_in_sys_modules:
        patch_targets["triton"] = types.ModuleType("triton")

    old_modules: dict[str, Any] = {}
    for name in list(patch_targets) + ([] if triton_in_sys_modules else ["triton"]):
        old_modules[name] = sys.modules.get(name, _SENTINEL)

    for name, mod in patch_targets.items():
        sys.modules[name] = mod

    if not triton_in_sys_modules:
        # Ensure triton is absent
        sys.modules.pop("triton", None)

    old_real_gpu = os.environ.get("DOJO_REAL_GPU")
    old_mlflow_uri = os.environ.get("DOJO_MLFLOW_URI")
    os.environ["DOJO_REAL_GPU"] = "1"
    os.environ["DOJO_MLFLOW_URI"] = "http://127.0.0.1:19999"

    try:
        yield {
            "trl": mock_trl,
            "transformers": mock_transformers,
            "peft": mock_peft,
            "datasets": mock_datasets,
            "mlflow": mock_mlflow,
            "config": cfg,
            "pack": pack,
            "hal": hal,
            "trainer_result": trainer_result,
        }
    finally:
        for name, orig in old_modules.items():
            if orig is _SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig

        if old_real_gpu is None:
            os.environ.pop("DOJO_REAL_GPU", None)
        else:
            os.environ["DOJO_REAL_GPU"] = old_real_gpu

        if old_mlflow_uri is None:
            os.environ.pop("DOJO_MLFLOW_URI", None)
        else:
            os.environ["DOJO_MLFLOW_URI"] = old_mlflow_uri


# ===========================================================================
# 1. TestTritonKernelAvailability (4 tests)
# ===========================================================================

class TestTritonKernelAvailability:
    """Basic probes of the kernel module without involving the training stack."""

    def test_flash_attention_forward_callable(self):
        """flash_attention_forward must be callable."""
        from dojo.kernels.attention import flash_attention_forward
        assert callable(flash_attention_forward)

    def test_flash_attention_returns_tensor_shape(self):
        """flash_attention_forward returns a tensor with the same shape as q."""
        from dojo.kernels.attention import flash_attention_forward
        q, k, v = _qkv()
        out = flash_attention_forward(q, k, v)
        assert isinstance(out, torch.Tensor)
        assert out.shape == q.shape

    def test_is_triton_available_returns_bool(self):
        """is_triton_available() must return a plain bool (not a truthy object)."""
        from dojo.kernels.attention import is_triton_available
        result = is_triton_available()
        assert type(result) is bool  # noqa: E721 — strict type check

    def test_kernel_graceful_fallback_no_triton(self):
        """When triton is absent from sys.modules, forward still runs without error."""
        # Temporarily hide triton if it happens to be installed
        old = sys.modules.pop("triton", _SENTINEL)
        try:
            from dojo.kernels.attention import flash_attention_forward
            q, k, v = _qkv()
            out = flash_attention_forward(q, k, v)
            assert out.shape == q.shape
        finally:
            if old is not _SENTINEL:
                sys.modules["triton"] = old


# ===========================================================================
# 2. TestTritonQLoRAIntegration (6 tests)
# ===========================================================================

class TestTritonQLoRAIntegration:
    """End-to-end QLoRA training with mocked GPU stack; verifies kernel evidence."""

    def _get_kernel_events(self) -> list[dict]:
        return [
            e for e in EvidenceLogger._events
            if e.get("evidence", {}).get("type") == "train.kernel"
        ]

    def test_qlora_training_emits_kernel_evidence(self, tmp_path):
        """After QLoRA training, at least one 'train.kernel' evidence event exists."""
        with _mock_gpu_env(tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])

        assert len(self._get_kernel_events()) >= 1

    def test_qlora_uses_triton_when_available(self, tmp_path):
        """When triton is in sys.modules, kernel_used='flash_attn2' in evidence."""
        with _mock_gpu_env(tmp_path=tmp_path, triton_in_sys_modules=True) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])

        kernel_events = self._get_kernel_events()
        assert kernel_events, "Expected at least one train.kernel evidence event"
        kernel_used = kernel_events[-1]["evidence"]["data"]["kernel_used"]
        assert kernel_used == "flash_attn2"

    def test_qlora_falls_back_when_triton_missing(self, tmp_path):
        """When triton is absent, kernel_used='standard' in evidence."""
        with _mock_gpu_env(tmp_path=tmp_path, triton_in_sys_modules=False) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])

        kernel_events = self._get_kernel_events()
        assert kernel_events, "Expected at least one train.kernel evidence event"
        kernel_used = kernel_events[-1]["evidence"]["data"]["kernel_used"]
        assert kernel_used == "standard"

    def test_qlora_kernel_evidence_has_kernel_used_field(self, tmp_path):
        """The train.kernel evidence data dict has a 'kernel_used' key."""
        with _mock_gpu_env(tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])

        kernel_events = self._get_kernel_events()
        assert kernel_events
        assert "kernel_used" in kernel_events[-1]["evidence"]["data"]

    def test_qlora_training_still_completes_with_kernel(self, tmp_path):
        """Training completes without error when the kernel path is wired."""
        with _mock_gpu_env(tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            from dojo.train.base import FinetuneResult
            result = finetune(ctx["config"], ctx["hal"], ctx["pack"])

        assert isinstance(result, FinetuneResult)
        assert result.steps_completed > 0

    def test_qlora_training_still_emits_train_complete(self, tmp_path):
        """'train.finetune.completed' evidence is still emitted alongside kernel evidence."""
        with _mock_gpu_env(tmp_path=tmp_path) as ctx:
            from dojo.train.finetune import finetune
            finetune(ctx["config"], ctx["hal"], ctx["pack"])

        event_types = {e.get("type", "") for e in EvidenceLogger.get_events()}
        assert EvidenceType.TRAINING_COMPLETE.value in event_types


# ===========================================================================
# 3. TestKernelBenchmarkContracts (4 tests)
# ===========================================================================

class TestKernelBenchmarkContracts:
    """Correctness and shape contracts for flash_attention_forward in CI mode."""

    def test_flash_attention_causal_mask_applied(self):
        """causal=True must produce a different result than causal=False."""
        from dojo.kernels.attention import flash_attention_forward
        torch.manual_seed(42)
        q, k, v = _qkv(S=16)
        out_full = flash_attention_forward(q, k, v, causal=False)
        out_causal = flash_attention_forward(q, k, v, causal=True)
        # Causal masking changes attention weights — outputs must differ.
        assert not torch.allclose(out_full, out_causal, atol=1e-4), (
            "causal=True and causal=False produced identical outputs; mask not applied"
        )

    def test_flash_attention_online_softmax(self):
        """Result must be equivalent to scaled dot-product attention within tolerance."""
        from dojo.kernels.attention import flash_attention_forward
        torch.manual_seed(7)
        q, k, v = _qkv(S=8, D=16)
        D = q.shape[-1]
        sm_scale = 1.0 / math.sqrt(D)

        # Reference: manual O(S²) scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
        attn_ref = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        ref = torch.matmul(attn_ref, v)

        out = flash_attention_forward(q, k, v, sm_scale=sm_scale)
        assert torch.allclose(out.float(), ref.float(), atol=1e-4), (
            f"Max diff from reference: {(out.float() - ref.float()).abs().max().item():.6f}"
        )

    def test_flash_attention_handles_batch_size_1(self):
        """No crash and correct output shape with batch_size=1."""
        from dojo.kernels.attention import flash_attention_forward
        q, k, v = _qkv(B=1, H=4, S=32, D=32)
        out = flash_attention_forward(q, k, v)
        assert out.shape == q.shape

    def test_flash_attention_handles_variable_seq_len(self):
        """Both seq_len=128 and seq_len=512 produce outputs with the right shape."""
        from dojo.kernels.attention import flash_attention_forward
        for seq_len in (128, 512):
            q, k, v = _qkv(B=1, H=2, S=seq_len, D=16)
            out = flash_attention_forward(q, k, v)
            assert out.shape == q.shape, (
                f"Shape mismatch at seq_len={seq_len}: got {out.shape}, expected {q.shape}"
            )

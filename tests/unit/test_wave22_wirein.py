"""Wave 22/23 wire-in integration tests.

Each test asserts production-side imports/instantiations of the previously
orphaned modules and exercises the wired callsite.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import patch

import torch  # noqa: F401 — pre-load torch so patch.dict(sys.modules) is safe


DOJO_ROOT = Path(__file__).resolve().parents[2] / "dojo"


def _has_import(module_path: str) -> bool:
    for p in DOJO_ROOT.rglob("*.py"):
        if "tests" in p.parts:
            continue
        if module_path in p.read_text(encoding="utf-8", errors="ignore"):
            return True
    return False


# ---------------------------------------------------------------------------
# d31 vLLM inline-engine wire-in
# ---------------------------------------------------------------------------

class TestInitEngineWiredIn:
    def test_inline_engine_helper_exists(self):
        from dojo.serve.vllm_backend import _inline_engine_requested
        assert callable(_inline_engine_requested)

    def test_inline_engine_env_toggle(self, monkeypatch):
        from dojo.serve.vllm_backend import _inline_engine_requested
        monkeypatch.setenv("DOJO_VLLM_INLINE", "1")
        assert _inline_engine_requested() is True
        monkeypatch.setenv("DOJO_VLLM_INLINE", "0")
        assert _inline_engine_requested() is False

    def test_start_calls_init_engine_when_inline_requested(self, monkeypatch):
        from datetime import datetime, timezone
        from dojo.serve.vllm_backend import VLLMBackend
        from dojo.core.types import HALProfile, HardwareTier

        profile = HALProfile(
            tier=HardwareTier.TIER_3090,
            gpu_name="t", vram_bytes=1 << 30,
            compute_capability="8.6",
            has_native_fp8=False, has_native_fp4=False,
            driver_version="5", profile_sha256="a"*64,
            signed_at=datetime(2026,1,1,tzinfo=timezone.utc),
        )
        backend = VLLMBackend(
            model_path=Path("/fake"),
            profile=profile,
            port=8000,
            context_length=2048,
        )
        monkeypatch.setenv("DOJO_VLLM_INLINE", "1")
        with patch.object(VLLMBackend, "init_engine") as mock_init:
            with patch("dojo.serve.vllm_backend._require_vllm"):
                pid = backend.start()
        mock_init.assert_called_once()
        assert pid != 0


# ---------------------------------------------------------------------------
# d35 cost-receipts wire-in
# ---------------------------------------------------------------------------

class TestCostReceiptsWiredIn:
    def test_router_imports_cost_receipts(self):
        from dojo.route import router
        assert "from dojo.route.cost_receipts" in inspect.getsource(router)

    def test_dispatch_returns_signed_cost_receipt(self):
        from dojo.core.types import PrivacyTier, RoutingPolicy, RoutingOption
        from dojo.route.router import dispatch
        policy = RoutingPolicy(
            task_name="hello",
            preference_order=(RoutingOption.LOCAL_GENERALIST,),
            max_latency_ms=10_000,
            max_cost_usd=1.0,
            privacy_tier=PrivacyTier.INTERNAL,
            fallback_on_unavailable=False,
        )
        result = dispatch(task="hello", input_text="hi", policy=policy)
        assert "signed_cost_receipt" in result
        signed = result["signed_cost_receipt"]
        assert signed["kind"] == "cost_receipt"
        assert "signature" in signed
        assert signed["signature_version"] == "v1"


# ---------------------------------------------------------------------------
# d38 contamination_strict wire-in
# ---------------------------------------------------------------------------

class TestContaminationStrictWiredIn:
    def test_harness_imports_contamination_strict(self):
        from dojo.eval import harness
        src = inspect.getsource(harness)
        assert "contamination_strict" in src

    def test_contamination_event_emitted_during_run_eval(self):
        from dojo.core.evidence import EvidenceLogger
        from dojo.core.types import ModelArtifact
        from dojo.eval.harness import EvalSuite, run_eval
        EvidenceLogger.clear()
        m = ModelArtifact(
            artifact_id="x", base_model="b",
            lora_weights=None, quantization=None,
            training_receipt="r"*64, passport=None,
        )
        run_eval(m, suite=EvalSuite(suite_type="dante_v1", tasks=("toxicity_basic",)),
                 seeds=1, allow_synthetic_fallback=True)
        events = EvidenceLogger.get_events()
        assert any(e.get("type") == "eval.contamination_strict" for e in events)


# ---------------------------------------------------------------------------
# d04 EvidenceCoverage wire-in
# ---------------------------------------------------------------------------

class TestEvidenceCoverageWiredIn:
    def test_harness_imports_evidence_coverage(self):
        from dojo.eval import harness
        src = inspect.getsource(harness)
        assert "evidence_coverage" in src or "EvidenceCoverage" in src

    def test_coverage_audit_event_emitted(self):
        from dojo.core.evidence import EvidenceLogger
        from dojo.core.types import ModelArtifact
        from dojo.eval.harness import EvalSuite, run_eval
        EvidenceLogger.clear()
        m = ModelArtifact(
            artifact_id="x", base_model="b",
            lora_weights=None, quantization=None,
            training_receipt="r"*64, passport=None,
        )
        run_eval(m, suite=EvalSuite(suite_type="dante_v1", tasks=("toxicity_basic",)),
                 seeds=1, allow_synthetic_fallback=True)
        events = EvidenceLogger.get_events()
        audit = [e for e in events if e.get("type") == "eval.coverage_audit"]
        assert audit
        assert audit[0]["data"]["total_emitted"] > 0


# ---------------------------------------------------------------------------
# d20 prompt_loader wire-in
# ---------------------------------------------------------------------------

class TestPromptLoaderWiredIn:
    def test_daemon_exposes_fixture_step_runner_factory(self):
        from dojo.idle.daemon import make_fixture_step_runner
        assert callable(make_fixture_step_runner)

    def test_daemon_imports_prompt_loader(self):
        from dojo.idle import daemon
        src = inspect.getsource(daemon)
        assert "FixtureBasedPromptLoader" in src or "prompt_loader" in src

    def test_fixture_runner_produces_real_prompt_metadata(self):
        from dojo.idle.daemon import make_fixture_step_runner
        runner = make_fixture_step_runner()
        result = runner(0, 0)
        assert result["mode"] == "fixture"
        assert "prompt_chars" in result


# ---------------------------------------------------------------------------
# d27 kv_streamed_inference wire-in
# ---------------------------------------------------------------------------

class TestKvStreamedInferenceWiredIn:
    def test_inference_loop_exposes_kv_runner(self):
        from dojo.stream.inference_loop import run_kv_streamed_inference
        assert callable(run_kv_streamed_inference)

    def test_inference_loop_imports_kv_streamed(self):
        from dojo.stream import inference_loop
        src = inspect.getsource(inference_loop)
        assert "kv_streamed_inference" in src

    def test_run_kv_streamed_returns_tokens(self):
        from dojo.stream.inference_loop import run_kv_streamed_inference
        tokens = run_kv_streamed_inference(
            layer_ids=["L0", "L1"],
            query=torch.zeros(32),
            head_dim=32,
            vocab_size=100,
        )
        assert len(tokens) == 2
        assert all(0 <= t < 100 for t in tokens)


# ---------------------------------------------------------------------------
# d08 revocation_strict wire-in
# ---------------------------------------------------------------------------

class TestRevocationStrictWiredIn:
    def test_promote_gate_imports_revocation_strict(self):
        from dojo.promote import gate
        src = inspect.getsource(gate)
        assert "revocation_strict" in src or "StrictRevocationCheck" in src


# ---------------------------------------------------------------------------
# Production-import audit
# ---------------------------------------------------------------------------

class TestWave22NotOrphaned:
    def test_cost_receipts_imported(self):
        assert _has_import("dojo.route.cost_receipts")

    def test_contamination_strict_imported(self):
        assert _has_import("dojo.eval.contamination_strict")

    def test_evidence_coverage_imported(self):
        assert _has_import("dojo.core.evidence_coverage")

    def test_prompt_loader_imported(self):
        assert _has_import("dojo.idle.prompt_loader")

    def test_kv_streamed_inference_imported(self):
        assert _has_import("dojo.stream.kv_streamed_inference")

    def test_revocation_strict_imported(self):
        assert _has_import("dojo.core.revocation_strict")

"""Unit tests for dojo/serve/ — Phase 3B inference serving layer.

Covers:
- build_manifest returns DeploymentManifest with non-empty manifest_sha256
- launch_server runs FSM to READY state, returns DeploymentManifest
- launch_server emits SERVE_START and SERVE_READY evidence
- Backend selection: TIER_4060_LAPTOP → LLAMACPP, TIER_3090 → VLLM
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.types import Backend, HALProfile, HardwareTier
from dojo.serve.manifest import build_manifest
from dojo.serve.launcher import LaunchFSM, LaunchState, launch_server


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_hal(
    tier: HardwareTier = HardwareTier.TIER_3090,
    vram_bytes: int = 24 * 1024 ** 3,
    compute_capability: str = "8.6",
) -> HALProfile:
    return HALProfile(
        tier=tier,
        gpu_name="Test GPU",
        vram_bytes=vram_bytes,
        compute_capability=compute_capability,
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


@pytest.fixture(autouse=True)
def clear_evidence():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ---------------------------------------------------------------------------
# build_manifest tests
# ---------------------------------------------------------------------------

class TestBuildManifest:
    def test_returns_deployment_manifest(self):
        profile = _make_hal()
        manifest = build_manifest(
            model_artifact_id="dojo://test-model/v1",
            backend=Backend.VLLM,
            endpoint_url="http://localhost:8080",
            context_length=4096,
        )
        assert manifest.model_artifact_id == "dojo://test-model/v1"
        assert manifest.backend == Backend.VLLM
        assert manifest.endpoint_url == "http://localhost:8080"

    def test_manifest_sha256_is_non_empty(self):
        manifest = build_manifest(
            model_artifact_id="dojo://test-model/v1",
            backend=Backend.VLLM,
            endpoint_url="http://localhost:8080",
            context_length=4096,
        )
        assert manifest.manifest_sha256
        assert len(manifest.manifest_sha256) == 64  # SHA-256 hex = 64 chars

    def test_manifest_sha256_changes_with_different_inputs(self):
        m1 = build_manifest(
            model_artifact_id="dojo://model-a/v1",
            backend=Backend.VLLM,
            endpoint_url="http://localhost:8080",
            context_length=4096,
        )
        m2 = build_manifest(
            model_artifact_id="dojo://model-b/v1",
            backend=Backend.LLAMACPP,
            endpoint_url="http://localhost:9090",
            context_length=2048,
        )
        assert m1.manifest_sha256 != m2.manifest_sha256

    def test_manifest_defaults(self):
        manifest = build_manifest(
            model_artifact_id="dojo://test/v1",
            backend=Backend.LLAMACPP,
            endpoint_url="http://localhost:8080",
            context_length=2048,
        )
        assert manifest.tp_size == 1
        assert manifest.dtype == "bfloat16"
        assert manifest.passport_id == ""
        assert manifest.pid == 0
        assert manifest.quantization is None


# ---------------------------------------------------------------------------
# launch_server tests
# ---------------------------------------------------------------------------

class TestLaunchServer:
    def _patch_backends(self):
        """Patch backends so they don't require installed packages."""
        vllm_patcher = patch.dict(sys.modules, {"vllm": object()})
        llama_patcher = patch.dict(sys.modules, {"llama_cpp": object()})
        return vllm_patcher, llama_patcher

    def test_launch_server_returns_manifest(self, tmp_path):
        profile = _make_hal(tier=HardwareTier.TIER_3090)
        with patch("dojo.serve.launcher._provision_backend", return_value=12345), \
             patch("dojo.serve.launcher._warm_backend"):
            manifest = launch_server(
                model_artifact_id="dojo://test-model/v1",
                model_path=tmp_path / "model",
                profile=profile,
            )
        assert manifest is not None
        assert manifest.manifest_sha256
        assert len(manifest.manifest_sha256) == 64

    def test_launch_server_fsm_reaches_ready(self, tmp_path):
        """FSM must transition through IDLE→CONFIGURING→PROVISIONING→WARMING→READY."""
        profile = _make_hal(tier=HardwareTier.TIER_3090)

        with patch("dojo.serve.launcher._provision_backend", return_value=0), \
             patch("dojo.serve.launcher._warm_backend"):
            manifest = launch_server(
                model_artifact_id="dojo://test-model/v1",
                model_path=tmp_path / "model",
                profile=profile,
            )
        # If we got a manifest without exception, FSM reached READY
        assert manifest is not None

    def test_launch_server_emits_serve_start_evidence(self, tmp_path):
        profile = _make_hal(tier=HardwareTier.TIER_3090)
        with patch("dojo.serve.launcher._provision_backend", return_value=0), \
             patch("dojo.serve.launcher._warm_backend"):
            launch_server(
                model_artifact_id="dojo://test-model/v1",
                model_path=tmp_path / "model",
                profile=profile,
            )

        events = EvidenceLogger._events
        evidence_types = [
            e["evidence"].get("type")
            for e in events
            if isinstance(e.get("evidence"), dict)
        ]
        assert "serve.start" in evidence_types

    def test_launch_server_emits_serve_ready_evidence(self, tmp_path):
        profile = _make_hal(tier=HardwareTier.TIER_3090)
        with patch("dojo.serve.launcher._provision_backend", return_value=0), \
             patch("dojo.serve.launcher._warm_backend"):
            launch_server(
                model_artifact_id="dojo://test-model/v1",
                model_path=tmp_path / "model",
                profile=profile,
            )

        events = EvidenceLogger._events
        evidence_types = [
            e["evidence"].get("type")
            for e in events
            if isinstance(e.get("evidence"), dict)
        ]
        assert "serve.ready" in evidence_types

    def test_manifest_artifact_id_matches(self, tmp_path):
        profile = _make_hal(tier=HardwareTier.TIER_3090)
        with patch("dojo.serve.launcher._provision_backend", return_value=42), \
             patch("dojo.serve.launcher._warm_backend"):
            manifest = launch_server(
                model_artifact_id="dojo://specific-model/v7",
                model_path=tmp_path / "model",
                profile=profile,
            )
        assert manifest.model_artifact_id == "dojo://specific-model/v7"


# ---------------------------------------------------------------------------
# Backend selection tests
# ---------------------------------------------------------------------------

class TestBackendSelection:
    def test_tier_4060_selects_llamacpp(self, tmp_path):
        profile = _make_hal(tier=HardwareTier.TIER_4060_LAPTOP, vram_bytes=8 * 1024 ** 3)
        with patch("dojo.serve.launcher._provision_backend", return_value=0) as mock_prov, \
             patch("dojo.serve.launcher._warm_backend"):
            manifest = launch_server(
                model_artifact_id="dojo://test/v1",
                model_path=tmp_path / "model",
                profile=profile,
            )
        assert manifest.backend == Backend.LLAMACPP

    def test_tier_3090_selects_vllm(self, tmp_path):
        profile = _make_hal(tier=HardwareTier.TIER_3090)
        with patch("dojo.serve.launcher._provision_backend", return_value=0), \
             patch("dojo.serve.launcher._warm_backend"):
            manifest = launch_server(
                model_artifact_id="dojo://test/v1",
                model_path=tmp_path / "model",
                profile=profile,
            )
        assert manifest.backend == Backend.VLLM

    def test_tier_3090_dual_selects_vllm_tp2(self, tmp_path):
        profile = _make_hal(tier=HardwareTier.TIER_3090_DUAL, vram_bytes=48 * 1024 ** 3)
        with patch("dojo.serve.launcher._provision_backend", return_value=0), \
             patch("dojo.serve.launcher._warm_backend"):
            manifest = launch_server(
                model_artifact_id="dojo://test/v1",
                model_path=tmp_path / "model",
                profile=profile,
            )
        assert manifest.backend == Backend.VLLM
        assert manifest.tp_size == 2

    def test_tier_h100_selects_vllm(self, tmp_path):
        profile = _make_hal(
            tier=HardwareTier.TIER_H100,
            vram_bytes=80 * 1024 ** 3,
            compute_capability="9.0",
        )
        with patch("dojo.serve.launcher._provision_backend", return_value=0), \
             patch("dojo.serve.launcher._warm_backend"):
            manifest = launch_server(
                model_artifact_id="dojo://test/v1",
                model_path=tmp_path / "model",
                profile=profile,
            )
        assert manifest.backend == Backend.VLLM

    def test_tier_apple_m4_selects_mlx(self, tmp_path):
        profile = _make_hal(
            tier=HardwareTier.TIER_APPLE_M4,
            vram_bytes=32 * 1024 ** 3,
            compute_capability="0.0",
        )
        with patch("dojo.serve.launcher._provision_backend", return_value=0), \
             patch("dojo.serve.launcher._warm_backend"):
            manifest = launch_server(
                model_artifact_id="dojo://test/v1",
                model_path=tmp_path / "model",
                profile=profile,
            )
        assert manifest.backend == Backend.MLX_LM

    def test_backend_override(self, tmp_path):
        """Explicit backend argument overrides auto-selection."""
        profile = _make_hal(tier=HardwareTier.TIER_3090)
        with patch("dojo.serve.launcher._provision_backend", return_value=0), \
             patch("dojo.serve.launcher._warm_backend"):
            manifest = launch_server(
                model_artifact_id="dojo://test/v1",
                model_path=tmp_path / "model",
                profile=profile,
                backend=Backend.LLAMACPP,
            )
        assert manifest.backend == Backend.LLAMACPP


# ---------------------------------------------------------------------------
# LaunchFSM tests
# ---------------------------------------------------------------------------

class TestLaunchFSM:
    def test_initial_state_is_idle(self):
        fsm = LaunchFSM()
        assert fsm.state == LaunchState.IDLE

    def test_valid_transition_sequence(self):
        fsm = LaunchFSM()
        fsm.transition(LaunchState.CONFIGURING)
        fsm.transition(LaunchState.PROVISIONING)
        fsm.transition(LaunchState.WARMING)
        fsm.transition(LaunchState.READY)
        assert fsm.state == LaunchState.READY

    def test_illegal_transition_raises(self):
        from dojo.core.exceptions import FSMTransitionError
        fsm = LaunchFSM()
        with pytest.raises(FSMTransitionError):
            fsm.transition(LaunchState.READY)  # Can't jump from IDLE to READY

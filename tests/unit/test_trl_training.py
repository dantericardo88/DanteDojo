"""Unit tests for TRL / Windows-native training path in dojo/train/finetune.py.

Tests that:
  - On Windows-native, _select_backend() returns TrainingBackend.TRL
  - On Windows+WSL, _select_backend() returns TrainingBackend.WSL_UNSLOTH
  - On macOS, _select_backend() returns TrainingBackend.MLX
  - On Linux/WSL2, _select_backend() uses the tier-based map
  - finetune() dispatches to _run_real_sft for TRL + sft mode
  - finetune() dispatches to _run_real_qlora for TRL + qlora mode
  - finetune() delegates to run_training_in_wsl for WSL_UNSLOTH backend
  - ensure_training_deps is auto-called before the training loop
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.core.types import HALProfile, HardwareTier, TrainingBackend
from dojo.hal.env_detect import RuntimeEnv, _clear_cache
from dojo.train.finetune import _select_backend


@pytest.fixture(autouse=True)
def clear_env_cache():
    _clear_cache()
    EvidenceLogger.clear()
    yield
    _clear_cache()
    EvidenceLogger.clear()


# ---------------------------------------------------------------------------
# _select_backend — platform routing
# ---------------------------------------------------------------------------

class TestSelectBackend:
    def _profile(self, tier: HardwareTier = HardwareTier.TIER_4060_LAPTOP) -> HALProfile:
        return HALProfile(
            tier=tier,
            gpu_name="RTX 4060 Laptop",
            vram_bytes=8 * 1024**3,
            compute_capability="8.9",
            has_native_fp8=True,
            has_native_fp4=False,
        )

    @patch("dojo.train.finetune.detect_runtime")
    def test_windows_native_returns_trl(self, mock_detect):
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.WINDOWS_NATIVE
        mock_detect.return_value = mock_env
        assert _select_backend(self._profile()) == TrainingBackend.TRL

    @patch("dojo.train.finetune.detect_runtime")
    def test_windows_wsl_returns_wsl_unsloth(self, mock_detect):
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.WINDOWS_WSL
        mock_detect.return_value = mock_env
        assert _select_backend(self._profile()) == TrainingBackend.WSL_UNSLOTH

    @patch("dojo.train.finetune.detect_runtime")
    def test_macos_apple_m4_returns_mlx(self, mock_detect):
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.MACOS
        mock_detect.return_value = mock_env
        assert _select_backend(self._profile(HardwareTier.TIER_APPLE_M4)) == TrainingBackend.MLX

    @patch("dojo.train.finetune.detect_runtime")
    def test_macos_non_apple_tier_returns_trl(self, mock_detect):
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.MACOS
        mock_detect.return_value = mock_env
        assert _select_backend(self._profile(HardwareTier.TIER_4060_LAPTOP)) == TrainingBackend.TRL

    @patch("dojo.train.finetune.detect_runtime")
    def test_linux_4060_returns_unsloth(self, mock_detect):
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.LINUX
        mock_detect.return_value = mock_env
        assert _select_backend(self._profile(HardwareTier.TIER_4060_LAPTOP)) == TrainingBackend.UNSLOTH

    @patch("dojo.train.finetune.detect_runtime")
    def test_linux_3090_dual_returns_axolotl(self, mock_detect):
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.LINUX
        mock_detect.return_value = mock_env
        assert _select_backend(self._profile(HardwareTier.TIER_3090_DUAL)) == TrainingBackend.AXOLOTL

    @patch("dojo.train.finetune.detect_runtime")
    def test_linux_cpu_tier_returns_native(self, mock_detect):
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.LINUX
        mock_detect.return_value = mock_env
        assert _select_backend(self._profile(HardwareTier.TIER_CPU)) == TrainingBackend.NATIVE

    @patch("dojo.train.finetune.detect_runtime")
    def test_wsl2_4060_returns_unsloth(self, mock_detect):
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.WSL2
        mock_detect.return_value = mock_env
        assert _select_backend(self._profile(HardwareTier.TIER_4060_LAPTOP)) == TrainingBackend.UNSLOTH


# ---------------------------------------------------------------------------
# finetune() dispatch — TRL backend
# ---------------------------------------------------------------------------

@pytest.fixture
def trl_finetune_fixtures(tmp_path):
    """Return (config, hal_profile, pack) wired for TRL on Windows-native."""
    from dojo.pack.schema import SpecializationPack
    from dojo.train.base import FinetuneConfig

    config = FinetuneConfig(
        base_model="sshleifer/tiny-gpt2",
        pack_id="test-pack",
        output_dir=tmp_path / "out",
        num_epochs=1,
        batch_size=1,
        lora_r=4,
        lora_alpha=8,
        max_seq_length=64,
        training_mode="sft",
        seed=42,
    )
    hal_profile = HALProfile(
        tier=HardwareTier.TIER_4060_LAPTOP,
        gpu_name="RTX 4060 Laptop",
        vram_bytes=8 * 1024**3,
        compute_capability="8.9",
        has_native_fp8=True,
        has_native_fp4=False,
    )
    pack = MagicMock(spec=SpecializationPack)
    pack.pack_id = "test-pack"
    pack.license = MagicMock()
    pack.license.allows_derived_ft = True
    pack.license.allows_commercial_use = True
    pack.samples = [{"text": "hello world"}]
    return config, hal_profile, pack


def _make_tracker_mock(mock_tracker_cls):
    """Return a properly-configured ExperimentTracker mock for finetune() tests."""
    mock_tracker = MagicMock()
    mock_tracker.__enter__ = MagicMock(return_value=mock_tracker)
    mock_tracker.__exit__ = MagicMock(return_value=False)
    mock_tracker._active_run_id = "test-run-123"  # must be str, not MagicMock
    mock_tracker_cls.return_value = mock_tracker
    mock_tracker.run = MagicMock(return_value=mock_tracker)
    return mock_tracker


class TestFinetuneTRLDispatch:
    """Verify finetune() dispatches to _run_real_sft/_run_real_qlora on TRL backend."""

    @patch("dojo.train.finetune.detect_runtime")
    @patch("dojo.train.finetune._is_backend_available", return_value=True)
    @patch("dojo.train.finetune._run_real_sft", return_value=(10, 0.5))
    @patch("dojo.train.finetune.ensure_training_deps")
    @patch("dojo.train.finetune.ExperimentTracker")
    def test_trl_sft_dispatches_to_run_real_sft(
        self, mock_tracker_cls, mock_ensure_deps, mock_run_sft, mock_backend_avail, mock_detect,
        trl_finetune_fixtures,
    ):
        config, hal_profile, pack = trl_finetune_fixtures
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.WINDOWS_NATIVE
        mock_detect.return_value = mock_env
        _make_tracker_mock(mock_tracker_cls)

        from dojo.train.finetune import finetune
        with patch("dojo.train.finetune.check_license_allows_training", return_value=None):
            result = finetune(config, hal_profile, pack)

        mock_run_sft.assert_called_once()
        assert result.steps_completed == 10
        assert result.final_loss == 0.5
        assert result.backend_used == "trl"

    @patch("dojo.train.finetune.detect_runtime")
    @patch("dojo.train.finetune._is_backend_available", return_value=True)
    @patch("dojo.train.finetune._run_real_qlora", return_value=(8, 0.6))
    @patch("dojo.train.finetune.ensure_training_deps")
    @patch("dojo.train.finetune.ExperimentTracker")
    def test_trl_qlora_dispatches_to_run_real_qlora(
        self, mock_tracker_cls, mock_ensure_deps, mock_run_qlora, mock_backend_avail, mock_detect,
        trl_finetune_fixtures,
    ):
        config, hal_profile, pack = trl_finetune_fixtures
        config = config.__class__(
            **{**{f: getattr(config, f) for f in config.__dataclass_fields__},
               "training_mode": "qlora"}
        )
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.WINDOWS_NATIVE
        mock_detect.return_value = mock_env
        _make_tracker_mock(mock_tracker_cls)

        from dojo.train.finetune import finetune
        with patch("dojo.train.finetune.check_license_allows_training", return_value=None):
            result = finetune(config, hal_profile, pack)

        mock_run_qlora.assert_called_once()
        assert result.steps_completed == 8


# ---------------------------------------------------------------------------
# finetune() dispatch — WSL_UNSLOTH backend
# ---------------------------------------------------------------------------

class TestFinetuneWSLUnslothDispatch:
    @patch("dojo.train.finetune.detect_runtime")
    @patch("dojo.train.finetune.wsl_available", return_value=True)
    @patch("dojo.train.finetune.ensure_training_deps")
    @patch("dojo.train.finetune.ExperimentTracker")
    def test_wsl_unsloth_delegates_to_run_training_in_wsl(
        self, mock_tracker_cls, mock_ensure_deps, mock_wsl_available, mock_detect,
        trl_finetune_fixtures, tmp_path
    ):
        config, hal_profile, pack = trl_finetune_fixtures
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.WINDOWS_WSL
        mock_detect.return_value = mock_env
        _make_tracker_mock(mock_tracker_cls)

        mock_wsl_result = MagicMock()
        mock_wsl_result.steps_completed = 15
        mock_wsl_result.final_loss = 0.3

        from dojo.train.finetune import finetune
        with (
            patch("dojo.train.finetune.check_license_allows_training", return_value=None),
            patch("dojo.train.finetune.run_training_in_wsl", return_value=mock_wsl_result) as mock_wsl_fn,
        ):
            result = finetune(config, hal_profile, pack)

        mock_wsl_fn.assert_called_once()
        assert result.steps_completed == 15
        assert result.final_loss == 0.3


# ---------------------------------------------------------------------------
# ensure_training_deps auto-called in finetune()
# ---------------------------------------------------------------------------

class TestAutoDepInstall:
    @patch("dojo.train.finetune.detect_runtime")
    @patch("dojo.train.finetune._run_real_sft", return_value=(5, 0.4))
    @patch("dojo.train.finetune.ExperimentTracker")
    def test_ensure_deps_called_with_backend(
        self, mock_tracker_cls, mock_run_sft, mock_detect,
        trl_finetune_fixtures,
    ):
        config, hal_profile, pack = trl_finetune_fixtures
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.WINDOWS_NATIVE
        mock_detect.return_value = mock_env
        _make_tracker_mock(mock_tracker_cls)

        from dojo.train.finetune import finetune
        with (
            patch("dojo.train.finetune.check_license_allows_training", return_value=None),
            patch("dojo.train.finetune.ensure_training_deps") as mock_ensure,
        ):
            finetune(config, hal_profile, pack)

        mock_ensure.assert_called_once()
        call_args = mock_ensure.call_args
        assert call_args[0][0] == TrainingBackend.TRL

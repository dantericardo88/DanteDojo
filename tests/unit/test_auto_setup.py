"""Unit tests for dojo/train/auto_setup.py — cross-platform dep installer."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from dojo.core.types import TrainingBackend
from dojo.hal.env_detect import RuntimeEnv, _clear_cache
from dojo.train.auto_setup import (
    SetupResult,
    _install_in_wsl,
    _is_importable,
    _pip_to_import_name,
    _resolve_backend_key,
    check_deps_present,
    ensure_training_deps,
    recommended_backend_for_platform,
)


@pytest.fixture(autouse=True)
def clear_env_cache():
    _clear_cache()
    yield
    _clear_cache()


# ---------------------------------------------------------------------------
# _pip_to_import_name
# ---------------------------------------------------------------------------

class TestPipToImportName:
    def test_mlx_lm_maps_to_mlx_lm(self):
        assert _pip_to_import_name("mlx-lm") == "mlx_lm"

    def test_scikit_learn_maps_to_sklearn(self):
        assert _pip_to_import_name("scikit-learn") == "sklearn"

    def test_unsloth_colab_maps_to_unsloth(self):
        assert _pip_to_import_name("unsloth[colab-new]") == "unsloth"

    def test_plain_trl_unchanged(self):
        assert _pip_to_import_name("trl") == "trl"

    def test_plain_transformers_unchanged(self):
        assert _pip_to_import_name("transformers") == "transformers"


# ---------------------------------------------------------------------------
# _resolve_backend_key
# ---------------------------------------------------------------------------

class TestResolveBackendKey:
    def test_trl_key(self):
        assert _resolve_backend_key(TrainingBackend.TRL, RuntimeEnv.WINDOWS_NATIVE) == "trl"

    def test_unsloth_key(self):
        assert _resolve_backend_key(TrainingBackend.UNSLOTH, RuntimeEnv.LINUX) == "unsloth"

    def test_wsl_unsloth_maps_to_unsloth_key(self):
        assert _resolve_backend_key(TrainingBackend.WSL_UNSLOTH, RuntimeEnv.WINDOWS_WSL) == "unsloth"

    def test_mlx_key(self):
        assert _resolve_backend_key(TrainingBackend.MLX, RuntimeEnv.MACOS) == "mlx"

    def test_native_key(self):
        assert _resolve_backend_key(TrainingBackend.NATIVE, RuntimeEnv.LINUX) == "native"

    def test_axolotl_key(self):
        assert _resolve_backend_key(TrainingBackend.AXOLOTL, RuntimeEnv.LINUX) == "axolotl"


# ---------------------------------------------------------------------------
# _is_importable
# ---------------------------------------------------------------------------

class TestIsImportable:
    def test_sys_always_importable(self):
        from dojo.train.auto_setup import _is_importable
        assert _is_importable("sys") is True

    def test_nonexistent_pkg_not_importable(self):
        from dojo.train.auto_setup import _is_importable
        assert _is_importable("_this_pkg_does_not_exist_xyzzy") is False


# ---------------------------------------------------------------------------
# ensure_training_deps — dry run
# ---------------------------------------------------------------------------

class TestEnsureTrainingDepsDryRun:
    @patch("dojo.train.auto_setup.detect_runtime")
    @patch("dojo.train.auto_setup._is_importable", return_value=True)
    @patch("dojo.train.auto_setup.EvidenceLogger")
    def test_all_present_no_installs(self, mock_logger_cls, mock_importable, mock_detect):
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.WINDOWS_NATIVE
        mock_detect.return_value = mock_env
        mock_logger = MagicMock()
        mock_logger_cls.get_instance.return_value = mock_logger

        result = ensure_training_deps(TrainingBackend.TRL, dry_run=True)

        assert result.success is True
        assert result.failed == []
        assert result.installed == []
        assert result.backend == "trl"

    @patch("dojo.train.auto_setup.detect_runtime")
    @patch("dojo.train.auto_setup._is_importable", return_value=False)
    @patch("dojo.train.auto_setup.EvidenceLogger")
    def test_all_missing_dry_run_does_not_install(self, mock_logger_cls, mock_importable, mock_detect):
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.LINUX
        mock_detect.return_value = mock_env
        mock_logger = MagicMock()
        mock_logger_cls.get_instance.return_value = mock_logger

        result = ensure_training_deps(TrainingBackend.TRL, dry_run=True)

        assert result.failed == []  # dry_run — nothing attempted
        assert result.installed == []
        assert len(result.already_present) == 0

    @patch("dojo.train.auto_setup.detect_runtime")
    @patch("dojo.train.auto_setup._is_importable", return_value=True)
    @patch("dojo.train.auto_setup.EvidenceLogger")
    def test_emits_evidence(self, mock_logger_cls, mock_importable, mock_detect):
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.WINDOWS_NATIVE
        mock_detect.return_value = mock_env
        mock_logger = MagicMock()
        mock_logger_cls.get_instance.return_value = mock_logger

        ensure_training_deps(TrainingBackend.TRL, dry_run=True)

        mock_logger.emit.assert_called_once()
        call_args = mock_logger.emit.call_args[0][0]
        assert call_args["type"] == "train.auto_setup"
        assert call_args["data"]["backend"] == "trl"


# ---------------------------------------------------------------------------
# ensure_training_deps — real install (mocked subprocess)
# ---------------------------------------------------------------------------

class TestEnsureTrainingDepsInstall:
    @patch("dojo.train.auto_setup.detect_runtime")
    @patch("dojo.train.auto_setup._is_importable", return_value=False)
    @patch("dojo.train.auto_setup._install_package", return_value=True)
    @patch("dojo.train.auto_setup.EvidenceLogger")
    def test_missing_packages_installed(self, mock_logger_cls, mock_install, mock_importable, mock_detect):
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.WINDOWS_NATIVE
        mock_detect.return_value = mock_env
        mock_logger = MagicMock()
        mock_logger_cls.get_instance.return_value = mock_logger

        result = ensure_training_deps(TrainingBackend.TRL)

        assert len(result.installed) > 0
        assert result.failed == []
        assert result.success is True

    @patch("dojo.train.auto_setup.detect_runtime")
    @patch("dojo.train.auto_setup._is_importable", return_value=False)
    @patch("dojo.train.auto_setup._install_package", return_value=False)
    @patch("dojo.train.auto_setup.EvidenceLogger")
    def test_failed_install_reported(self, mock_logger_cls, mock_install, mock_importable, mock_detect):
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.WINDOWS_NATIVE
        mock_detect.return_value = mock_env
        mock_logger = MagicMock()
        mock_logger_cls.get_instance.return_value = mock_logger

        result = ensure_training_deps(TrainingBackend.TRL)

        assert result.success is False
        assert len(result.failed) > 0


# ---------------------------------------------------------------------------
# recommended_backend_for_platform
# ---------------------------------------------------------------------------

class TestRecommendedBackend:
    @patch("dojo.train.auto_setup.detect_runtime")
    def test_wsl2_with_unsloth_installed(self, mock_detect):
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.WSL2
        mock_detect.return_value = mock_env
        with patch("dojo.train.auto_setup._is_importable", return_value=True):
            backend = recommended_backend_for_platform()
        assert backend == TrainingBackend.UNSLOTH

    @patch("dojo.train.auto_setup.detect_runtime")
    def test_wsl2_without_unsloth_falls_back_to_trl(self, mock_detect):
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.WSL2
        mock_detect.return_value = mock_env
        with patch("dojo.train.auto_setup._is_importable", return_value=False):
            backend = recommended_backend_for_platform()
        assert backend == TrainingBackend.TRL

    @patch("dojo.train.auto_setup.detect_runtime")
    def test_windows_native_returns_trl(self, mock_detect):
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.WINDOWS_NATIVE
        mock_detect.return_value = mock_env
        backend = recommended_backend_for_platform()
        assert backend == TrainingBackend.TRL

    @patch("dojo.train.auto_setup.detect_runtime")
    def test_windows_wsl_returns_wsl_unsloth(self, mock_detect):
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.WINDOWS_WSL
        mock_detect.return_value = mock_env
        backend = recommended_backend_for_platform()
        assert backend == TrainingBackend.WSL_UNSLOTH

    @patch("dojo.train.auto_setup.detect_runtime")
    def test_macos_returns_mlx(self, mock_detect):
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.MACOS
        mock_detect.return_value = mock_env
        backend = recommended_backend_for_platform()
        assert backend == TrainingBackend.MLX

    @patch("dojo.train.auto_setup.detect_runtime")
    def test_linux_with_unsloth_returns_unsloth(self, mock_detect):
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.LINUX
        mock_detect.return_value = mock_env
        with patch("dojo.train.auto_setup._is_importable", return_value=True):
            backend = recommended_backend_for_platform()
        assert backend == TrainingBackend.UNSLOTH

    @patch("dojo.train.auto_setup.detect_runtime")
    def test_unknown_platform_returns_native(self, mock_detect):
        mock_env = MagicMock()
        mock_env.runtime = RuntimeEnv.UNKNOWN
        mock_detect.return_value = mock_env
        backend = recommended_backend_for_platform()
        assert backend == TrainingBackend.NATIVE


# ---------------------------------------------------------------------------
# SetupResult.success
# ---------------------------------------------------------------------------

class TestSetupResult:
    def test_success_when_no_failed(self):
        r = SetupResult(backend="trl", platform="windows_native",
                        already_present=["trl"], installed=[], failed=[])
        assert r.success is True

    def test_failure_when_any_failed(self):
        r = SetupResult(backend="trl", platform="windows_native",
                        already_present=[], installed=[], failed=["peft"])
        assert r.success is False

"""Unit tests for dojo/hal/env_detect.py — platform detection."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest

from dojo.hal.env_detect import (
    EnvInfo,
    RuntimeEnv,
    _clear_cache,
    detect_runtime,
    is_linux,
    is_macos,
    is_windows,
    is_wsl,
    trl_installable,
    unsloth_installable,
    wsl_available,
)


@pytest.fixture(autouse=True)
def clear_detect_cache():
    """Ensure lru_cache is cleared before and after every test."""
    _clear_cache()
    yield
    _clear_cache()


# ---------------------------------------------------------------------------
# detect_runtime — WSL2
# ---------------------------------------------------------------------------

class TestDetectWSL2:
    def test_wsl2_detected_via_microsoft_in_proc_version(self):
        with (
            patch("sys.platform", "linux"),
            patch("builtins.open", mock_open(read_data="Linux version 5.15.90 Microsoft WSL2\n")),
            patch("os.environ.get", side_effect=lambda k, d="": "Ubuntu-22.04" if k == "WSL_DISTRO_NAME" else d),
            patch("dojo.hal.env_detect._cuda_available", return_value=False),
        ):
            env = detect_runtime()
        assert env.runtime == RuntimeEnv.WSL2
        assert env.wsl_distro == "Ubuntu-22.04"
        assert env.platform == "linux"

    def test_wsl2_detected_via_lowercase_wsl_string(self):
        with (
            patch("sys.platform", "linux"),
            patch("builtins.open", mock_open(read_data="linux version 5.15 wsl\n")),
            patch("os.environ.get", side_effect=lambda k, d="": d),
            patch("dojo.hal.env_detect._cuda_available", return_value=False),
        ):
            env = detect_runtime()
        assert env.runtime == RuntimeEnv.WSL2
        assert env.wsl_distro == "wsl2"  # fallback default returned by os.environ.get

    def test_native_linux_when_proc_version_has_no_microsoft(self):
        with (
            patch("sys.platform", "linux"),
            patch("builtins.open", mock_open(read_data="Linux version 5.15 Ubuntu\n")),
            patch("dojo.hal.env_detect._cuda_available", return_value=False),
        ):
            env = detect_runtime()
        assert env.runtime == RuntimeEnv.LINUX

    def test_native_linux_when_proc_version_unreadable(self):
        with (
            patch("sys.platform", "linux"),
            patch("builtins.open", side_effect=OSError("no file")),
            patch("dojo.hal.env_detect._cuda_available", return_value=False),
        ):
            env = detect_runtime()
        assert env.runtime == RuntimeEnv.LINUX


# ---------------------------------------------------------------------------
# detect_runtime — macOS
# ---------------------------------------------------------------------------

class TestDetectMacOS:
    def test_macos_detected(self):
        with (
            patch("sys.platform", "darwin"),
            patch("dojo.hal.env_detect._cuda_available", return_value=False),
        ):
            env = detect_runtime()
        assert env.runtime == RuntimeEnv.MACOS
        assert env.wsl_distro == ""

    def test_macos_cuda_false(self):
        with (
            patch("sys.platform", "darwin"),
            patch("dojo.hal.env_detect._cuda_available", return_value=False),
        ):
            env = detect_runtime()
        assert env.cuda_visible is False


# ---------------------------------------------------------------------------
# detect_runtime — Windows native / Windows+WSL
# ---------------------------------------------------------------------------

class TestDetectWindows:
    def test_windows_native_no_wsl(self):
        with (
            patch("sys.platform", "win32"),
            patch("dojo.hal.env_detect._probe_wsl", return_value=("", False)),
            patch("dojo.hal.env_detect._cuda_available", return_value=True),
        ):
            env = detect_runtime()
        assert env.runtime == RuntimeEnv.WINDOWS_NATIVE
        assert env.cuda_visible is True

    def test_windows_with_wsl_available(self):
        with (
            patch("sys.platform", "win32"),
            patch("dojo.hal.env_detect._probe_wsl", return_value=("Ubuntu", True)),
            patch("dojo.hal.env_detect._cuda_available", return_value=False),
        ):
            env = detect_runtime()
        assert env.runtime == RuntimeEnv.WINDOWS_WSL
        assert env.wsl_distro == "Ubuntu"


# ---------------------------------------------------------------------------
# Helper predicates
# ---------------------------------------------------------------------------

class TestHelperPredicates:
    def test_is_windows_true_for_native(self):
        with (
            patch("sys.platform", "win32"),
            patch("dojo.hal.env_detect._probe_wsl", return_value=("", False)),
            patch("dojo.hal.env_detect._cuda_available", return_value=False),
        ):
            assert is_windows() is True

    def test_is_windows_true_for_windows_wsl(self):
        with (
            patch("sys.platform", "win32"),
            patch("dojo.hal.env_detect._probe_wsl", return_value=("Ubuntu", True)),
            patch("dojo.hal.env_detect._cuda_available", return_value=False),
        ):
            assert is_windows() is True

    def test_is_wsl_true_when_inside_wsl2(self):
        with (
            patch("sys.platform", "linux"),
            patch("builtins.open", mock_open(read_data="Microsoft WSL2")),
            patch("os.environ.get", side_effect=lambda k, d="": "Ubuntu" if k == "WSL_DISTRO_NAME" else d),
            patch("dojo.hal.env_detect._cuda_available", return_value=False),
        ):
            assert is_wsl() is True

    def test_is_linux_true_for_native_linux(self):
        with (
            patch("sys.platform", "linux"),
            patch("builtins.open", side_effect=OSError),
            patch("dojo.hal.env_detect._cuda_available", return_value=False),
        ):
            assert is_linux() is True

    def test_is_macos_true(self):
        with (
            patch("sys.platform", "darwin"),
            patch("dojo.hal.env_detect._cuda_available", return_value=False),
        ):
            assert is_macos() is True

    def test_wsl_available_only_from_windows_wsl(self):
        with (
            patch("sys.platform", "win32"),
            patch("dojo.hal.env_detect._probe_wsl", return_value=("Ubuntu", True)),
            patch("dojo.hal.env_detect._cuda_available", return_value=False),
        ):
            assert wsl_available() is True

    def test_wsl_available_false_on_native_windows(self):
        with (
            patch("sys.platform", "win32"),
            patch("dojo.hal.env_detect._probe_wsl", return_value=("", False)),
            patch("dojo.hal.env_detect._cuda_available", return_value=False),
        ):
            assert wsl_available() is False

    def test_unsloth_installable_on_linux(self):
        with (
            patch("sys.platform", "linux"),
            patch("builtins.open", side_effect=OSError),
            patch("dojo.hal.env_detect._cuda_available", return_value=True),
        ):
            assert unsloth_installable() is True

    def test_unsloth_not_installable_on_windows_native(self):
        with (
            patch("sys.platform", "win32"),
            patch("dojo.hal.env_detect._probe_wsl", return_value=("", False)),
            patch("dojo.hal.env_detect._cuda_available", return_value=True),
        ):
            assert unsloth_installable() is False

    def test_trl_installable_on_windows(self):
        with (
            patch("sys.platform", "win32"),
            patch("dojo.hal.env_detect._probe_wsl", return_value=("", False)),
            patch("dojo.hal.env_detect._cuda_available", return_value=False),
        ):
            assert trl_installable() is True

    def test_trl_installable_on_linux(self):
        with (
            patch("sys.platform", "linux"),
            patch("builtins.open", side_effect=OSError),
            patch("dojo.hal.env_detect._cuda_available", return_value=False),
        ):
            assert trl_installable() is True


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------

class TestCacheBehaviour:
    def test_result_is_cached_second_call_not_re_detected(self):
        call_count = 0

        def counting_probe():
            nonlocal call_count
            call_count += 1
            return ("", False)

        with (
            patch("sys.platform", "win32"),
            patch("dojo.hal.env_detect._probe_wsl", side_effect=counting_probe),
            patch("dojo.hal.env_detect._cuda_available", return_value=False),
        ):
            detect_runtime()
            detect_runtime()
        assert call_count == 1  # only probed once

    def test_clear_cache_allows_re_detection(self):
        with (
            patch("sys.platform", "win32"),
            patch("dojo.hal.env_detect._probe_wsl", return_value=("", False)),
            patch("dojo.hal.env_detect._cuda_available", return_value=False),
        ):
            env1 = detect_runtime()

        _clear_cache()

        with (
            patch("sys.platform", "linux"),
            patch("builtins.open", side_effect=OSError),
            patch("dojo.hal.env_detect._cuda_available", return_value=False),
        ):
            env2 = detect_runtime()

        assert env1.runtime == RuntimeEnv.WINDOWS_NATIVE
        assert env2.runtime == RuntimeEnv.LINUX

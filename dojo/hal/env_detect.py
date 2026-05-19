"""dojo/hal/env_detect.py — Runtime environment detection.

Detects the host platform and available compute substrates so the training
router can select the correct backend without manual configuration.

Detection hierarchy (in order):
  1. Inside WSL2          — /proc/version contains 'microsoft' or 'WSL'
  2. Windows native       — sys.platform == 'win32'
  3. macOS                — sys.platform == 'darwin'
  4. Linux native         — sys.platform == 'linux' (not WSL)
  5. Unknown              — fallback

WSL2 GPU passthrough:
  When running Windows-native, wsl_available() checks whether wsl.exe is on
  PATH and can enumerate distros with GPU passthrough.  If so, training can
  be delegated to a WSL2 subprocess where unsloth and triton are installable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache


class RuntimeEnv(str, Enum):
    WSL2           = "wsl2"           # Running inside WSL2 (Linux kernel under Windows)
    WINDOWS_NATIVE = "windows_native" # Windows Python, no WSL subprocess possible
    WINDOWS_WSL    = "windows_wsl"    # Windows Python, WSL2 available for delegation
    MACOS          = "macos"
    LINUX          = "linux"          # Native Linux (bare-metal or Docker)
    UNKNOWN        = "unknown"


@dataclass(frozen=True)
class EnvInfo:
    runtime: RuntimeEnv
    platform: str           # sys.platform value
    wsl_distro: str         # non-empty if inside WSL or if WSL available from Windows
    cuda_visible: bool      # torch.cuda.is_available() result (cached at detection time)
    python_executable: str  # full path to the current Python binary


@lru_cache(maxsize=1)
def detect_runtime() -> EnvInfo:
    """Detect the current runtime environment.

    Cached after first call — environment does not change during a process
    lifetime.  Use ``_clear_cache()`` in tests to reset.

    Returns:
        EnvInfo with runtime classification and auxiliary facts.
    """
    platform = sys.platform

    # --- Inside WSL2? ---------------------------------------------------------
    if platform == "linux":
        try:
            with open("/proc/version") as fh:
                proc_version = fh.read().lower()
            if "microsoft" in proc_version or "wsl" in proc_version:
                distro = os.environ.get("WSL_DISTRO_NAME", "wsl2")
                return EnvInfo(
                    runtime=RuntimeEnv.WSL2,
                    platform=platform,
                    wsl_distro=distro,
                    cuda_visible=_cuda_available(),
                    python_executable=sys.executable,
                )
        except OSError:
            pass

    # --- macOS ---------------------------------------------------------------
    if platform == "darwin":
        return EnvInfo(
            runtime=RuntimeEnv.MACOS,
            platform=platform,
            wsl_distro="",
            cuda_visible=_cuda_available(),
            python_executable=sys.executable,
        )

    # --- Windows native -------------------------------------------------------
    if platform == "win32":
        distro, wsl_ok = _probe_wsl()
        runtime = RuntimeEnv.WINDOWS_WSL if wsl_ok else RuntimeEnv.WINDOWS_NATIVE
        return EnvInfo(
            runtime=runtime,
            platform=platform,
            wsl_distro=distro,
            cuda_visible=_cuda_available(),
            python_executable=sys.executable,
        )

    # --- Native Linux --------------------------------------------------------
    if platform == "linux":
        return EnvInfo(
            runtime=RuntimeEnv.LINUX,
            platform=platform,
            wsl_distro="",
            cuda_visible=_cuda_available(),
            python_executable=sys.executable,
        )

    return EnvInfo(
        runtime=RuntimeEnv.UNKNOWN,
        platform=platform,
        wsl_distro="",
        cuda_visible=_cuda_available(),
        python_executable=sys.executable,
    )


def is_windows() -> bool:
    """True when running Windows-native (not inside WSL)."""
    return detect_runtime().runtime in (
        RuntimeEnv.WINDOWS_NATIVE,
        RuntimeEnv.WINDOWS_WSL,
    )


def is_wsl() -> bool:
    """True when running inside WSL2."""
    return detect_runtime().runtime == RuntimeEnv.WSL2


def is_linux() -> bool:
    """True when running on native Linux or inside WSL2."""
    return detect_runtime().runtime in (RuntimeEnv.LINUX, RuntimeEnv.WSL2)


def is_macos() -> bool:
    """True when running on macOS."""
    return detect_runtime().runtime == RuntimeEnv.MACOS


def wsl_available() -> bool:
    """True when WSL2 is available for subprocess delegation from Windows."""
    return detect_runtime().runtime == RuntimeEnv.WINDOWS_WSL


def unsloth_installable() -> bool:
    """Return True if the current environment can install/use unsloth.

    Unsloth requires Linux + CUDA (works in WSL2 or native Linux).
    It does NOT work on Windows-native Python even with CUDA drivers.
    """
    env = detect_runtime()
    return env.runtime in (RuntimeEnv.WSL2, RuntimeEnv.LINUX)


def trl_installable() -> bool:
    """Return True if TRL + transformers + peft can be installed/used here.

    TRL works on Windows-native, Linux, and macOS.
    """
    return detect_runtime().runtime in (
        RuntimeEnv.WINDOWS_NATIVE,
        RuntimeEnv.WINDOWS_WSL,
        RuntimeEnv.LINUX,
        RuntimeEnv.WSL2,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _probe_wsl() -> tuple[str, bool]:
    """Check if wsl.exe is present and has at least one distro.

    Returns (distro_name, wsl_available).
    """
    wsl_exe = shutil.which("wsl.exe") or shutil.which("wsl")
    if not wsl_exe:
        return "", False

    try:
        result = subprocess.run(
            [wsl_exe, "--list", "--quiet"],
            capture_output=True,
            text=True,
            timeout=5,
            encoding="utf-16-le",   # wsl --list outputs UTF-16 LE on Windows
        )
        output = result.stdout.strip()
        if not output:
            # Try plain UTF-8 fallback
            result2 = subprocess.run(
                [wsl_exe, "--list", "--quiet"],
                capture_output=True,
                timeout=5,
            )
            output = result2.stdout.decode("utf-8", errors="ignore").strip()

        lines = [l.strip() for l in output.splitlines() if l.strip()]
        if lines:
            # First non-empty line is the default distro name
            return lines[0], True
        return "", False
    except Exception:
        return "", False


def _clear_cache() -> None:
    """Clear the lru_cache — for testing only."""
    detect_runtime.cache_clear()

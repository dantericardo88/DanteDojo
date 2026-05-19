"""dojo/train/auto_setup.py — Auto-install missing training dependencies.

On first call per process, checks which training packages are present and
installs any that are missing for the selected backend.  Uses pip subprocess
so it works regardless of how the parent process was launched.

Platform routing:
  Windows-native  → pip install transformers peft trl datasets accelerate
  Linux / WSL2    → pip install unsloth[colab-new] transformers trl datasets
  macOS           → pip install mlx-lm transformers trl datasets

The function is idempotent — if all required packages are already present it
returns immediately without spawning any subprocess.

All installs are emitted as constitutional evidence so the setup history is
auditable.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Sequence

from dojo.core.evidence import EvidenceLogger
from dojo.core.types import TrainingBackend
from dojo.hal.env_detect import RuntimeEnv, detect_runtime


# ---------------------------------------------------------------------------
# Package requirements per backend
# ---------------------------------------------------------------------------

_BACKEND_PACKAGES: dict[str, list[str]] = {
    "trl": [
        "transformers",
        "peft",
        "trl",
        "datasets",
        "accelerate",
    ],
    "unsloth": [
        "transformers",
        "trl",
        "datasets",
        "accelerate",
        # unsloth itself — special install handled below
    ],
    "mlx": [
        "mlx-lm",
        "transformers",
        "datasets",
    ],
    "axolotl": [
        "axolotl",
        "transformers",
        "peft",
        "trl",
        "datasets",
        "accelerate",
    ],
    "native": [],   # native_rl needs only torch (always present)
}

# Packages that need special install commands (not plain pip install <name>)
_SPECIAL_INSTALL: dict[str, list[str]] = {
    "unsloth": ["pip", "install", "unsloth[colab-new]", "--no-deps"],
    "mlx-lm": ["pip", "install", "mlx-lm"],
}


@dataclass(frozen=True)
class SetupResult:
    backend: str
    platform: str
    already_present: list[str]
    installed: list[str]
    failed: list[str]

    @property
    def success(self) -> bool:
        return len(self.failed) == 0


def ensure_training_deps(
    backend: TrainingBackend,
    *,
    dry_run: bool = False,
    quiet: bool = True,
) -> SetupResult:
    """Ensure all packages required for *backend* are installed.

    Args:
        backend: The resolved TrainingBackend for this machine.
        dry_run: If True, check presence but do not install anything.
        quiet: Suppress pip output (default True).

    Returns:
        SetupResult describing what was already present and what was installed.
    """
    env_info = detect_runtime()
    backend_key = _resolve_backend_key(backend, env_info.runtime)
    required = _BACKEND_PACKAGES.get(backend_key, [])

    present: list[str] = []
    to_install: list[str] = []

    for pkg in required:
        import_name = _pip_to_import_name(pkg)
        if _is_importable(import_name):
            present.append(pkg)
        else:
            to_install.append(pkg)

    # Also check unsloth itself when targeting unsloth backend
    if backend_key == "unsloth" and not _is_importable("unsloth"):
        to_install.append("unsloth")

    installed: list[str] = []
    failed: list[str] = []

    if not dry_run:
        for pkg in to_install:
            ok = _install_package(pkg, quiet=quiet, runtime=env_info.runtime)
            if ok:
                installed.append(pkg)
            else:
                failed.append(pkg)

    result = SetupResult(
        backend=backend_key,
        platform=env_info.runtime.value,
        already_present=present,
        installed=installed,
        failed=failed,
    )

    EvidenceLogger.get_instance().emit(
        {
            "type": "train.auto_setup",
            "data": {
                "backend": backend_key,
                "platform": env_info.runtime.value,
                "already_present": present,
                "installed": installed,
                "failed": failed,
                "dry_run": dry_run,
            },
        },
        organ="train",
    )

    return result


def recommended_backend_for_platform() -> TrainingBackend:
    """Return the best available TrainingBackend for the current platform.

    Selection priority:
      WSL2 / Linux     → UNSLOTH if installable, else TRL
      Windows-native   → TRL (unsloth not installable; TRL works natively)
      Windows + WSL2   → WSL_UNSLOTH (delegate to WSL subprocess)
      macOS            → MLX
      Unknown          → NATIVE (always works, zero deps beyond torch)
    """
    runtime = detect_runtime().runtime

    if runtime == RuntimeEnv.WSL2:
        return TrainingBackend.UNSLOTH if _is_importable("unsloth") else TrainingBackend.TRL
    if runtime == RuntimeEnv.LINUX:
        return TrainingBackend.UNSLOTH if _is_importable("unsloth") else TrainingBackend.TRL
    if runtime == RuntimeEnv.WINDOWS_WSL:
        return TrainingBackend.WSL_UNSLOTH
    if runtime == RuntimeEnv.WINDOWS_NATIVE:
        return TrainingBackend.TRL
    if runtime == RuntimeEnv.MACOS:
        return TrainingBackend.MLX
    return TrainingBackend.NATIVE


def check_deps_present(backend: TrainingBackend) -> bool:
    """Quick check — returns True if all required packages are importable right now."""
    result = ensure_training_deps(backend, dry_run=True)
    return result.success and len(result.failed) == 0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_backend_key(backend: TrainingBackend, runtime: RuntimeEnv) -> str:
    """Map TrainingBackend + RuntimeEnv → package group key."""
    if backend == TrainingBackend.TRL:
        return "trl"
    if backend in (TrainingBackend.UNSLOTH, TrainingBackend.WSL_UNSLOTH):
        return "unsloth"
    if backend == TrainingBackend.MLX:
        return "mlx"
    if backend == TrainingBackend.AXOLOTL:
        return "axolotl"
    return "native"


def _pip_to_import_name(pip_name: str) -> str:
    """Convert a pip package name to its Python import name."""
    mapping = {
        "scikit-learn": "sklearn",
        "pillow": "PIL",
        "mlx-lm": "mlx_lm",
        "unsloth[colab-new]": "unsloth",
    }
    return mapping.get(pip_name, pip_name.replace("-", "_"))


def _is_importable(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def _install_package(pkg: str, *, quiet: bool, runtime: RuntimeEnv) -> bool:
    """Install *pkg* via pip in the appropriate environment.

    For WSL_UNSLOTH packages that need Linux, delegates to wsl.exe subprocess.
    """
    if pkg in _SPECIAL_INSTALL:
        cmd = _SPECIAL_INSTALL[pkg]
    else:
        cmd = [sys.executable, "-m", "pip", "install", pkg]
        if quiet:
            cmd.append("--quiet")

    # If we're on Windows-native and the package requires Linux (e.g. unsloth),
    # route through WSL instead.
    if runtime == RuntimeEnv.WINDOWS_WSL and pkg in ("unsloth",):
        return _install_in_wsl(pkg, quiet=quiet)

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        return result.returncode == 0
    except Exception:
        return False


def _install_in_wsl(pkg: str, *, quiet: bool) -> bool:
    """Install *pkg* inside WSL using wsl.exe pip."""
    import shutil
    wsl_exe = shutil.which("wsl.exe") or shutil.which("wsl")
    if not wsl_exe:
        return False

    wsl_cmd = f"python3 -m pip install {pkg}"
    if quiet:
        wsl_cmd += " --quiet"

    try:
        result = subprocess.run(
            [wsl_exe, "--", "bash", "-c", wsl_cmd],
            capture_output=True,
            timeout=300,
        )
        return result.returncode == 0
    except Exception:
        return False

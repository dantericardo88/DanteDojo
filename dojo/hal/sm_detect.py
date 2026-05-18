"""GPU SM (compute capability) detection with nvidia-ml-py preferred (d09).

The inspector flagged d09 for using the deprecated ``pynvml`` package.  The
official replacement is ``nvidia-ml-py`` (also called ``pynvml`` upstream but
under the ``nvidia-ml-py`` distribution name).  This module preferentially
imports the new package, falls back to the legacy import path, and exposes
a clean API used by the FP8 gate and HAL probe.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dojo.core.evidence import EvidenceLogger
from dojo.core.exceptions import DojoError


class SmDetectError(DojoError):
    """Raised when SM detection cannot complete."""


@dataclass(frozen=True)
class SmInfo:
    """Detected GPU SM info."""

    available: bool
    sm_major: int
    sm_minor: int
    gpu_name: str
    source: str  # "nvidia-ml-py" | "pynvml-legacy" | "torch.cuda" | "env" | "none"

    @property
    def sm_float(self) -> float:
        """SM as a float, e.g. 8.9 for Ada Lovelace."""
        if not self.available:
            return 0.0
        return float(f"{self.sm_major}.{self.sm_minor}")


def _try_nvidia_ml_py() -> Optional[SmInfo]:
    """Try the modern nvidia-ml-py path."""
    try:
        import pynvml  # nvidia-ml-py exposes this name too
        if not hasattr(pynvml, "nvmlInit"):
            return None
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            # Modern nvidia-ml-py reports a __version__ string distinguishing
            # it from the legacy pynvml package.
            version = getattr(pynvml, "__version__", "")
            source = "nvidia-ml-py" if version else "pynvml-legacy"
            return SmInfo(
                available=True,
                sm_major=int(major),
                sm_minor=int(minor),
                gpu_name=str(name),
                source=source,
            )
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        return None


def _try_torch_cuda() -> Optional[SmInfo]:
    """Try ``torch.cuda.get_device_capability`` as a backup."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        return SmInfo(
            available=True,
            sm_major=int(major),
            sm_minor=int(minor),
            gpu_name=str(name),
            source="torch.cuda",
        )
    except Exception:
        return None


def _try_env_var() -> Optional[SmInfo]:
    """Last-resort: read DOJO_MOCK_SM (for CI/testing only)."""
    val = os.environ.get("DOJO_MOCK_SM")
    if not val:
        return None
    try:
        major_str, minor_str = val.split(".")
        return SmInfo(
            available=True,
            sm_major=int(major_str),
            sm_minor=int(minor_str),
            gpu_name="env:DOJO_MOCK_SM",
            source="env",
        )
    except (ValueError, AttributeError):
        return None


def detect_sm() -> SmInfo:
    """Detect the GPU's SM version.

    Probe order:
      1. nvidia-ml-py (preferred — pynvml deprecated)
      2. torch.cuda  (works when CUDA toolkit is loaded)
      3. DOJO_MOCK_SM env var (CI/testing only)
      4. ``SmInfo(available=False, ...)`` when nothing is available.

    Emits ``hal.sm_detect`` evidence with the chosen source.
    """
    for probe in (_try_nvidia_ml_py, _try_torch_cuda, _try_env_var):
        info = probe()
        if info is not None:
            EvidenceLogger.get_instance().emit(
                {
                    "type": "hal.sm_detect",
                    "data": {
                        "sm": info.sm_float,
                        "source": info.source,
                        "gpu_name": info.gpu_name,
                    },
                },
                organ="hal",
            )
            return info

    fallback = SmInfo(
        available=False,
        sm_major=0,
        sm_minor=0,
        gpu_name="cpu-only",
        source="none",
    )
    EvidenceLogger.get_instance().emit(
        {
            "type": "hal.sm_detect",
            "data": {"sm": 0.0, "source": "none", "gpu_name": "cpu-only"},
        },
        organ="hal",
    )
    return fallback


def sm_supports_fp8(info: Optional[SmInfo] = None) -> bool:
    """True iff the detected SM supports native FP8 (Ada Lovelace 8.9 / Hopper 9.0+)."""
    info = info or detect_sm()
    return info.available and info.sm_float >= 8.9


def sm_supports_fp4(info: Optional[SmInfo] = None) -> bool:
    """True iff the detected SM supports native FP4 (Blackwell 10.0+)."""
    info = info or detect_sm()
    return info.available and info.sm_float >= 10.0


__all__ = [
    "SmInfo",
    "SmDetectError",
    "detect_sm",
    "sm_supports_fp8",
    "sm_supports_fp4",
]

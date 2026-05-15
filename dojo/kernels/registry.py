"""DanteDojo kernel registry — probes and exposes available compute kernels.

The ``KernelRegistry`` is a singleton that surveys the runtime environment
(Triton availability, CUDA availability, environment flags) and reports which
kernel implementations are ready.  Every probe call emits a
``TRAINING_BACKEND_SELECTED`` evidence bundle so the selection is stamped into
the Merkle chain.

Constitutional guarantees:
- ``probe()`` is idempotent — repeated calls re-emit evidence but do not
  change the returned dict.
- ``get_best_attention()`` and ``get_best_rms_norm()`` always return a valid
  string; they never raise.
- ``triton`` is imported only inside try/except — never at module level.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _triton_importable() -> bool:
    try:
        import importlib
        importlib.import_module("triton")
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _cuda_available() -> bool:
    try:
        import torch  # noqa: PLC0415
        return torch.cuda.is_available()
    except (ImportError, ModuleNotFoundError):
        return False


def _real_gpu_requested() -> bool:
    return os.environ.get("DOJO_REAL_GPU", "0") == "1"


# ---------------------------------------------------------------------------
# KernelRegistry
# ---------------------------------------------------------------------------

class KernelRegistry:
    """Registry of available constitutional kernels.

    Usage::

        reg = KernelRegistry()
        avail = reg.probe()
        backend = reg.get_best_attention()  # "triton" | "pytorch"
    """

    _instance_lock = threading.Lock()
    _singleton: Optional["KernelRegistry"] = None

    def __init__(self) -> None:
        self._kernels: dict[str, bool] = {}
        self._probed = False

    # ------------------------------------------------------------------
    # Singleton access (optional — callers may also instantiate directly)
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "KernelRegistry":
        with cls._instance_lock:
            if cls._singleton is None:
                cls._singleton = KernelRegistry()
            return cls._singleton

    @classmethod
    def clear(cls) -> None:
        """Reset singleton (for tests)."""
        with cls._instance_lock:
            cls._singleton = None

    # ------------------------------------------------------------------
    # Core probe
    # ------------------------------------------------------------------

    def probe(self) -> dict[str, bool]:
        """Probe which kernels are available. Returns ``{kernel_name: available}``.

        Checks:
        - ``triton``   — Triton package importable.
        - ``cuda``     — CUDA device available via PyTorch.
        - ``real_gpu`` — ``DOJO_REAL_GPU=1`` env var is set.
        - ``triton_flash_attn``   — Triton + CUDA + real_gpu all true.
        - ``triton_rms_norm``     — Same preconditions as flash_attn.
        - ``pytorch_flash_attn``  — Always available (PyTorch is required).
        - ``pytorch_rms_norm``    — Always available.

        Emits ``TRAINING_BACKEND_SELECTED`` evidence with full results.
        """
        triton_ok = _triton_importable()
        cuda_ok   = _cuda_available()
        gpu_flag  = _real_gpu_requested()

        # Triton kernels require triton + cuda + explicit GPU flag
        triton_ready = triton_ok and cuda_ok and gpu_flag

        results: dict[str, bool] = {
            "triton":              triton_ok,
            "cuda":                cuda_ok,
            "real_gpu":            gpu_flag,
            "triton_flash_attn":   triton_ready,
            "triton_rms_norm":     triton_ready,
            "pytorch_flash_attn":  True,   # always available
            "pytorch_rms_norm":    True,   # always available
        }

        self._kernels = results
        self._probed = True

        logger = EvidenceLogger.get_instance()
        logger.emit(
            {
                "type": EvidenceType.TRAINING_BACKEND_SELECTED.value,
                "data": {
                    "subsystem": "kernel_registry",
                    "probe_results": results,
                    "selected_attention": self.get_best_attention(),
                    "selected_rms_norm":  self.get_best_rms_norm(),
                },
            },
            organ="kernels",
        )

        return results

    # ------------------------------------------------------------------
    # Best-backend selectors
    # ------------------------------------------------------------------

    def get_best_attention(self) -> str:
        """Return ``'triton'`` or ``'pytorch'`` based on availability."""
        if not self._probed:
            self.probe()
        return "triton" if self._kernels.get("triton_flash_attn", False) else "pytorch"

    def get_best_rms_norm(self) -> str:
        """Return ``'triton'`` or ``'pytorch'`` based on availability."""
        if not self._probed:
            self.probe()
        return "triton" if self._kernels.get("triton_rms_norm", False) else "pytorch"


__all__ = ["KernelRegistry"]

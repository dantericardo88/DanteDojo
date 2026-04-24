"""dojo/hal/vram.py — VRAM monitor.

Wraps pynvml to query per-device VRAM usage.  If pynvml is unavailable or
CUDA is not present the monitor falls back to conservative mock values
(8 GiB total, 7 GiB free) so callers are never broken by a missing driver.
"""

from __future__ import annotations

_MOCK_TOTAL = 8 * 1024 ** 3   # 8 GiB
_MOCK_FREE  = 7 * 1024 ** 3   # 7 GiB


def _try_nvml_meminfo(device: int):  # type: ignore[return]
    """Return pynvml MemoryInfo for *device* or None on any failure."""
    try:
        import pynvml  # type: ignore[import-not-found]
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        return pynvml.nvmlDeviceGetMemoryInfo(handle)
    except Exception:
        return None


class VRAMMonitor:
    """Query per-device VRAM statistics.

    All methods fall back gracefully to mock values when pynvml is unavailable
    or the requested device does not exist — they never raise.
    """

    def get_total_bytes(self, device: int = 0) -> int:
        """Return total VRAM in bytes for *device*.

        Returns the mock value (8 GiB) when pynvml is unavailable.
        """
        info = _try_nvml_meminfo(device)
        if info is None:
            return _MOCK_TOTAL
        return int(info.total)

    def get_free_bytes(self, device: int = 0) -> int:
        """Return free (available) VRAM in bytes for *device*.

        Returns the mock value (7 GiB) when pynvml is unavailable.
        """
        info = _try_nvml_meminfo(device)
        if info is None:
            return _MOCK_FREE
        return int(info.free)

    def get_used_bytes(self, device: int = 0) -> int:
        """Return used VRAM in bytes for *device*.

        Returns total − free (1 GiB mock) when pynvml is unavailable.
        """
        info = _try_nvml_meminfo(device)
        if info is None:
            return _MOCK_TOTAL - _MOCK_FREE
        return int(info.used)

    def exceeds_budget(self, required_bytes: int, device: int = 0) -> bool:
        """Return True if *required_bytes* exceeds available free VRAM.

        Uses mock free VRAM when pynvml is unavailable, so callers that ask
        for more than 7 GiB on an unknown system will get True.
        """
        return required_bytes > self.get_free_bytes(device)


__all__ = ["VRAMMonitor"]

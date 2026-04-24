"""DanteDojo resource guard — VRAM availability check combining all guards.

Aggregates ThermalGuard, PowerGuard, and a VRAM check into a single
``all_guards_clear()`` method used by the idle training loop.
"""

from __future__ import annotations

from dojo.idle.guardian import PowerGuard, ThermalGuard


class ResourceGuard:
    """Aggregate guard that checks thermal, power, and VRAM availability."""

    VRAM_RESERVE_BYTES: int = 512 * 1024 * 1024  # 512 MB reserve for foreground apps

    def __init__(self, device: int = 0) -> None:
        self._device = device
        self._thermal = ThermalGuard()
        self._power = PowerGuard()

    # ------------------------------------------------------------------
    # VRAM check
    # ------------------------------------------------------------------

    def check_vram_available(self, required_bytes: int) -> bool:
        """Return ``True`` if enough free VRAM is available after the reserve.

        Tries pynvml to query free VRAM.  If pynvml is unavailable, returns
        ``True`` (optimistic — caller is responsible for handling OOM).
        """
        try:
            import pynvml  # type: ignore[import-not-found]
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self._device)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_usable = mem_info.free - self.VRAM_RESERVE_BYTES
            return free_usable >= required_bytes
        except Exception:
            # pynvml unavailable or query failed — optimistic pass-through
            return True

    # ------------------------------------------------------------------
    # Aggregate guard
    # ------------------------------------------------------------------

    def all_guards_clear(
        self, required_vram_bytes: int
    ) -> tuple[bool, list[str]]:
        """Check all guards and return ``(can_train, blocking_reasons)``.

        Returns ``(True, [])`` if no guard is blocking.
        Returns ``(False, [reason, ...])`` with reasons for each failing guard.
        """
        blocking: list[str] = []

        if self._thermal.should_pause():
            temp = self._thermal.get_gpu_temp_celsius(self._device)
            blocking.append(
                f"Thermal pause: GPU temp {temp:.1f}°C "
                f"exceeds {ThermalGuard.PAUSE_THRESHOLD_CELSIUS}°C"
            )

        if self._power.should_pause():
            blocking.append("Power pause: system is running on battery")

        if not self.check_vram_available(required_vram_bytes):
            blocking.append(
                f"VRAM insufficient: need {required_vram_bytes / 1e6:.0f} MB "
                f"(+{self.VRAM_RESERVE_BYTES / 1e6:.0f} MB reserve)"
            )

        return (len(blocking) == 0, blocking)


__all__ = ["ResourceGuard"]

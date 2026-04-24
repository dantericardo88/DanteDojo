"""DanteDojo idle training guardians — thermal and power safety checks.

ThermalGuard: pauses idle training when GPU temperature is too high.
PowerGuard: pauses idle training when the laptop is on battery power.
"""

from __future__ import annotations

import platform


class ThermalGuard:
    """Monitor GPU temperature and gate idle training on safe thermal state."""

    PAUSE_THRESHOLD_CELSIUS: float = 80.0
    HYSTERESIS_CELSIUS: float = 5.0   # Resume when temp < 75°C

    def get_gpu_temp_celsius(self, device: int = 0) -> float:
        """Return GPU temperature in Celsius.

        Tries pynvml first.  Returns ``0.0`` if pynvml is unavailable or
        the query fails — 0.0 is safe (below any pause threshold).
        """
        try:
            import pynvml  # type: ignore[import-not-found]
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            temp: float = float(pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            ))
            return temp
        except Exception:
            return 0.0

    def should_pause(self) -> bool:
        """Return ``True`` if GPU temperature exceeds the pause threshold."""
        return self.get_gpu_temp_celsius() > self.PAUSE_THRESHOLD_CELSIUS

    def can_resume(self) -> bool:
        """Return ``True`` if GPU temperature has cooled below the resume threshold."""
        resume_threshold = self.PAUSE_THRESHOLD_CELSIUS - self.HYSTERESIS_CELSIUS
        return self.get_gpu_temp_celsius() <= resume_threshold


class PowerGuard:
    """Stop idle training when the laptop is running on battery."""

    def is_on_battery(self) -> bool:
        """Return ``True`` if the system is on battery (not plugged in).

        Uses ``psutil.sensors_battery()``.  Returns ``False`` if psutil is
        unavailable or the platform has no battery sensor (desktops).
        """
        try:
            import psutil  # type: ignore[import-not-found]
            battery = psutil.sensors_battery()
            if battery is None:
                # Desktop or unsupported platform — assume plugged in
                return False
            return not battery.power_plugged
        except Exception:
            return False

    def should_pause(self) -> bool:
        """Return ``True`` if on battery power (idle training should stop)."""
        return self.is_on_battery()


__all__ = ["PowerGuard", "ThermalGuard"]

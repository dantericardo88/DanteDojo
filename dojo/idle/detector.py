"""DanteDojo idle detector — platform-specific user-idle detection.

Detects when the user has been idle for longer than a configurable threshold
and emits an evidence event when transitioning from active to idle.
"""

from __future__ import annotations

import platform
import sys
from typing import Optional

from dojo.core.evidence import EvidenceLogger, EvidenceType


_LARGE_IDLE_SECONDS = 86400.0  # 24 h — used as a safe "assume idle" fallback


class IdleDetector:
    """Detect user idle time across Windows, macOS, Linux, and unknown platforms."""

    DEFAULT_IDLE_THRESHOLD_SECONDS: int = 300  # 5 minutes

    def __init__(self, threshold_seconds: int = DEFAULT_IDLE_THRESHOLD_SECONDS) -> None:
        self._threshold = threshold_seconds
        self._logger = EvidenceLogger.get_instance()
        self._was_idle: Optional[bool] = None  # tracks previous idle state

    # ------------------------------------------------------------------
    # Platform-specific idle detection
    # ------------------------------------------------------------------

    def seconds_since_last_input(self) -> float:
        """Return seconds since the last detected user input.

        Platform dispatch:
          - Windows: ``GetLastInputInfo`` via ``ctypes.windll.user32``
          - macOS: ``CoreGraphics`` event timestamp via ``ctypes``
          - Linux: ``/proc/uptime`` − ``/proc/stat`` heuristic, or fallback
          - Other: return large value (assume idle)
        """
        os_name = platform.system()
        try:
            if os_name == "Windows":
                return self._idle_windows()
            if os_name == "Darwin":
                return self._idle_macos()
            if os_name == "Linux":
                return self._idle_linux()
        except Exception:
            pass
        return _LARGE_IDLE_SECONDS

    def is_idle(self) -> bool:
        """Return ``True`` if idle time >= threshold.

        Also emits an ``IDLE_WINDOW_DETECTED`` evidence event when the system
        transitions from active → idle (edge-triggered, not level-triggered).
        """
        idle_seconds = self.seconds_since_last_input()
        currently_idle = idle_seconds >= self._threshold

        if currently_idle and self._was_idle is False:
            # Transition: active → idle
            self._logger.emit(
                {
                    "type": EvidenceType.IDLE_WINDOW_DETECTED.value,
                    "data": {
                        "idle_seconds": idle_seconds,
                        "threshold_seconds": self._threshold,
                    },
                },
                organ="idle",
            )

        self._was_idle = currently_idle
        return currently_idle

    # ------------------------------------------------------------------
    # Platform implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _idle_windows() -> float:
        """Use ``GetLastInputInfo`` to get milliseconds since last input."""
        import ctypes

        class _LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_ulong)]

        info = _LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(_LASTINPUTINFO)
        ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info))  # type: ignore[attr-defined]
        tick_ms = ctypes.windll.kernel32.GetTickCount()  # type: ignore[attr-defined]
        idle_ms = tick_ms - info.dwTime
        return max(0.0, idle_ms / 1000.0)

    @staticmethod
    def _idle_macos() -> float:
        """Use CoreGraphics ``CGEventSourceSecondsSinceLastEventType`` on macOS."""
        import ctypes
        import ctypes.util

        cg_path = ctypes.util.find_library("CoreGraphics")
        if not cg_path:
            return _LARGE_IDLE_SECONDS

        cg = ctypes.cdll.LoadLibrary(cg_path)
        # kCGEventSourceStateHIDSystemState = 1
        # kCGAnyInputEventType = 0xFFFFFFFF
        fn = cg.CGEventSourceSecondsSinceLastEventType
        fn.restype = ctypes.c_double
        fn.argtypes = [ctypes.c_int, ctypes.c_uint]
        secs: float = fn(1, 0xFFFFFFFF)
        return max(0.0, secs)

    @staticmethod
    def _idle_linux() -> float:
        """Estimate idle time on Linux via /proc/uptime.

        This is a coarse approximation — we return the uptime if no X11
        idle query succeeds, which causes the detector to assume the system
        is idle (safe default for headless servers).  For desktop sessions
        with python-xlib or xprintidle available the result would be more
        accurate; we keep this dependency-free.
        """
        # Try xprintidle subprocess first (common on desktop Linux)
        try:
            import subprocess
            result = subprocess.run(
                ["xprintidle"], capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                ms = int(result.stdout.strip())
                return ms / 1000.0
        except Exception:
            pass

        # Fallback: assume idle (safe for headless CI/servers)
        return _LARGE_IDLE_SECONDS


__all__ = ["IdleDetector"]

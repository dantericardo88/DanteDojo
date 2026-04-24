"""dojo/hal/apple.py — Apple Silicon (MLX) hardware detector.

Detects Apple ARM Macs by checking platform.system() and platform.machine().
Estimates unified memory via system_profiler or psutil.  Returns None on any
non-Mac-ARM platform without raising.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Optional

from dojo.core.types import HALProfile, HardwareTier


def _estimate_unified_memory_bytes() -> int:
    """Estimate total unified memory in bytes.

    Attempts system_profiler JSON output first; falls back to psutil; finally
    returns a conservative 16 GiB placeholder.
    """
    # Try system_profiler (available on macOS)
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            import json as _json
            data = _json.loads(result.stdout)
            hw_list = data.get("SPHardwareDataType", [])
            if hw_list:
                raw = hw_list[0].get("physical_memory", "")
                # raw is something like "24 GB" or "36 GB"
                parts = raw.strip().split()
                if len(parts) == 2 and parts[1].upper() == "GB":
                    return int(parts[0]) * 1024 ** 3
    except Exception:
        pass

    # Try psutil
    try:
        import psutil  # type: ignore[import-not-found]
        return psutil.virtual_memory().total
    except Exception:
        pass

    # Conservative 16 GiB fallback
    return 16 * 1024 ** 3


def detect_apple() -> Optional[HALProfile]:
    """Detect Apple Silicon hardware and return a HALProfile.

    Returns None if the current platform is not macOS arm64.

    The hardware tier is conservatively set to TIER_APPLE_M4.  Precise chip
    identification (M1/M2/M3/M4 variant) is deferred to a future detector.
    FP8 and FP4 are not natively supported on Apple MLX hardware.

    Returns:
        A signed HALProfile if running on Apple arm64, else None.
    """
    import platform

    if platform.system() != "Darwin":
        return None
    if platform.machine() != "arm64":
        return None

    vram_bytes = _estimate_unified_memory_bytes()

    # Build preliminary profile to compute signature
    _tmp = {
        "tier": HardwareTier.TIER_APPLE_M4.value,
        "gpu_name": "Apple Silicon",
        "vram_bytes": vram_bytes,
        "compute_capability": "0.0",
        "has_native_fp8": False,
        "has_native_fp4": False,
        "has_nvlink": False,
        "pcie_gen": 0,
        "memory_bandwidth_gb_s": 0.0,
        "driver_version": "mlx",
        "profile_sha256": "",
    }
    preliminary_sha = hashlib.sha256(
        json.dumps(_tmp, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    profile = HALProfile(
        tier=HardwareTier.TIER_APPLE_M4,
        gpu_name="Apple Silicon",
        vram_bytes=vram_bytes,
        compute_capability="0.0",
        has_native_fp8=False,
        has_native_fp4=False,
        has_nvlink=False,
        pcie_gen=0,
        interconnect_bandwidth_gb_s=0.0,
        memory_bandwidth_gb_s=0.0,
        driver_version="mlx",
        profile_sha256=preliminary_sha,
    )

    from dojo.hal.profile import sign_profile
    final_sha = sign_profile(profile)

    profile = HALProfile(
        tier=HardwareTier.TIER_APPLE_M4,
        gpu_name="Apple Silicon",
        vram_bytes=vram_bytes,
        compute_capability="0.0",
        has_native_fp8=False,
        has_native_fp4=False,
        has_nvlink=False,
        pcie_gen=0,
        interconnect_bandwidth_gb_s=0.0,
        memory_bandwidth_gb_s=0.0,
        driver_version="mlx",
        profile_sha256=final_sha,
    )

    try:
        from dojo.core.evidence import EvidenceLogger, EvidenceType
        logger = EvidenceLogger.get_instance()
        logger.emit(
            {
                "type": EvidenceType.HAL_DETECT_FOUND.value,
                "data": {
                    "backend": "apple",
                    "tier": HardwareTier.TIER_APPLE_M4.value,
                    "profile_sha256": final_sha,
                },
            },
            organ="hal",
        )
    except Exception:
        pass

    return profile


__all__ = ["detect_apple"]

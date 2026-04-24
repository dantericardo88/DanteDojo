"""dojo/hal/intel.py — Intel Arc GPU hardware detector.

Detects Intel Arc GPUs via platform/subprocess heuristics.
Returns None without raising if no Intel discrete GPU is detected.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from typing import Optional

from dojo.core.types import HALProfile, HardwareTier


def _detect_intel_via_sysfs() -> bool:
    """Return True if an Intel discrete GPU is found via /sys/bus/pci on Linux."""
    try:
        import pathlib
        drm = pathlib.Path("/sys/bus/pci/drivers/i915")
        if not drm.exists():
            return False
        # Look for discrete GPU markers (vendor=8086, class=0x030200)
        for dev in drm.iterdir():
            class_path = dev / "class"
            if class_path.exists():
                raw = class_path.read_text().strip().lower()
                # 0x030200 = Display controller, VGA, discrete
                if raw in ("0x030200", "0x038000", "0x030000"):
                    return True
        return False
    except Exception:
        return False


def _detect_intel_via_wmic() -> bool:
    """Return True if an Intel Arc GPU is found via wmic on Windows."""
    try:
        result = subprocess.run(
            ["wmic", "path", "win32_videocontroller", "get", "name", "/format:list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return False
        output_lower = result.stdout.lower()
        return "intel" in output_lower and ("arc" in output_lower or "a380" in output_lower
                                             or "a580" in output_lower or "b580" in output_lower
                                             or "a750" in output_lower or "a770" in output_lower)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


def _detect_intel_via_lspci() -> bool:
    """Return True if an Intel discrete GPU is found via lspci on Linux/macOS."""
    try:
        result = subprocess.run(
            ["lspci", "-d", "8086:", "-k"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return False
        output_lower = result.stdout.lower()
        return ("display" in output_lower or "vga" in output_lower) and "arc" in output_lower
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


def detect_intel() -> Optional[HALProfile]:
    """Detect Intel Arc GPU and return a HALProfile.

    Detection strategy (platform-aware):
    - Windows: wmic path win32_videocontroller
    - Linux:   lspci -d 8086: or /sys/bus/pci/drivers/i915

    Intel Arc B580/A-series does not support native FP8 or FP4.
    The tier is conservatively set to TIER_INTEL_B580.

    Returns:
        A signed HALProfile if an Intel Arc GPU is detected, else None.
    """
    found = False

    if sys.platform == "win32":
        found = _detect_intel_via_wmic()
    elif sys.platform.startswith("linux"):
        found = _detect_intel_via_lspci() or _detect_intel_via_sysfs()
    # macOS does not have discrete Intel Arc GPUs (post-2020 Macs)

    if not found:
        return None

    vram_bytes = 12 * 1024 ** 3  # 12 GiB — B580 reference spec

    _tmp = {
        "tier": HardwareTier.TIER_INTEL_B580.value,
        "gpu_name": "Intel Arc GPU",
        "vram_bytes": vram_bytes,
        "compute_capability": "0.0",
        "has_native_fp8": False,
        "has_native_fp4": False,
        "has_nvlink": False,
        "pcie_gen": 0,
        "memory_bandwidth_gb_s": 0.0,
        "driver_version": "intel",
        "profile_sha256": "",
    }
    preliminary_sha = hashlib.sha256(
        json.dumps(_tmp, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    profile = HALProfile(
        tier=HardwareTier.TIER_INTEL_B580,
        gpu_name="Intel Arc GPU",
        vram_bytes=vram_bytes,
        compute_capability="0.0",
        has_native_fp8=False,
        has_native_fp4=False,
        has_nvlink=False,
        pcie_gen=0,
        interconnect_bandwidth_gb_s=0.0,
        memory_bandwidth_gb_s=0.0,
        driver_version="intel",
        profile_sha256=preliminary_sha,
    )

    from dojo.hal.profile import sign_profile
    final_sha = sign_profile(profile)

    profile = HALProfile(
        tier=HardwareTier.TIER_INTEL_B580,
        gpu_name="Intel Arc GPU",
        vram_bytes=vram_bytes,
        compute_capability="0.0",
        has_native_fp8=False,
        has_native_fp4=False,
        has_nvlink=False,
        pcie_gen=0,
        interconnect_bandwidth_gb_s=0.0,
        memory_bandwidth_gb_s=0.0,
        driver_version="intel",
        profile_sha256=final_sha,
    )

    try:
        from dojo.core.evidence import EvidenceLogger, EvidenceType
        logger = EvidenceLogger.get_instance()
        logger.emit(
            {
                "type": EvidenceType.HAL_DETECT_FOUND.value,
                "data": {
                    "backend": "intel",
                    "tier": HardwareTier.TIER_INTEL_B580.value,
                    "profile_sha256": final_sha,
                },
            },
            organ="hal",
        )
    except Exception:
        pass

    return profile


__all__ = ["detect_intel"]

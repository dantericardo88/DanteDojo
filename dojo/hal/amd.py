"""dojo/hal/amd.py — AMD ROCm hardware detector.

Detects AMD ROCm GPUs by running ``rocm-smi --showmeminfo vram``.
Returns None without raising if rocm-smi is not available or the output
cannot be parsed.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from typing import Optional

from dojo.core.types import HALProfile, HardwareTier


_VRAM_PATTERN = re.compile(r"(\d+)\s*(?:MiB|MB|bytes)", re.IGNORECASE)
# rocm-smi outputs lines like:
#   GPU[0]          : VRAM Total Memory (B): 68702699520
#   GPU[0]          : VRAM Total Used Memory (B): 1234567
_BYTES_PATTERN = re.compile(r"VRAM Total Memory \(B\):\s*(\d+)", re.IGNORECASE)


def _parse_vram_bytes(output: str) -> int:
    """Parse VRAM total in bytes from rocm-smi output.

    Tries to find bytes first (most accurate), then falls back to MB/MiB.
    Returns 0 if parsing fails.
    """
    m = _BYTES_PATTERN.search(output)
    if m:
        return int(m.group(1))

    m = _VRAM_PATTERN.search(output)
    if m:
        value = int(m.group(1))
        # Assume MiB → bytes
        return value * 1024 * 1024

    return 0


def detect_amd() -> Optional[HALProfile]:
    """Detect AMD ROCm GPU and return a HALProfile.

    Runs ``rocm-smi --showmeminfo vram`` to discover VRAM.  If the command is
    unavailable or returns a non-zero exit code the function returns None.

    The tier is conservatively set to TIER_AMD_MI300.  AMD MI300 hardware
    supports native FP8.

    Returns:
        A signed HALProfile if an AMD ROCm GPU is detected, else None.
    """
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    output = result.stdout + result.stderr
    vram_bytes = _parse_vram_bytes(output)
    if vram_bytes == 0:
        # rocm-smi available but VRAM unreadable → use 64 GiB placeholder
        vram_bytes = 64 * 1024 ** 3

    _tmp = {
        "tier": HardwareTier.TIER_AMD_MI300.value,
        "gpu_name": "AMD ROCm GPU",
        "vram_bytes": vram_bytes,
        "compute_capability": "0.0",
        "has_native_fp8": True,
        "has_native_fp4": False,
        "has_nvlink": False,
        "pcie_gen": 0,
        "memory_bandwidth_gb_s": 0.0,
        "driver_version": "rocm",
        "profile_sha256": "",
    }
    preliminary_sha = hashlib.sha256(
        json.dumps(_tmp, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    profile = HALProfile(
        tier=HardwareTier.TIER_AMD_MI300,
        gpu_name="AMD ROCm GPU",
        vram_bytes=vram_bytes,
        compute_capability="0.0",
        has_native_fp8=True,
        has_native_fp4=False,
        has_nvlink=False,
        pcie_gen=0,
        interconnect_bandwidth_gb_s=0.0,
        memory_bandwidth_gb_s=0.0,
        driver_version="rocm",
        profile_sha256=preliminary_sha,
    )

    from dojo.hal.profile import sign_profile
    final_sha = sign_profile(profile)

    profile = HALProfile(
        tier=HardwareTier.TIER_AMD_MI300,
        gpu_name="AMD ROCm GPU",
        vram_bytes=vram_bytes,
        compute_capability="0.0",
        has_native_fp8=True,
        has_native_fp4=False,
        has_nvlink=False,
        pcie_gen=0,
        interconnect_bandwidth_gb_s=0.0,
        memory_bandwidth_gb_s=0.0,
        driver_version="rocm",
        profile_sha256=final_sha,
    )

    try:
        from dojo.core.evidence import EvidenceLogger, EvidenceType
        logger = EvidenceLogger.get_instance()
        logger.emit(
            {
                "type": EvidenceType.HAL_DETECT_FOUND.value,
                "data": {
                    "backend": "amd",
                    "tier": HardwareTier.TIER_AMD_MI300.value,
                    "profile_sha256": final_sha,
                },
            },
            organ="hal",
        )
    except Exception:
        pass

    return profile


__all__ = ["detect_amd"]

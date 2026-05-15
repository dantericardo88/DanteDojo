"""dojo/hal/nvidia.py — NVIDIA GPU hardware detector.

Detects CUDA-capable NVIDIA GPUs via pynvml (primary) or torch.cuda (fallback)
and returns a signed HALProfile.  Returns None (never raises) if neither
backend is available or no CUDA device is present.
"""

from __future__ import annotations

from typing import Optional

from dojo.core.types import HALProfile, HardwareTier


# ---------------------------------------------------------------------------
# Tier mapping helpers
# ---------------------------------------------------------------------------

def _tier_from_name(name: str, vram_bytes: int) -> HardwareTier:
    """Infer HardwareTier from GPU name string, falling back to VRAM size."""
    n = name.upper()

    if "H100" in n:
        return HardwareTier.TIER_H100
    if "3090" in n:
        return HardwareTier.TIER_3090
    if "4060" in n:
        return HardwareTier.TIER_4060_LAPTOP

    # VRAM-based heuristics for unknown names
    vram_gb = vram_bytes / (1024 ** 3)
    if vram_gb >= 70:
        return HardwareTier.TIER_H100
    if vram_gb >= 20:
        return HardwareTier.TIER_3090
    return HardwareTier.TIER_4060_LAPTOP


# ---------------------------------------------------------------------------
# pynvml backend
# ---------------------------------------------------------------------------

def _detect_via_pynvml() -> Optional[HALProfile]:
    """Primary path: use pynvml for full GPU info."""
    try:
        import pynvml  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        pynvml.nvmlInit()
    except Exception:
        return None

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # GPU name (may return bytes on older pynvml versions)
        raw_name = pynvml.nvmlDeviceGetName(handle)
        gpu_name: str = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)

        # Memory
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_bytes = int(mem_info.total)

        # Compute capability
        cc_major, cc_minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        compute_capability = f"{cc_major}.{cc_minor}"
        cc_float = float(compute_capability)

        # Driver version
        try:
            driver_version: str = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_version, bytes):
                driver_version = driver_version.decode("utf-8")
        except Exception:
            driver_version = "unknown"

    except Exception:
        return None

    # Precision support
    has_native_fp8 = cc_float >= 8.9
    has_native_fp4 = cc_float >= 10.0

    # Tier
    tier = _tier_from_name(gpu_name, vram_bytes)

    import hashlib, json, datetime

    _tmp_dict = {
        "tier": tier.value,
        "gpu_name": gpu_name,
        "vram_bytes": vram_bytes,
        "compute_capability": compute_capability,
        "has_native_fp8": has_native_fp8,
        "has_native_fp4": has_native_fp4,
        "has_nvlink": False,
        "pcie_gen": 0,
        "memory_bandwidth_gb_s": 0.0,
        "driver_version": driver_version,
        "profile_sha256": "",  # placeholder
        "signed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    preliminary_sha = hashlib.sha256(
        json.dumps(_tmp_dict, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    profile = HALProfile(
        tier=tier,
        gpu_name=gpu_name,
        vram_bytes=vram_bytes,
        compute_capability=compute_capability,
        has_native_fp8=has_native_fp8,
        has_native_fp4=has_native_fp4,
        has_nvlink=False,
        pcie_gen=0,
        interconnect_bandwidth_gb_s=0.0,
        memory_bandwidth_gb_s=0.0,
        driver_version=driver_version,
        profile_sha256=preliminary_sha,
    )

    # Re-sign with actual profile dict
    from dojo.hal.profile import sign_profile
    final_sha = sign_profile(profile)

    # Rebuild with accurate sha256
    profile = HALProfile(
        tier=tier,
        gpu_name=gpu_name,
        vram_bytes=vram_bytes,
        compute_capability=compute_capability,
        has_native_fp8=has_native_fp8,
        has_native_fp4=has_native_fp4,
        has_nvlink=False,
        pcie_gen=0,
        interconnect_bandwidth_gb_s=0.0,
        memory_bandwidth_gb_s=0.0,
        driver_version=driver_version,
        profile_sha256=final_sha,
    )

    # Emit evidence
    try:
        from dojo.core.evidence import EvidenceLogger, EvidenceType
        logger = EvidenceLogger.get_instance()
        logger.emit(
            {
                "type": EvidenceType.HAL_DETECT_FOUND.value,
                "data": {
                    "backend": "nvidia",
                    "gpu_name": gpu_name,
                    "tier": tier.value,
                    "compute_capability": compute_capability,
                    "sm_version": f"SM {cc_major}.{cc_minor}",
                    "profile_sha256": final_sha,
                },
            },
            organ="hal",
        )
    except Exception:
        pass

    return profile


# ---------------------------------------------------------------------------
# torch.cuda fallback backend
# ---------------------------------------------------------------------------

def _detect_via_torch() -> Optional[HALProfile]:
    """Fallback: use torch.cuda when pynvml is unavailable."""
    try:
        import torch  # type: ignore[import-not-found]
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        compute_capability = f"{major}.{minor}"
        cc_float = float(compute_capability)
        gpu_name = torch.cuda.get_device_name(0)
        # total_memory returns bytes
        vram_bytes = torch.cuda.get_device_properties(0).total_memory

        has_native_fp8 = cc_float >= 8.9
        has_native_fp4 = cc_float >= 10.0
        tier = _tier_from_name(gpu_name, vram_bytes)

        # Sign profile
        import hashlib, json
        _tmp = {
            "tier": tier.value,
            "gpu_name": gpu_name,
            "vram_bytes": vram_bytes,
            "compute_capability": compute_capability,
            "has_native_fp8": has_native_fp8,
        }
        sha = hashlib.sha256(json.dumps(_tmp, sort_keys=True).encode()).hexdigest()

        profile = HALProfile(
            tier=tier,
            gpu_name=gpu_name,
            vram_bytes=vram_bytes,
            compute_capability=compute_capability,
            has_native_fp8=has_native_fp8,
            has_native_fp4=has_native_fp4,
            has_nvlink=False,
            pcie_gen=0,
            interconnect_bandwidth_gb_s=0.0,
            memory_bandwidth_gb_s=0.0,
            driver_version="torch_fallback",
            profile_sha256=sha,
        )

        # Re-sign with proper profile fields
        from dojo.hal.profile import sign_profile
        final_sha = sign_profile(profile)

        profile = HALProfile(
            tier=tier,
            gpu_name=gpu_name,
            vram_bytes=vram_bytes,
            compute_capability=compute_capability,
            has_native_fp8=has_native_fp8,
            has_native_fp4=has_native_fp4,
            has_nvlink=False,
            pcie_gen=0,
            interconnect_bandwidth_gb_s=0.0,
            memory_bandwidth_gb_s=0.0,
            driver_version="torch_fallback",
            profile_sha256=final_sha,
        )

        from dojo.core.evidence import EvidenceLogger, EvidenceType
        logger = EvidenceLogger.get_instance()
        logger.emit(
            {
                "type": EvidenceType.HAL_DETECT_FOUND.value,
                "data": {
                    "backend": "nvidia_torch",
                    "gpu_name": gpu_name,
                    "compute_capability": compute_capability,
                    "sm_version": f"SM {major}.{minor}",
                    "has_native_fp8": has_native_fp8,
                    "profile_sha256": final_sha,
                },
            },
            organ="hal",
        )

        return profile
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_nvidia() -> Optional[HALProfile]:
    """Detect the primary NVIDIA GPU and return a signed HALProfile.

    Detection steps:
    1. Try pynvml — full GPU info including driver version.
    2. Fall back to torch.cuda — compute capability, name, VRAM.

    Returns:
        A fully populated, signed HALProfile, or None if no NVIDIA GPU is found.
    """
    result = _detect_via_pynvml()
    if result is not None:
        return result
    return _detect_via_torch()


__all__ = ["detect_nvidia"]

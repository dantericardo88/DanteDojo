"""dojo/hal/detector.py — Top-level hardware autodetector.

This is the single @one_door entry point for HAL detection.  Detection order:
  1. NVIDIA CUDA
  2. Apple MLX
  3. AMD ROCm
  4. Intel Arc
  5. CPU-only fallback

Evidence events are emitted at each stage.  The function never returns None —
it always constructs at minimum a CPU-only HALProfile.  HALDetectionError is
raised only if even the CPU fallback cannot be constructed (should never happen).
"""

from __future__ import annotations

import hashlib

from dojo.core.evidence import EvidenceLogger, EvidenceType
from dojo.core.exceptions import HALDetectionError
from dojo.core.one_door import one_door
from dojo.core.types import HALProfile, HardwareTier


@one_door("hal.detect")
def detect() -> HALProfile:
    """Autodetect hardware and return a signed HALProfile.

    Detection order:
      1. NVIDIA CUDA   (pynvml)
      2. Apple MLX     (platform check)
      3. AMD ROCm      (rocm-smi)
      4. Intel Arc     (wmic / lspci)
      5. CPU-only      (always succeeds)

    Evidence events emitted:
    - ``hal.detect.start``       — before any probe
    - ``hal.detect.found``       — when a backend succeeds
    - ``hal.detect.profile.signed`` — with profile_sha256

    Returns:
        A fully constructed, signed HALProfile.

    Raises:
        HALDetectionError: Only if the CPU fallback profile itself fails to
                           construct (effectively impossible in practice).
    """
    logger = EvidenceLogger.get_instance()
    logger.emit(
        {"type": EvidenceType.HAL_DETECT_START.value, "data": {}},
        organ="hal",
    )

    from dojo.hal.nvidia import detect_nvidia
    from dojo.hal.apple import detect_apple
    from dojo.hal.amd import detect_amd
    from dojo.hal.intel import detect_intel

    detectors = [
        (detect_nvidia, "nvidia"),
        (detect_apple, "apple"),
        (detect_amd, "amd"),
        (detect_intel, "intel"),
    ]

    for detector_fn, label in detectors:
        try:
            profile = detector_fn()
            if profile is not None:
                logger.emit(
                    {
                        "type": EvidenceType.HAL_DETECT_FOUND.value,
                        "data": {"backend": label, "tier": profile.tier.value},
                    },
                    organ="hal",
                )
                logger.emit(
                    {
                        "type": EvidenceType.HAL_PROFILE_SIGNED.value,
                        "data": {"profile_sha256": profile.profile_sha256},
                    },
                    organ="hal",
                )
                return profile
        except Exception as e:
            logger.warn(f"HAL detector '{label}' raised: {e}", organ="hal")

    # CPU-only fallback — always succeeds
    try:
        cpu_sha = hashlib.sha256(b"cpu_fallback").hexdigest()
        cpu_profile = HALProfile(
            tier=HardwareTier.TIER_CPU,
            gpu_name="CPU-only",
            vram_bytes=0,
            compute_capability="0.0",
            has_native_fp8=False,
            has_native_fp4=False,
            has_nvlink=False,
            pcie_gen=0,
            interconnect_bandwidth_gb_s=0.0,
            memory_bandwidth_gb_s=0.0,
            driver_version="none",
            profile_sha256=cpu_sha,
        )
    except Exception as exc:
        raise HALDetectionError(
            "Failed to construct CPU-only fallback HALProfile",
            reason=str(exc),
        ) from exc

    logger.emit(
        {
            "type": EvidenceType.HAL_DETECT_FOUND.value,
            "data": {"backend": "cpu", "tier": "tier_cpu"},
        },
        organ="hal",
    )
    logger.emit(
        {
            "type": EvidenceType.HAL_PROFILE_SIGNED.value,
            "data": {"profile_sha256": cpu_profile.profile_sha256},
        },
        organ="hal",
    )
    return cpu_profile


__all__ = ["detect"]

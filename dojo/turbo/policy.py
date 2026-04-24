"""dojo/turbo/policy.py — GPU Turbo policy selection.

Maps HALProfile hardware tiers to the appropriate list of PolicyType values.
Critically, Ampere hardware (SM 8.6, e.g. RTX 3090) does NOT get FP8 — it
receives BF16_DEFAULT + W8A16_MARLIN_SERVING instead.

Architecture → policy mapping (deterministic, no randomness):

    NVIDIA Blackwell  → FP4_QUANT, HET_COMM
    NVIDIA Hopper     → FP8_QUANT, HET_COMM
    NVIDIA Ada        → FP8_QUANT, HET_COMM
    NVIDIA Ampere     → BF16_DEFAULT, W8A16_MARLIN_SERVING   (NO FP8)
    AMD MI350         → FP4_QUANT, HET_COMM
    AMD MI300         → FP8_QUANT, HET_COMM
    Apple M4 Max      → INT4_QUANT
    Intel Arc B580    → INT4_QUANT
    CPU               → INT4_QUANT
"""

from __future__ import annotations

from enum import Enum
from typing import List

from dojo.core.types import HALProfile, HardwareTier


# ---------------------------------------------------------------------------
# Policy type enumeration
# ---------------------------------------------------------------------------

class PolicyType(str, Enum):
    """Turbo optimization policies available in DanteDojo."""

    FP4_QUANT = "fp4_quant"
    FP8_QUANT = "fp8_quant"
    INT4_QUANT = "int4_quant"
    BF16_DEFAULT = "bf16_default"
    W8A16_MARLIN_SERVING = "w8a16_marlin_serving"
    HET_COMM = "het_comm"


# ---------------------------------------------------------------------------
# Architecture policy map (string-keyed, architecture name → policy list)
# ---------------------------------------------------------------------------

ARCH_POLICY_MAP: dict[str, list[PolicyType]] = {
    "NVIDIA Blackwell": [PolicyType.FP4_QUANT, PolicyType.HET_COMM],
    "NVIDIA Hopper":    [PolicyType.FP8_QUANT, PolicyType.HET_COMM],
    "NVIDIA Ada":       [PolicyType.FP8_QUANT, PolicyType.HET_COMM],
    "NVIDIA Ampere":    [PolicyType.BF16_DEFAULT, PolicyType.W8A16_MARLIN_SERVING],  # NO FP8
    "AMD MI350":        [PolicyType.FP4_QUANT, PolicyType.HET_COMM],
    "AMD MI300":        [PolicyType.FP8_QUANT, PolicyType.HET_COMM],
    "Apple M4 Max":     [PolicyType.INT4_QUANT],
    "Intel Arc B580":   [PolicyType.INT4_QUANT],
    "CPU":              [PolicyType.INT4_QUANT],
}


# ---------------------------------------------------------------------------
# Tier → architecture name mapping
# ---------------------------------------------------------------------------

_TIER_TO_ARCH: dict[HardwareTier, str] = {
    HardwareTier.TIER_H100:          "NVIDIA Hopper",
    HardwareTier.TIER_3090:          "NVIDIA Ampere",
    HardwareTier.TIER_3090_DUAL:     "NVIDIA Ampere",
    HardwareTier.TIER_4060_LAPTOP:   "NVIDIA Ada",
    HardwareTier.TIER_AMD_MI300:     "AMD MI300",
    HardwareTier.TIER_APPLE_M4:      "Apple M4 Max",
    HardwareTier.TIER_INTEL_B580:    "Intel Arc B580",
    HardwareTier.TIER_CPU:           "CPU",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_arch_name(profile: HALProfile) -> str:
    """Return the architecture string for *profile*'s tier.

    Falls back to "CPU" for any unknown tier so the caller always gets a
    valid architecture label.
    """
    return _TIER_TO_ARCH.get(profile.tier, "CPU")


def get_policies_for_profile(profile: HALProfile) -> List[PolicyType]:
    """Return the ordered list of PolicyType values for *profile*.

    Mapping is deterministic: tier → architecture name → ARCH_POLICY_MAP.
    Ampere hardware (TIER_3090, TIER_3090_DUAL) receives BF16_DEFAULT +
    W8A16_MARLIN_SERVING — never FP8_QUANT.

    Args:
        profile: A HALProfile produced by dojo.hal.detector.detect().

    Returns:
        A non-empty list of PolicyType values.  Falls back to [INT4_QUANT]
        for any unrecognised tier.
    """
    arch = get_arch_name(profile)
    return list(ARCH_POLICY_MAP.get(arch, [PolicyType.INT4_QUANT]))


__all__ = ["PolicyType", "ARCH_POLICY_MAP", "get_arch_name", "get_policies_for_profile"]

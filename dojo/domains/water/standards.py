"""dojo/domains/water/standards.py — formal compendial rule table.

This module encodes representative design rules for compendial water distribution
systems as machine-checkable thresholds. It is the single source of truth used by:
  - generator.py  (to label synthetic designs as compliant)
  - verifier.py   (to gate any design — RL reward + eval judge)
  - solver.py     (as the constraint set to satisfy)

IMPORTANT — these thresholds are *representative engineering defaults* distilled
from public guidance (USP <1231>, EP water monographs, ASME BPE, ISPE Baseline
Guides). They are configurable and MUST be reviewed against the current editions
of the governing standards by a licensed engineer before any real-world use.
Nothing here is a substitute for the standards themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dojo.domains.water.schema import (
    PipeMaterial,
    SanitizationMethod,
    WaterGrade,
)


@dataclass(frozen=True)
class VelocityRule:
    """Turbulent-flow rule for biofilm suppression in distribution loops."""

    min_velocity_ms: float = 0.91   # ~3.0 ft/s, common GMP minimum
    min_reynolds: float = 10_000.0  # fully turbulent target


@dataclass(frozen=True)
class DeadLegRule:
    """Branch stagnation rule. Ratio = unused branch length / branch diameter."""

    max_l_over_d: float = 3.0       # ASME BPE-era "3D"; some sites enforce 2D


@dataclass(frozen=True)
class SurfaceRule:
    """Internal surface finish (product-contact) by material."""

    max_ra_um: dict[PipeMaterial, float] = field(
        default_factory=lambda: {
            PipeMaterial.SS_316L: 0.5,   # electropolished product-contact, um Ra
            PipeMaterial.PVDF:    0.8,
            PipeMaterial.PP:      1.0,
        }
    )


@dataclass(frozen=True)
class GradeQualityRule:
    """Compendial quality limits per water grade (action/spec limits)."""

    conductivity_us_cm: float
    toc_ppb: float
    microbial_cfu_ml: float
    endotoxin_eu_ml: float | None
    min_distribution_temp_c: float  # 0.0 means ambient permitted


# Per-grade quality envelope. Conductivity limits are temperature-dependent in the
# pharmacopeia (stage-1 tables); the single value here is a conservative screening
# limit suitable for a first-pass gate.
GRADE_QUALITY: dict[WaterGrade, GradeQualityRule] = {
    WaterGrade.PW: GradeQualityRule(
        conductivity_us_cm=1.3, toc_ppb=500, microbial_cfu_ml=100,
        endotoxin_eu_ml=None, min_distribution_temp_c=0.0,
    ),
    WaterGrade.HPW: GradeQualityRule(
        conductivity_us_cm=1.3, toc_ppb=500, microbial_cfu_ml=10,
        endotoxin_eu_ml=0.25, min_distribution_temp_c=0.0,
    ),
    WaterGrade.WFI: GradeQualityRule(
        conductivity_us_cm=1.3, toc_ppb=500, microbial_cfu_ml=10,
        endotoxin_eu_ml=0.25, min_distribution_temp_c=0.0,
    ),
    WaterGrade.CS: GradeQualityRule(
        conductivity_us_cm=1.3, toc_ppb=500, microbial_cfu_ml=10,
        endotoxin_eu_ml=0.25, min_distribution_temp_c=0.0,
    ),
}


# Which sanitization methods are accepted for each grade, and the minimum hot-loop
# temperature when a thermal method is selected.
ACCEPTED_SANITIZATION: dict[WaterGrade, set[SanitizationMethod]] = {
    WaterGrade.PW: {
        SanitizationMethod.HOT_WATER,
        SanitizationMethod.OZONE,
        SanitizationMethod.CHEMICAL,
    },
    WaterGrade.HPW: {SanitizationMethod.HOT_WATER, SanitizationMethod.CLEAN_STEAM},
    WaterGrade.WFI: {SanitizationMethod.HOT_WATER, SanitizationMethod.CLEAN_STEAM},
    WaterGrade.CS:  {SanitizationMethod.CLEAN_STEAM},
}

# Minimum loop temperature considered "self-sanitizing" when hot circulation is used.
HOT_LOOP_MIN_TEMP_C: float = 65.0
HOT_WATER_SANITIZATION_MIN_TEMP_C: float = 80.0

# Materials acceptable for product contact per grade. WFI is the strictest.
ACCEPTED_MATERIALS: dict[WaterGrade, set[PipeMaterial]] = {
    WaterGrade.PW:  {PipeMaterial.SS_316L, PipeMaterial.PVDF, PipeMaterial.PP},
    WaterGrade.HPW: {PipeMaterial.SS_316L},
    WaterGrade.WFI: {PipeMaterial.SS_316L},
    WaterGrade.CS:  {PipeMaterial.SS_316L},
}


@dataclass(frozen=True)
class StandardsTable:
    """Aggregate, injectable rule set. Pass a custom instance to retune any rule."""

    velocity: VelocityRule = field(default_factory=VelocityRule)
    dead_leg: DeadLegRule = field(default_factory=DeadLegRule)
    surface: SurfaceRule = field(default_factory=SurfaceRule)

    def quality_for(self, grade: WaterGrade) -> GradeQualityRule:
        return GRADE_QUALITY[grade]

    def sanitization_ok(self, grade: WaterGrade, method: SanitizationMethod) -> bool:
        return method in ACCEPTED_SANITIZATION[grade]

    def material_ok(self, grade: WaterGrade, material: PipeMaterial) -> bool:
        return material in ACCEPTED_MATERIALS[grade]


DEFAULT_STANDARDS = StandardsTable()

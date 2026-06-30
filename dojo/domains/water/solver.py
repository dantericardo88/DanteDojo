"""dojo/domains/water/solver.py — constraint solver: spec -> guaranteed-valid design.

This is the "guarantee" half of the hybrid system. Given an intent (WaterSystemSpec),
it searches the discrete catalog of standard pipe sizes and selects design parameters
that satisfy every rule in the StandardsTable. The output is verified before return,
so solve() either yields a compliant design or raises (it never emits a silent
non-compliant result).

The LLM half of the hybrid handles intent->spec and design->narrative; this module
owns the physics so the model can never hallucinate an invalid loop.
"""

from __future__ import annotations

from dojo.domains.water import hydraulics
from dojo.domains.water.schema import (
    LoopTopology,
    PipeMaterial,
    QualityTargets,
    SanitizationMethod,
    WaterGrade,
    WaterSystemDesign,
    WaterSystemSpec,
)
from dojo.domains.water.standards import (
    DEFAULT_STANDARDS,
    GRADE_QUALITY,
    StandardsTable,
)
from dojo.domains.water.verifier import verify


class InfeasibleSpecError(ValueError):
    """Raised when no design in the catalog can satisfy the spec + standards."""


# Standard sanitary tube nominal IDs (mm), ascending — a representative subset of
# ASME BPE tube sizes (1/2" .. 4" OD families, wall-adjusted ID).
STD_PIPE_ID_MM = [9.4, 15.7, 22.1, 34.8, 47.5, 60.2, 72.9, 97.4]


def _default_material(grade: WaterGrade, pref: PipeMaterial | None) -> PipeMaterial:
    if grade in {WaterGrade.WFI, WaterGrade.HPW, WaterGrade.CS}:
        return PipeMaterial.SS_316L  # only 316L accepted for these grades
    return pref or PipeMaterial.SS_316L


def _default_sanitization(grade: WaterGrade, requested: SanitizationMethod) -> SanitizationMethod:
    from dojo.domains.water.standards import ACCEPTED_SANITIZATION

    if requested in ACCEPTED_SANITIZATION[grade]:
        return requested
    # fall back to the strongest accepted method for the grade
    for m in (SanitizationMethod.CLEAN_STEAM, SanitizationMethod.HOT_WATER,
              SanitizationMethod.OZONE, SanitizationMethod.CHEMICAL):
        if m in ACCEPTED_SANITIZATION[grade]:
            return m
    raise InfeasibleSpecError(f"no accepted sanitization for {grade.value}")


def solve(
    spec: WaterSystemSpec,
    standards: StandardsTable = DEFAULT_STANDARDS,
) -> WaterSystemDesign:
    """Select a compliant design for the spec, or raise InfeasibleSpecError.

    Strategy: for each standard pipe size (largest first, to minimise pump duty),
    compute the recirculation flow needed to hold the minimum velocity, ensure it
    also covers peak demand, and verify the full design. Return the first that
    passes every critical check.
    """
    material = _default_material(spec.grade, spec.material_preference)
    sanitization = _default_sanitization(spec.grade, spec.sanitization)
    max_ra = standards.surface.max_ra_um.get(material, 1.0)
    q = GRADE_QUALITY[spec.grade]

    targets = QualityTargets(
        conductivity_us_cm=q.conductivity_us_cm,
        toc_ppb=q.toc_ppb,
        endotoxin_eu_ml=q.endotoxin_eu_ml,
        microbial_cfu_ml=q.microbial_cfu_ml,
    )

    topology = (
        LoopTopology.RECIRCULATING
        if spec.grade in {WaterGrade.PW, WaterGrade.HPW, WaterGrade.WFI}
        else LoopTopology.RECIRCULATING
    )

    # Prefer larger pipe (lower pressure drop) but it must still hit min velocity at a
    # flow we're willing to pump. Iterate largest->smallest and keep the first feasible.
    candidates: list[WaterSystemDesign] = []
    for pipe_id in sorted(STD_PIPE_ID_MM, reverse=True):
        # Flow to achieve exactly the minimum velocity in this pipe:
        flow_for_v = hydraulics.required_flow_for_velocity(
            standards.velocity.min_velocity_ms, pipe_id
        )
        # Must also cover peak demand; pump at the larger of the two, +10% margin.
        pump_flow = max(flow_for_v, spec.peak_demand_lph) * 1.10
        v = hydraulics.velocity_ms(pump_flow, pipe_id)

        design = WaterSystemDesign(
            spec=spec,
            topology=topology,
            material=material,
            pipe_id_mm=pipe_id,
            design_velocity_ms=round(v, 4),
            pump_flow_lph=round(pump_flow, 1),
            max_dead_leg_diameters=min(2.0, standards.dead_leg.max_l_over_d),
            surface_finish_ra_um=max_ra,
            sanitization=sanitization,
            quality_targets=targets,
            rationale="",
        )
        report = verify(design, standards)
        if report.compliant:
            candidates.append(design)

    if not candidates:
        raise InfeasibleSpecError(
            f"No catalog pipe satisfies spec for {spec.grade.value} "
            f"(peak={spec.peak_demand_lph} L/h)"
        )

    # Pick the candidate with the lowest pump duty that still passes — least energy.
    best = min(candidates, key=lambda d: d.pump_flow_lph)
    best.rationale = _explain(best)
    return best


def _explain(d: WaterSystemDesign) -> str:
    """Generate the natural-language design justification (the SFT label target)."""
    s = d.spec
    re = hydraulics.reynolds(d.pump_flow_lph, d.pipe_id_mm, s.distribution_temp_c)
    return (
        f"{s.grade.value.replace('_', ' ').title()} loop sized for {s.peak_demand_lph:.0f} L/h "
        f"peak across {s.use_points} use point(s). Selected {d.pipe_id_mm:.1f} mm ID "
        f"{d.material.value} tubing with a {d.pump_flow_lph:.0f} L/h recirculation pump, "
        f"giving {d.design_velocity_ms:.2f} m/s "
        f"({hydraulics.ms_to_fts(d.design_velocity_ms):.2f} ft/s) and Re={re:,.0f} - "
        f"fully turbulent flow that suppresses biofilm formation. The {d.topology.value} "
        f"topology and <= {d.max_dead_leg_diameters:.0f}D dead-leg limit prevent "
        f"stagnation, internal finish is held at Ra <= {d.surface_finish_ra_um:.2f} um, "
        f"and the loop is sanitized by {d.sanitization.value.replace('_', ' ')}. "
        f"Quality is gated to <= {d.quality_targets.conductivity_us_cm} uS/cm conductivity, "
        f"<= {d.quality_targets.toc_ppb} ppb TOC, and <= {d.quality_targets.microbial_cfu_ml} "
        f"CFU/mL. Requires review and sign-off by a licensed engineer."
    )

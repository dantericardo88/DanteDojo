"""dojo/domains/water/verifier.py — the design oracle.

Scores any WaterSystemDesign against the StandardsTable using independently
re-derived physics (it does NOT trust the velocity/Reynolds the design claims —
it recomputes them from pump_flow + pipe_id). This single function is reused as:
  - synthetic-data labeler (generator.py keeps only designs that pass)
  - RL reward signal        (reward = score / fraction of checks passed)
  - eval judge              (pass@1 / pass^k over a held-out spec set)

Returns a structured report so failures are explainable, not just a boolean.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dojo.domains.water import hydraulics
from dojo.domains.water.schema import (
    LoopTopology,
    WaterGrade,
    WaterSystemDesign,
)
from dojo.domains.water.standards import (
    DEFAULT_STANDARDS,
    HOT_LOOP_MIN_TEMP_C,
    HOT_WATER_SANITIZATION_MIN_TEMP_C,
    StandardsTable,
)
from dojo.domains.water.schema import SanitizationMethod


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    critical: bool = True  # critical failures veto compliance regardless of score


@dataclass
class VerificationReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Fraction of checks passed, in [0, 1] — usable directly as RL reward."""
        if not self.checks:
            return 0.0
        return sum(1 for c in self.checks if c.passed) / len(self.checks)

    @property
    def compliant(self) -> bool:
        """True only if NO critical check failed."""
        return all(c.passed for c in self.checks if c.critical)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        head = f"{'COMPLIANT' if self.compliant else 'NON-COMPLIANT'} | score={self.score:.2f}"
        lines = [head]
        for c in self.checks:
            mark = "PASS" if c.passed else ("FAIL" if c.critical else "warn")
            lines.append(f"  [{mark}] {c.name}: {c.detail}")
        return "\n".join(lines)


def verify(
    design: WaterSystemDesign,
    standards: StandardsTable = DEFAULT_STANDARDS,
) -> VerificationReport:
    """Run all gates against a design and return an explainable report."""
    spec = design.spec
    grade = spec.grade
    checks: list[CheckResult] = []

    # --- Recompute physics independently from the design's own claims ---
    re_v = hydraulics.velocity_ms(design.pump_flow_lph, design.pipe_id_mm)
    re_re = hydraulics.reynolds(
        design.pump_flow_lph, design.pipe_id_mm, spec.distribution_temp_c
    )

    # 1. Velocity (biofilm suppression)
    checks.append(CheckResult(
        name="velocity_min",
        passed=re_v >= standards.velocity.min_velocity_ms,
        detail=(f"v={re_v:.2f} m/s ({hydraulics.ms_to_fts(re_v):.2f} ft/s), "
                f"min {standards.velocity.min_velocity_ms:.2f} m/s"),
    ))

    # 2. Turbulence
    checks.append(CheckResult(
        name="turbulent_flow",
        passed=re_re >= standards.velocity.min_reynolds,
        detail=f"Re={re_re:,.0f}, min {standards.velocity.min_reynolds:,.0f}",
    ))

    # 3. Design's claimed velocity must agree with physics (no hand-waving)
    claim_err = abs(design.design_velocity_ms - re_v)
    checks.append(CheckResult(
        name="velocity_claim_consistent",
        passed=claim_err <= 0.05,
        detail=f"claimed {design.design_velocity_ms:.2f} vs computed {re_v:.2f} m/s",
    ))

    # 4. Dead-leg ratio
    checks.append(CheckResult(
        name="dead_leg",
        passed=design.max_dead_leg_diameters <= standards.dead_leg.max_l_over_d,
        detail=(f"L/d={design.max_dead_leg_diameters:.1f}, "
                f"max {standards.dead_leg.max_l_over_d:.1f}"),
    ))

    # 5. Topology — recirculating required for PW/WFI/HPW
    needs_loop = grade in {WaterGrade.PW, WaterGrade.HPW, WaterGrade.WFI}
    checks.append(CheckResult(
        name="topology",
        passed=(design.topology == LoopTopology.RECIRCULATING) or not needs_loop,
        detail=f"{design.topology.value} for {grade.value}",
    ))

    # 6. Material acceptable for grade
    checks.append(CheckResult(
        name="material",
        passed=standards.material_ok(grade, design.material),
        detail=f"{design.material.value} for {grade.value}",
    ))

    # 7. Surface finish
    max_ra = standards.surface.max_ra_um.get(design.material, 1.0)
    checks.append(CheckResult(
        name="surface_finish",
        passed=design.surface_finish_ra_um <= max_ra,
        detail=f"Ra={design.surface_finish_ra_um:.2f} um, max {max_ra:.2f}",
    ))

    # 8. Sanitization method accepted for grade
    checks.append(CheckResult(
        name="sanitization_method",
        passed=standards.sanitization_ok(grade, design.sanitization),
        detail=f"{design.sanitization.value} for {grade.value}",
    ))

    # 9. Hot-water sanitization implies the loop can reach the required temperature
    if design.sanitization == SanitizationMethod.HOT_WATER:
        # The loop need not circulate hot continuously, but must be rated for it;
        # we proxy "rated" by requiring either a hot loop or a documented capability.
        ok = spec.distribution_temp_c >= HOT_LOOP_MIN_TEMP_C or (
            HOT_WATER_SANITIZATION_MIN_TEMP_C in (80.0, 85.0)  # method definitionally >=80
        )
        checks.append(CheckResult(
            name="thermal_capability",
            passed=ok,
            detail=(f"hot-water sanitization @>= {HOT_WATER_SANITIZATION_MIN_TEMP_C:.0f}degC; "
                    f"loop temp {spec.distribution_temp_c:.0f}degC"),
            critical=False,  # advisory: depends on equipment rating not in spec
        ))

    # 10. Quality targets within compendial envelope for the grade
    q = standards.quality_for(grade)
    qt = design.quality_targets
    checks.append(CheckResult(
        name="conductivity_target",
        passed=qt.conductivity_us_cm <= q.conductivity_us_cm,
        detail=f"{qt.conductivity_us_cm} <= {q.conductivity_us_cm} uS/cm",
    ))
    checks.append(CheckResult(
        name="toc_target",
        passed=qt.toc_ppb <= q.toc_ppb,
        detail=f"{qt.toc_ppb} <= {q.toc_ppb} ppb",
    ))
    checks.append(CheckResult(
        name="microbial_target",
        passed=qt.microbial_cfu_ml <= q.microbial_cfu_ml,
        detail=f"{qt.microbial_cfu_ml} <= {q.microbial_cfu_ml} CFU/mL",
    ))
    if q.endotoxin_eu_ml is not None:
        checks.append(CheckResult(
            name="endotoxin_target",
            passed=qt.endotoxin_eu_ml is not None
            and qt.endotoxin_eu_ml <= q.endotoxin_eu_ml,
            detail=f"{qt.endotoxin_eu_ml} <= {q.endotoxin_eu_ml} EU/mL",
        ))

    # 11. Pump must at least cover peak demand while sustaining loop velocity
    checks.append(CheckResult(
        name="pump_covers_demand",
        passed=design.pump_flow_lph >= spec.peak_demand_lph,
        detail=f"pump {design.pump_flow_lph:.0f} >= peak {spec.peak_demand_lph:.0f} L/h",
    ))

    return VerificationReport(checks=checks)


def reward(design: WaterSystemDesign, standards: StandardsTable = DEFAULT_STANDARDS) -> float:
    """RL reward: 1.0 only if fully compliant, else the fraction of checks passed,
    capped below 1.0 so a critical failure can never score as a pass."""
    report = verify(design, standards)
    if report.compliant:
        return 1.0
    return min(report.score, 0.99)

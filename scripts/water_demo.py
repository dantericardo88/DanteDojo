"""scripts/water_demo.py — end-to-end demo of the compendial water design domain.

Run:  python scripts/water_demo.py

Shows: spec -> solver design -> verifier report -> rationale, then builds a small
synthetic SFT corpus and reports compliance stats. No GPU/model required.
"""

from __future__ import annotations

from dojo.domains.water.generator import build_dataset
from dojo.domains.water.schema import (
    SanitizationMethod,
    WaterGrade,
    WaterSystemSpec,
)
from dojo.domains.water.solver import solve
from dojo.domains.water.verifier import verify


def demo_single() -> None:
    spec = WaterSystemSpec(
        grade=WaterGrade.WFI,
        peak_demand_lph=2000,
        average_demand_lph=800,
        use_points=12,
        loop_length_m=85,
        distribution_temp_c=80,
        sanitization=SanitizationMethod.HOT_WATER,
        notes="Biotech fill-finish suite",
    )
    print("=" * 72)
    print("INPUT SPEC")
    print("=" * 72)
    print(spec.model_dump_json(indent=2))

    design = solve(spec)
    report = verify(design)

    print("\n" + "=" * 72)
    print("SOLVED DESIGN")
    print("=" * 72)
    print(f"  topology         : {design.topology.value}")
    print(f"  material         : {design.material.value}")
    print(f"  pipe ID          : {design.pipe_id_mm} mm")
    print(f"  velocity         : {design.design_velocity_ms} m/s")
    print(f"  pump duty        : {design.pump_flow_lph} L/h")
    print(f"  dead-leg limit   : {design.max_dead_leg_diameters}D")
    print(f"  surface finish   : Ra {design.surface_finish_ra_um} um")
    print(f"  sanitization     : {design.sanitization.value}")

    print("\n" + "=" * 72)
    print("VERIFIER REPORT")
    print("=" * 72)
    print(report.summary())

    print("\n" + "=" * 72)
    print("RATIONALE (SFT label target)")
    print("=" * 72)
    print(design.rationale)


def demo_corpus() -> None:
    print("\n" + "=" * 72)
    print("SYNTHETIC SFT CORPUS")
    print("=" * 72)
    records = build_dataset(n=200, seed=42)
    by_grade: dict[str, int] = {}
    for r in records:
        by_grade[r["meta"]["grade"]] = by_grade.get(r["meta"]["grade"], 0) + 1
    print(f"  generated {len(records)} compliant (spec -> design) pairs")
    for g, c in sorted(by_grade.items()):
        print(f"    {g:24s} {c}")
    print("  100% verifier-compliant by construction "
          "(non-compliant draws are dropped, never trained on)")


if __name__ == "__main__":
    demo_single()
    demo_corpus()

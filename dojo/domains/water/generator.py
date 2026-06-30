"""dojo/domains/water/generator.py — synthetic (spec -> design) pair factory.

Since real compliant designs are scarce/proprietary, we manufacture training data:
sample plausible specs across the grade/size/sanitization space, solve each with
the constraint solver, verify, and emit instruction-tuning records in the chat
format dojo.train.finetune expects.

Determinism: a seeded random.Random is threaded through so dataset generation is
reproducible (no reliance on global RNG state). Pass the same seed to regenerate
an identical corpus.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from dojo.domains.water.schema import (
    PipeMaterial,
    SanitizationMethod,
    WaterGrade,
    WaterSystemSpec,
)
from dojo.domains.water.solver import InfeasibleSpecError, solve
from dojo.domains.water.standards import ACCEPTED_SANITIZATION
from dojo.domains.water.verifier import verify

_SYSTEM_PROMPT = (
    "You are a compendial water-system design assistant for life-science facilities. "
    "Given a requirement, produce a first-pass distribution-loop design and justify it "
    "against USP/EP/ASME BPE design rules. Always note that a licensed engineer must "
    "review and sign off the design."
)


def sample_spec(rng: random.Random) -> WaterSystemSpec:
    """Draw one plausible, internally-consistent spec."""
    grade = rng.choice(list(WaterGrade))
    use_points = rng.randint(1, 24)
    avg = rng.uniform(200, 4000)
    peak = avg * rng.uniform(1.3, 3.0)
    loop_len = rng.uniform(15, 180)
    if grade in {WaterGrade.WFI, WaterGrade.HPW, WaterGrade.CS}:
        temp = rng.choice([80.0, 85.0, 25.0])  # often hot WFI loops
    else:
        temp = rng.choice([25.0, 25.0, 80.0])
    sani = rng.choice(sorted(ACCEPTED_SANITIZATION[grade], key=lambda m: m.value))
    material = PipeMaterial.SS_316L if grade != WaterGrade.PW else rng.choice(
        [PipeMaterial.SS_316L, PipeMaterial.SS_316L, PipeMaterial.PVDF]
    )
    return WaterSystemSpec(
        grade=grade,
        peak_demand_lph=round(peak, 1),
        average_demand_lph=round(avg, 1),
        use_points=use_points,
        loop_length_m=round(loop_len, 1),
        distribution_temp_c=temp,
        sanitization=sani,
        material_preference=material,
    )


def spec_to_prompt(spec: WaterSystemSpec) -> str:
    return (
        f"Design a {spec.grade.value.replace('_', ' ')} distribution loop. "
        f"Peak demand {spec.peak_demand_lph:.0f} L/h, average {spec.average_demand_lph:.0f} L/h, "
        f"{spec.use_points} use point(s), loop length {spec.loop_length_m:.0f} m, "
        f"circulation temperature {spec.distribution_temp_c:.0f} degC, "
        f"sanitization preference {spec.sanitization.value.replace('_', ' ')}."
    )


def make_record(spec: WaterSystemSpec) -> dict | None:
    """Solve + verify one spec; return a chat-format SFT record, or None if infeasible
    or (defensively) non-compliant. Returning None keeps the corpus 100% clean."""
    try:
        design = solve(spec)
    except InfeasibleSpecError:
        return None
    report = verify(design)
    if not report.compliant:
        return None  # never train on a non-compliant label
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": spec_to_prompt(spec)},
            {"role": "assistant", "content": _format_design(design)},
        ],
        "meta": {
            "grade": spec.grade.value,
            "verifier_score": round(report.score, 4),
            "compliant": True,
        },
    }


def _format_design(design) -> str:
    """Render the solved design as the assistant's target completion."""
    d = design
    body = {
        "topology": d.topology.value,
        "material": d.material.value,
        "pipe_id_mm": d.pipe_id_mm,
        "design_velocity_ms": d.design_velocity_ms,
        "pump_flow_lph": d.pump_flow_lph,
        "max_dead_leg_diameters": d.max_dead_leg_diameters,
        "surface_finish_ra_um": d.surface_finish_ra_um,
        "sanitization": d.sanitization.value,
        "quality_targets": d.quality_targets.model_dump(),
    }
    return (
        "DESIGN:\n"
        + json.dumps(body, indent=2)
        + "\n\nRATIONALE:\n"
        + d.rationale
    )


def build_dataset(
    n: int,
    seed: int,
    out_path: str | Path | None = None,
) -> list[dict]:
    """Generate up to n clean SFT records (infeasible draws are skipped).

    If out_path is given, write JSONL. Returns the records either way.
    """
    rng = random.Random(seed)
    records: list[dict] = []
    attempts = 0
    # Over-sample to absorb infeasible draws, with a hard attempt cap.
    while len(records) < n and attempts < n * 5:
        attempts += 1
        rec = make_record(sample_spec(rng))
        if rec is not None:
            records.append(rec)

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records

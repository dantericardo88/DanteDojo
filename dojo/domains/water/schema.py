"""dojo/domains/water/schema.py — typed structures for compendial water systems.

All physical quantities are stored in SI base-ish units to avoid unit ambiguity:
  - flow:        litres per hour (L/h)
  - diameter:    millimetres (mm)        [nominal pipe ID]
  - length:      metres (m)
  - velocity:    metres per second (m/s)
  - temperature: degrees Celsius (degC)
  - pressure:    bar (gauge)

Display/imperial conversions (e.g. ft/s) live in hydraulics.py, not here.

NOTE (GxP): these structures describe a *first-pass* design intent. Any real
deliverable requires review and sign-off by a licensed engineer. The schema is
deliberately explicit about quality attributes so the verifier can gate them.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class WaterGrade(str, Enum):
    """Compendial water grades relevant to life-science distribution loops."""

    PW   = "purified_water"        # USP/EP Purified Water
    WFI  = "water_for_injection"   # USP/EP Water for Injection
    HPW  = "highly_purified_water" # EP Highly Purified Water (legacy EP grade)
    CS   = "clean_steam"           # Pure/clean steam (pharma steam)


class SanitizationMethod(str, Enum):
    """How the loop is periodically sanitized — drives material + thermal design."""

    HOT_WATER  = "hot_water_80c"   # circulate >=80 degC
    CLEAN_STEAM = "clean_steam"    # SIP, typ. for WFI
    OZONE       = "ozone"          # dissolved O3 (PW, ambient loops), UV destruct pre-use
    CHEMICAL    = "chemical"       # periodic chemical sanitization


class PipeMaterial(str, Enum):
    """Distribution piping material of construction."""

    SS_316L = "316L_stainless"     # electropolished, the GMP default
    PVDF    = "pvdf"               # accepted for some ambient PW
    PP      = "polypropylene"      # limited use


class LoopTopology(str, Enum):
    """Distribution circuit topology."""

    RECIRCULATING = "recirculating"  # continuously circulated loop (GMP default)
    ONE_WAY       = "one_way"        # dead-ended branch (discouraged for PW/WFI)


class WaterSystemSpec(BaseModel):
    """Customer/engineer intent — the input the model designs against."""

    grade: WaterGrade
    peak_demand_lph: float = Field(gt=0, description="Peak simultaneous draw, L/h")
    average_demand_lph: float = Field(gt=0, description="Average draw, L/h")
    use_points: int = Field(ge=1, description="Number of points of use on the loop")
    loop_length_m: float = Field(gt=0, description="Total circulating loop length, m")
    distribution_temp_c: float = Field(description="Loop circulation temperature, degC")
    sanitization: SanitizationMethod
    material_preference: PipeMaterial | None = None
    notes: str = ""

    @property
    def is_hot_loop(self) -> bool:
        return self.distribution_temp_c >= 65.0


class QualityTargets(BaseModel):
    """Compendial quality attributes the design must be capable of holding."""

    conductivity_us_cm: float = Field(description="Max conductivity, uS/cm (USP <645>)")
    toc_ppb: float = Field(description="Max total organic carbon, ppb (USP <643>)")
    endotoxin_eu_ml: float | None = Field(
        default=None, description="Max endotoxin, EU/mL (WFI only)"
    )
    microbial_cfu_ml: float = Field(description="Action limit, CFU/mL")


class WaterSystemDesign(BaseModel):
    """A proposed design — the model's structured output, scored by the verifier."""

    spec: WaterSystemSpec
    topology: LoopTopology
    material: PipeMaterial
    pipe_id_mm: float = Field(gt=0, description="Selected nominal pipe inner diameter, mm")
    design_velocity_ms: float = Field(gt=0, description="Computed loop velocity, m/s")
    pump_flow_lph: float = Field(gt=0, description="Recirculation pump duty, L/h")
    max_dead_leg_diameters: float = Field(
        ge=0, description="Worst-case branch dead-leg ratio L/d"
    )
    surface_finish_ra_um: float = Field(
        gt=0, description="Internal surface roughness Ra, micrometres"
    )
    sanitization: SanitizationMethod
    quality_targets: QualityTargets
    rationale: str = Field(default="", description="Natural-language design justification")

    model_config = {"use_enum_values": False}

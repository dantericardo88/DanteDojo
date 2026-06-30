"""dojo/domains/water — compendial water distribution system design domain.

A hybrid LLM + solver vertical:
  schema      typed spec/design structures
  standards   formal, configurable compendial rule table (USP/EP/ASME BPE defaults)
  hydraulics  incompressible-flow physics (velocity, Reynolds, Darcy-Weisbach)
  verifier    the design oracle — labeler + RL reward + eval judge in one
  solver      constraint solver: spec -> guaranteed-compliant design
  generator   synthetic (spec -> design + rationale) SFT pair factory

The model designs; the solver/verifier guarantee. Output is positioned as a
first-pass design-assist that requires licensed-engineer sign-off.
"""

from dojo.domains.water.schema import (
    LoopTopology,
    PipeMaterial,
    QualityTargets,
    SanitizationMethod,
    WaterGrade,
    WaterSystemDesign,
    WaterSystemSpec,
)
from dojo.domains.water.solver import InfeasibleSpecError, solve
from dojo.domains.water.verifier import VerificationReport, reward, verify

__all__ = [
    "WaterGrade",
    "SanitizationMethod",
    "PipeMaterial",
    "LoopTopology",
    "QualityTargets",
    "WaterSystemSpec",
    "WaterSystemDesign",
    "solve",
    "InfeasibleSpecError",
    "verify",
    "reward",
    "VerificationReport",
]

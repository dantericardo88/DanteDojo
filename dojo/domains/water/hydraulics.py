"""dojo/domains/water/hydraulics.py — incompressible-flow physics for water loops.

Pure functions, no state. Used by the solver (to size pipe) and the verifier
(to re-derive velocity/Reynolds independently of whatever the design claims).

Water properties default to ~20 degC but vary with temperature, which matters for
hot WFI loops (lower viscosity raises Reynolds for the same velocity). A small
temperature-interpolated property model is included.
"""

from __future__ import annotations

import math

# Water density (kg/m^3) and dynamic viscosity (Pa.s) at reference temperatures.
# Linear interpolation between these anchors is accurate enough for loop sizing.
_RHO_TABLE = {0: 999.8, 20: 998.2, 40: 992.2, 60: 983.2, 80: 971.8, 100: 958.4}
_MU_TABLE = {0: 1.792e-3, 20: 1.002e-3, 40: 0.653e-3, 60: 0.467e-3,
             80: 0.355e-3, 100: 0.282e-3}

FT_PER_M = 3.280839895


def _interp(table: dict[int, float], temp_c: float) -> float:
    temps = sorted(table)
    if temp_c <= temps[0]:
        return table[temps[0]]
    if temp_c >= temps[-1]:
        return table[temps[-1]]
    lo = max(t for t in temps if t <= temp_c)
    hi = min(t for t in temps if t >= temp_c)
    if lo == hi:
        return table[lo]
    frac = (temp_c - lo) / (hi - lo)
    return table[lo] + frac * (table[hi] - table[lo])


def water_density(temp_c: float) -> float:
    """Density in kg/m^3 at the given temperature."""
    return _interp(_RHO_TABLE, temp_c)


def water_viscosity(temp_c: float) -> float:
    """Dynamic viscosity in Pa.s at the given temperature."""
    return _interp(_MU_TABLE, temp_c)


def velocity_ms(flow_lph: float, pipe_id_mm: float) -> float:
    """Mean flow velocity (m/s) for a volumetric flow through a circular pipe.

    v = Q / A, with Q in m^3/s and A = pi/4 * d^2.
    """
    q_m3s = flow_lph / 1000.0 / 3600.0
    area_m2 = math.pi * (pipe_id_mm / 1000.0) ** 2 / 4.0
    return q_m3s / area_m2


def reynolds(flow_lph: float, pipe_id_mm: float, temp_c: float) -> float:
    """Reynolds number Re = rho * v * d / mu."""
    v = velocity_ms(flow_lph, pipe_id_mm)
    d_m = pipe_id_mm / 1000.0
    return water_density(temp_c) * v * d_m / water_viscosity(temp_c)


def required_flow_for_velocity(target_v_ms: float, pipe_id_mm: float) -> float:
    """Inverse of velocity_ms: flow (L/h) needed to hit a target velocity."""
    area_m2 = math.pi * (pipe_id_mm / 1000.0) ** 2 / 4.0
    q_m3s = target_v_ms * area_m2
    return q_m3s * 1000.0 * 3600.0


def friction_factor(re: float, pipe_id_mm: float, roughness_um: float) -> float:
    """Darcy friction factor.

    Laminar (Re < 2300): f = 64/Re.
    Turbulent: Swamee-Jain explicit approximation to Colebrook.
    """
    if re <= 0:
        return 0.0
    if re < 2300:
        return 64.0 / re
    rel_rough = (roughness_um * 1e-6) / (pipe_id_mm / 1000.0)
    denom = math.log10(rel_rough / 3.7 + 5.74 / re**0.9)
    return 0.25 / denom**2


def pressure_drop_bar(
    flow_lph: float,
    pipe_id_mm: float,
    length_m: float,
    temp_c: float,
    roughness_um: float = 0.5,
) -> float:
    """Darcy-Weisbach friction pressure drop over a straight run, in bar (gauge delta).

    dP = f * (L/d) * (rho * v^2 / 2)
    Minor losses (fittings, valves) are not included; multiply by a fitting factor
    upstream if needed.
    """
    v = velocity_ms(flow_lph, pipe_id_mm)
    re = reynolds(flow_lph, pipe_id_mm, temp_c)
    f = friction_factor(re, pipe_id_mm, roughness_um)
    rho = water_density(temp_c)
    dp_pa = f * (length_m / (pipe_id_mm / 1000.0)) * (rho * v**2 / 2.0)
    return dp_pa / 1e5


def ms_to_fts(v_ms: float) -> float:
    """Convert m/s to ft/s for display."""
    return v_ms * FT_PER_M

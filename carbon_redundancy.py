"""
carbon_redundancy.py
====================
Two new objective modules for redundancy-aware, low-carbon truss optimisation.

(1) Embodied carbon (cradle-to-gate, EN 15978 modules A1-A3)
    EC = sum_i ( rho_steel * A_i * L_i * CF_route )
    Carbon factors (kgCO2e per kg steel), A1-A3:
        BOF (blast-oxygen furnace, BF-BOF)   CF = 2.30
        EAF (electric-arc furnace, scrap)    CF = 0.70
        Default mix (60% BOF / 40% EAF)      CF = 0.6*2.30 + 0.4*0.70 = 1.66
    BOF and EAF values are global representative figures (Energies 2025,
    19(2), 527); the 60/40 split follows RICS Professional Standard 2023.
    Reference values are documented in the manuscript; the model is a standard
    mass x emission-factor inventory used in practice and in the LCA literature.

(2) Combined reliability + redundancy robustness index (Spyridis & Strauss,
    Buildings 2020, 10, 213, Eq. 10):
        rho_red = sum_i r_ii (1 - Phi(-beta_i)) / sum_i (1 - r_ii)
    where r_ii are stiffness-determinant redundancy contributions and beta_i are
    element reliability indices. Here we adopt the element-uniform reliability
    assumption of the source paper (1 - Phi(-beta) -> 1 for beta >= 4.75), and
    let r_ii be computed from the assembled global stiffness matrix of each truss.

The redundancy components r_ii are evaluated on the *current design* (areas /
steel route), so robustness is design-dependent — which is exactly what lets it
serve as an optimisation objective rather than a fixed geometric property.
"""

import numpy as np
from scipy.stats import norm

# ---- carbon factors (kgCO2e / kg), A1-A3 ----
# BOF and EAF representative global values from Energies 2025, 19(2), 527;
# default 60/40 BOF:EAF split assumption from RICS Professional Standard 2023.
CF_BOF = 2.30      # blast-oxygen furnace (BF-BOF), global representative value
CF_EAF = 0.70      # electric-arc furnace (scrap), global representative value
CF_MIX = 0.60 * CF_BOF + 0.40 * CF_EAF   # 60/40 default split = 1.66

CARBON_FACTORS = {"BOF": CF_BOF, "EAF": CF_EAF, "MIX": CF_MIX}

# steel density [kg/m^3]; used when geometry is in SI. For imperial benchmarks
# (areas in in^2, lengths in in) we convert mass to kg before applying CF.
RHO_STEEL_SI = 7850.0           # kg/m^3
IN3_TO_M3 = (0.0254 ** 3)       # 1 in^3 in m^3
LB_PER_IN3_STEEL = 0.2836       # ~ matches RHO used in imperial benchmark codes


def embodied_carbon(member_mass_kg, route="MIX"):
    """
    Embodied carbon (kgCO2e), A1-A3, from total or per-member steel mass.
    member_mass_kg : scalar total mass or array of per-member masses (kg)
    route          : 'BOF' | 'EAF' | 'MIX'  (uniform), or array of routes per member
    """
    cf = CARBON_FACTORS
    if isinstance(route, (list, np.ndarray)):
        factors = np.array([cf[r] for r in route])
        return float(np.sum(np.asarray(member_mass_kg) * factors))
    return float(np.sum(member_mass_kg) * cf[route])


def member_mass_kg_from_imperial(A_in2, L_in):
    """Mass [kg] for imperial-unit members (areas in^2, lengths in)."""
    vol_in3 = np.asarray(A_in2) * np.asarray(L_in)
    mass_lb = vol_in3 * LB_PER_IN3_STEEL
    return mass_lb * 0.45359237  # lb -> kg


def member_mass_kg_from_si(A_m2, L_m, rho=RHO_STEEL_SI):
    """Mass [kg] for SI-unit members (areas m^2, lengths m)."""
    return np.asarray(A_m2) * np.asarray(L_m) * rho


# ----------------------------------------------------------------------
# Redundancy components r_ii from the global stiffness matrix
# ----------------------------------------------------------------------
def redundancy_components(assemble_Kfree_fn, n_elems):
    """
    Compute diagonal redundancy contributions r_ii for each element using the
    stiffness-determinant ratio (basis of Eq. 3 in Spyridis & Strauss):
        r_ii = 1 - det(K_free_without_i) / det(K_free_intact),  clipped to [0,1]

    assemble_Kfree_fn(active_mask) -> reduced (free-DOF) global stiffness matrix
        for the given boolean element mask.
    n_elems : number of elements.

    Returns r (n_elems,) clipped to [0,1].
    Uses slogdet for numerical stability on large/stiff matrices.
    """
    full = np.ones(n_elems, dtype=bool)
    K0 = assemble_Kfree_fn(full)
    sign0, logdet0 = np.linalg.slogdet(K0)
    r = np.zeros(n_elems)
    for e in range(n_elems):
        m = full.copy()
        m[e] = False
        Kd = assemble_Kfree_fn(m)
        signd, logdetd = np.linalg.slogdet(Kd)
        if signd <= 0 or not np.isfinite(logdetd):
            r[e] = 1.0   # removing element -> singular/mechanism -> max criticality
        else:
            ratio = np.exp(np.clip(logdetd - logdet0, -700, 0))
            r[e] = 1.0 - min(max(ratio, 0.0), 1.0)
    return np.clip(r, 0.0, 1.0)


def element_beta(stress, Fy, cov=0.05):
    """
    Element-level reliability index from member utilisation (design-dependent).
    Resistance mean = Fy (yield), demand mean = |axial stress|; both with CoV.
        beta_i = (Fy - |sigma_i|) / sqrt( (cov*Fy)^2 + (cov*|sigma_i|)^2 )
    This relaxes the uniform-beta assumption of Spyridis & Strauss (2020) so that
    rho_redundancy becomes sensitive to the cross-sectional sizing, enabling its
    use as an optimisation objective rather than a fixed topological property.
    """
    s = np.abs(np.asarray(stress, dtype=float))
    sR = cov * Fy
    sS = cov * np.maximum(s, 1e-9)
    return (Fy - s) / np.sqrt(sR**2 + sS**2)


def rho_redundancy(r, beta, eps_den=0.5):
    """
    Combined reliability+redundancy robustness index, Eq. (10):
        rho = sum_i r_ii (1 - Phi(-beta_i)) / sum_i (1 - r_ii)
    r    : (n,) redundancy components
    beta : scalar or (n,) element reliability indices
    Higher rho => more robust.

    Numerical note: as a structure approaches static determinacy, every member
    becomes critical (r_ii -> 1) and sum(1 - r_ii) -> 0, which makes the raw ratio
    blow up. A determinate (or near-determinate) truss has, by definition, no
    alternative load paths and hence minimal redundancy-robustness. We therefore
    floor the denominator at eps_den: when the available redundancy reserve
    sum(1 - r_ii) falls below eps_den the index saturates rather than diverging,
    which is the physically correct limit (no spare paths => bounded robustness).
    """
    r = np.asarray(r, dtype=float)
    beta = np.asarray(beta, dtype=float) if np.ndim(beta) else np.full_like(r, beta)
    rel = 1.0 - norm.cdf(-beta)          # ~1 for beta >= 4.75
    num = np.sum(r * rel)
    den = np.sum(1.0 - r)
    den = max(den, eps_den)
    return float(num / den)

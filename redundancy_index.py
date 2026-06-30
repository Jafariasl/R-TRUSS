"""
redundancy_index.py
===================
Redundancy / robustness objective for the RBRO formulation.

R_red(A) = mean over removed members i of  R_system( structure with member i absent )

This quantifies alternate-load-path capacity directly: a design where most
single-member removals leave a still-reliable system is robust; a design where
removals collapse the system (or push it below the reliability target) is not.
The measure is in [0, 1], sensitive to BOTH sizing and topology, numerically
stable, and independent of any proprietary redundancy-matrix formulation. It is a
direct operationalisation of the "alternative load path" robustness strategy of
Eurocode and of Spyridis & Strauss (2020), used here as an optimisation objective.

Two evaluation modes:
  exact       : remove every present member (small benchmarks)
  approximate : remove only the k most-utilised members (large benchmarks); the
                most-stressed members dominate the governing failure paths, as
                noted in the source paper, so this is a principled reduction.
We report agreement between exact and approximate on the small cases to justify
the approximation on the large ones.
"""

import numpy as np


def member_utilisation(p, A):
    """Per-member stress utilisation |sigma_i| (proxy for failure-path dominance)."""
    res = p.solve(A)
    if res is None:
        return None
    return np.abs(res["stress"])


def R_redundancy(p, A, mode="exact", k=None, area_floor=1e-3, dead_area=1e-9):
    """
    Mean residual system reliability under single-member removal.

    p          : truss problem (provides .solve, .reliability, .n_groups, .N_EL)
    A          : group areas (design vector)
    mode       : 'exact' (all present members) or 'approx' (k most-utilised)
    k          : number of members to probe in 'approx' mode (default ~ n/4)
    area_floor : members with A below this are considered absent (topology)
    dead_area  : tiny area used to represent a removed member in the FEM

    Returns (R_red in [0,1], n_probes, collapse_fraction).
    """
    A = np.asarray(A, float)
    n_groups = p.n_groups
    present = np.where(A > area_floor)[0]
    if len(present) == 0:
        return 0.0, 0, 1.0

    # choose which members to probe
    if mode == "approx":
        util = member_utilisation(p, A)
        if util is None:
            return 0.0, 0, 1.0
        # map element utilisation up to groups (max within group)
        groups = p.element_groups
        gutil = np.array([util[groups == g].max() if np.any(groups == g) else 0.0
                          for g in range(n_groups)])
        kk = k or max(3, n_groups // 4)
        order = [g for g in np.argsort(-gutil) if g in set(present)]
        probe = np.array(order[:kk])
    else:
        probe = present

    Rs = []
    collapses = 0
    for i in probe:
        A2 = A.copy()
        A2[i] = dead_area
        res = p.solve(A2)
        if res is None or not np.isfinite(res["max_disp"]):
            Rs.append(0.0)
            collapses += 1
            continue
        _, R = p.reliability(A2)
        Rs.append(max(0.0, min(1.0, R)))
    if not Rs:
        return 0.0, 0, 1.0
    return float(np.mean(Rs)), len(probe), collapses / len(probe)

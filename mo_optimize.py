"""
mo_optimize.py
==============
Redundancy-aware, low-carbon multi-objective truss optimisation (RBRO).

Objectives (both minimised internally):
    f1 = embodied carbon  [kgCO2e], A1-A3, steel-route dependent
    f2 = -rho_redundancy  (maximise robustness -> minimise its negative)
Constraint:
    R(A) >= TARGET_R     (FORM reliability), plus stress/displacement limits

Design variables:
    continuous member-area groups, plus one discrete steel-route choice
    (BOF / EAF / MIX) applied to the whole structure (a circularity lever).

Two solvers:
    (a) NSGA-II via pymoo  (baseline, full FEM+FORM evaluations)
    (b) surrogate-assisted: GP models the expensive rho_redundancy objective with
        active-learning infill (expected hypervolume improvement, EHVI), while
        carbon and the reliability constraint use direct (cheap) evaluation.

The point of (b) is to cut the number of expensive redundancy/FORM evaluations;
we report FEM-call counts for both so the saving is quantifiable.
"""

import numpy as np
from scipy.stats import norm
import carbon_redundancy as cr
import redundancy_index as ri

# yield stress per benchmark unit system (psi for imperial, Pa for SI)
YIELD = {"imperial": 36000.0, "SI": 355e6}


class RBROProblem:
    """Bundles a verified truss benchmark with the two RBRO objectives."""

    def __init__(self, prob, route="MIX", target_R=None, red_mode="exact", red_k=None):
        self.p = prob
        self.route = route
        self.target_R = target_R if target_R is not None else prob.TARGET_R
        self.Fy = YIELD[prob.units]
        self.n_elems = getattr(prob, "N_EL", prob.n_groups)
        self.red_mode = red_mode
        self.red_k = red_k
        self.eval_count = 0          # full design evaluations
        self.fem_calls = 0           # individual FEM solves (cost proxy)

    # ---- objective 1: embodied carbon ----
    def carbon(self, A):
        mass = self.p.member_mass_kg(A)
        return cr.embodied_carbon(mass, self.route)

    # ---- objective 2: redundancy (expensive: many FEM/FORM) ----
    def redundancy(self, A):
        Rred, nprobe, collapse = ri.R_redundancy(
            self.p, A, mode=self.red_mode, k=self.red_k)
        self.fem_calls += nprobe
        return Rred

    # ---- reliability + limits constraint (intact state) ----
    def feasible(self, A):
        res = self.p.solve(A)
        self.fem_calls += 1
        if res is None or not np.isfinite(res["max_disp"]):
            return False, 0.0, res
        beta, R = self.p.reliability(A)
        ok = R >= self.target_R
        if hasattr(self.p, "SIG_LIM"):
            ok = ok and (res["max_stress"] <= self.p.SIG_LIM)
        if hasattr(self.p, "DISP_LIM"):
            ok = ok and (res["max_disp"] <= self.p.DISP_LIM)
        if hasattr(self.p, "DELTA_MAX"):
            ok = ok and (res["max_disp"] <= self.p.DELTA_MAX)
        return ok, R, res

    def evaluate(self, A):
        """Full evaluation: (carbon, R_redundancy, R_intact, feasible)."""
        self.eval_count += 1
        ok, R, res = self.feasible(A)
        if res is None:
            return np.inf, 0.0, 0.0, False
        c = self.carbon(A)
        rred = self.redundancy(A)
        return c, rred, R, ok


# ----------------------------------------------------------------------
# NSGA-II baseline via pymoo
# ----------------------------------------------------------------------
def run_nsga2(rbro, pop=40, gens=40, seed=1, verbose=False):
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize as pymoo_min
    from pymoo.termination import get_termination

    lo, hi = rbro.p.bounds
    n = rbro.p.n_groups

    class _P(Problem):
        def __init__(self):
            super().__init__(n_var=n, n_obj=2, n_constr=1, xl=lo, xu=hi)

        def _evaluate(self, X, out, *a, **k):
            F = np.zeros((len(X), 2))
            G = np.zeros((len(X), 1))
            for i, x in enumerate(X):
                c, rred, R, ok = rbro.evaluate(x)
                F[i, 0] = c
                F[i, 1] = -rred           # minimise -R_red => maximise redundancy
                G[i, 0] = rbro.target_R - R   # <=0 feasible
            out["F"] = F
            out["G"] = G

    algo = NSGA2(pop_size=pop, sampling=FloatRandomSampling())
    res = pymoo_min(_P(), algo, get_termination("n_gen", gens),
                    seed=seed, verbose=verbose, save_history=False)
    return res

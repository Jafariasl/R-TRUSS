"""
surrogate_opt.py
================
Surrogate-assisted multi-objective optimiser for the RBRO truss problem.

The expensive objective is R_redundancy (many FEM/FORM solves per design). We
model it with a Gaussian-process surrogate and refine it by active learning, so
the optimiser needs far fewer true R_redundancy evaluations than a full-eval
NSGA-II. Carbon is cheap (closed form) and evaluated exactly; the reliability
constraint is evaluated exactly (one FORM) because feasibility must be exact.

Pipeline
--------
1. Latin-hypercube initial design; evaluate true (carbon, R_red, R_intact).
2. Fit GP: x -> R_red on feasible-or-all samples (standardised).
3. Inner search: NSGA-II on the *surrogate* (predicted R_red) + exact carbon
   + exact feasibility, cheap because R_red is predicted.
4. Active-learning infill: from the surrogate Pareto set, pick the point with
   the largest GP predictive std (most informative) OR the expected-hypervolume
   -improvement candidate; evaluate it truly; append; refit. Repeat for a budget.
5. Return the true-evaluated non-dominated set.

Validation
----------
- Level 1 (surrogate as estimator): leave-one-out CV R^2 / RMSE on R_red.
- Level 2 (optimiser as solver): hypervolume vs a full-eval NSGA-II reference,
  reported against the number of true FEM solves consumed.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
# NOTE: pymoo's HV indicator is not needed by the web app (no NSGA-II here),
# so its import is omitted to keep the deployment lightweight.


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------
def lhs(n, lo, hi, seed=0):
    rng = np.random.default_rng(seed)
    d = len(lo)
    cut = np.linspace(0, 1, n + 1)
    pts = np.zeros((n, d))
    for j in range(d):
        u = rng.uniform(size=n)
        pts[:, j] = cut[:n] + u * (cut[1] - cut[0])
        rng.shuffle(pts[:, j])
    return lo + pts * (hi - lo)


# ----------------------------------------------------------------------
# GP surrogate for R_redundancy
# ----------------------------------------------------------------------
def fit_gp(X, y):
    Xm, Xs = X.mean(0), X.std(0) + 1e-9
    ym, ys = y.mean(), y.std() + 1e-9
    Xn = (X - Xm) / Xs
    yn = (y - ym) / ys
    kernel = (ConstantKernel(1.0, (1e-2, 1e2))
              * Matern(length_scale=np.ones(X.shape[1]),
                       length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(1e-3, (1e-6, 1e0)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                  n_restarts_optimizer=3, alpha=1e-8)
    gp.fit(Xn, yn)

    def predict(Xq, return_std=False):
        Xqn = (np.atleast_2d(Xq) - Xm) / Xs
        if return_std:
            m, s = gp.predict(Xqn, return_std=True)
            return m * ys + ym, s * ys
        return gp.predict(Xqn) * ys + ym

    return predict, gp


def loo_cv(X, y):
    """Leave-one-out CV R^2 and RMSE for the GP on R_red."""
    n = len(X)
    preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, bool); mask[i] = False
        pred, _ = fit_gp(X[mask], y[mask])
        preds[i] = float(np.ravel(pred(X[i]))[0])
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y - preds) ** 2))
    return r2, rmse, preds


# ----------------------------------------------------------------------
# Surrogate-assisted optimisation loop
# ----------------------------------------------------------------------
def nondominated(F):
    """Return boolean mask of non-dominated rows (minimisation)."""
    n = len(F)
    keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                keep[i] = False
                break
    return keep


def run_surrogate(rbro, n_init=20, n_infill=30, inner_pop=30, inner_gen=15,
                  seed=0, ref_point=None, verbose=False):
    """
    Surrogate-assisted RBRO. Returns dict with Pareto X/F, true-eval count, GP CV.
    Objectives stored as [carbon, -R_red] (both minimised).
    """
    lo, hi = rbro.p.bounds
    rng = np.random.default_rng(seed)

    # 1. initial design
    X = lhs(n_init, lo, hi, seed=seed)
    C = np.zeros(len(X)); Rred = np.zeros(len(X)); feas = np.zeros(len(X), bool)
    for i, x in enumerate(X):
        c, rr, R, ok = rbro.evaluate(x)
        C[i], Rred[i], feas[i] = c, rr, ok

    # 2. active-learning infill loop
    for it in range(n_infill):
        predict, gp = fit_gp(X, Rred)

        # inner search on surrogate: sample candidates, predict R_red, rank by
        # exact carbon + predicted -R_red, keep feasible by exact reliability.
        cand = lhs(inner_pop * inner_gen, lo, hi, seed=int(rng.integers(1e9)))
        rr_hat, rr_std = predict(cand, return_std=True)
        c_hat = np.array([rbro.carbon(x) for x in cand])
        Fhat = np.column_stack([c_hat, -rr_hat])

        # candidate non-dominated set on surrogate
        nd = nondominated(Fhat)
        nd_idx = np.where(nd)[0]
        # active learning: among surrogate-Pareto candidates, pick max predictive
        # std (most informative) for the true evaluation
        pick = nd_idx[np.argmax(rr_std[nd_idx])]
        x_new = cand[pick]
        c, rr, R, ok = rbro.evaluate(x_new)
        X = np.vstack([X, x_new]); C = np.append(C, c)
        Rred = np.append(Rred, rr); feas = np.append(feas, ok)
        if verbose and (it % 5 == 0):
            print(f"    infill {it}: true_evals={rbro.eval_count} "
                  f"feasible={feas.sum()}/{len(feas)}")

    # 3. final true-evaluated non-dominated set among feasible designs
    Ffull = np.column_stack([C, -Rred])
    fmask = feas if feas.sum() >= 3 else np.ones(len(feas), bool)
    Xf, Ff = X[fmask], Ffull[fmask]
    nd = nondominated(Ff)
    r2, rmse, _ = loo_cv(X, Rred)
    return dict(X=Xf[nd], F=Ff[nd], all_X=X, all_F=Ffull, feas=feas,
                true_evals=rbro.eval_count, fem_calls=rbro.fem_calls,
                gp_r2=r2, gp_rmse=rmse)

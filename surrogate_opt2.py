"""
surrogate_opt2.py
=================
Surrogate-assisted multi-objective RBRO using the failure-aware two-stage
surrogate (surrogate_model.FailureAwareSurrogate) for the expensive
R_redundancy objective. Carbon is exact (cheap); reliability constraint is exact.

Infill strategy (per iteration):
  - fit surrogate on all true-evaluated samples
  - sample a large candidate pool; predict R_red (mean, std) and compute exact carbon
  - form the predicted Pareto set [carbon, -R_red_pred]
  - pick a *batch* of infill points spread along the predicted front, biased toward
    high predictive std (active learning), so the true front is filled in, not a
    single point. Evaluate the batch truly, append, refit.

This both improves the surrogate where it matters (the front) and builds a diverse
true-evaluated Pareto set, reported against the true FEM-solve budget.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import surrogate_opt as SO            # reuse lhs, nondominated
from surrogate_model import FailureAwareSurrogate, loo_cv


def _spread_select(F, std, k):
    """Select up to k indices from candidate objective rows F, spread along
    objective-1 (carbon) and biased to high predictive std (active learning)."""
    n = len(F)
    if n == 0:
        return np.array([], dtype=int)
    order = np.argsort(F[:, 0])
    if n <= k:
        return order
    bins = np.array_split(order, k)
    picks = []
    for b in bins:
        if len(b):
            picks.append(b[np.argmax(std[b])])
    return np.array(picks)


def run_surrogate2(rbro, n_init=24, n_iter=12, batch=4, pool=800,
                   seed=0, verbose=False):
    lo, hi = rbro.p.bounds
    rng = np.random.default_rng(seed)
    Fy = rbro.Fy

    # 1. initial design (true evaluations)
    X = SO.lhs(n_init, lo, hi, seed=seed)
    C = np.zeros(len(X)); R = np.zeros(len(X)); feas = np.zeros(len(X), bool)
    for i, x in enumerate(X):
        c, rr, Ri, ok = rbro.evaluate(x)
        C[i], R[i], feas[i] = c, rr, ok

    # active-learning history: per infill point, record predicted mean/std and
    # the subsequently observed true value (for parity + learning-curve plots)
    al_pred_mean, al_pred_std, al_true, al_iter, al_fem = [], [], [], [], []

    # 2. surrogate-assisted infill
    for it in range(n_iter):
        sur = FailureAwareSurrogate(rbro.p, Fy).fit(X, R)
        cand = SO.lhs(pool, lo, hi, seed=int(rng.integers(1e9)))
        rr_hat, rr_std = sur.predict(cand, return_std=True)
        c_hat = np.array([rbro.carbon(x) for x in cand])
        Fhat = np.column_stack([c_hat, -rr_hat])
        nd = SO.nondominated(Fhat)
        idx = np.where(nd)[0]
        if len(idx) < batch:
            extra = np.argsort(-rr_std)[:batch * 3]
            idx = np.unique(np.concatenate([idx, extra]))
        sel = _spread_select(Fhat[idx], rr_std[idx], batch)
        chosen = idx[sel]
        for j in chosen:
            # record surrogate prediction BEFORE observing the truth (honest parity)
            al_pred_mean.append(float(rr_hat[j]))
            al_pred_std.append(float(rr_std[j]))
            al_iter.append(it)
            c, rr, Ri, ok = rbro.evaluate(cand[j])
            al_true.append(float(rr))
            al_fem.append(rbro.fem_calls)
            X = np.vstack([X, cand[j]]); C = np.append(C, c)
            R = np.append(R, rr); feas = np.append(feas, ok)
        if verbose:
            print(f"    iter {it}: true_evals={rbro.eval_count} feas={feas.sum()}")

    # 3. final true-evaluated feasible non-dominated set (strict, for HV)
    Ffull = np.column_stack([C, -R])
    fmask = feas if feas.sum() >= 3 else np.ones(len(feas), bool)
    Xf, Ff = X[fmask], Ffull[fmask]
    nd = SO.nondominated(Ff)
    # a denser near-Pareto set for plotting: keep feasible points within a small
    # epsilon-band of the front (does not affect HV, only visual density)
    F_dense = Ff[nd]
    if nd.sum() >= 2:
        cmin, cmax = Ff[:, 0].min(), Ff[:, 0].max()
        rmin, rmax = Ff[:, 1].min(), Ff[:, 1].max()
        # normalise objectives, keep points whose distance to the front is small
        front = Ff[nd]
        fn = (front - [cmin, rmin]) / ([cmax - cmin + 1e-9, rmax - rmin + 1e-9])
        pts = (Ff[fmask if False else slice(None)] - [cmin, rmin]) / \
              ([cmax - cmin + 1e-9, rmax - rmin + 1e-9])
        dist = np.array([np.min(np.linalg.norm(fn - p, axis=1)) for p in pts])
        band = dist <= 0.04
        F_dense = Ff[band]
        F_dense = F_dense[SO.nondominated(F_dense)] if len(F_dense) else front

    # 4. final-surrogate parity on a held-out test set (true vs predicted + 90% PI)
    sur_final = FailureAwareSurrogate(rbro.p, Fy).fit(X, R)
    n_test = 40
    Xtest = SO.lhs(n_test, lo, hi, seed=seed + 777)
    y_true_test = np.array([rbro.redundancy(x) for x in Xtest])
    y_pred_test, y_std_test = sur_final.predict(Xtest, return_std=True)
    z90 = 1.645
    pi_lo = y_pred_test - z90 * y_std_test
    pi_hi = y_pred_test + z90 * y_std_test
    coverage90 = float(np.mean((y_true_test >= pi_lo) & (y_true_test <= pi_hi)))

    r2, rmse, _ = loo_cv(rbro.p, Fy, X, R)

    return dict(
        X=Xf[nd], F=Ff[nd], F_dense=F_dense, all_X=X, all_F=Ffull, feas=feas,
        true_evals=rbro.eval_count, fem_calls=rbro.fem_calls,
        gp_r2=r2, gp_rmse=rmse,
        # active-learning history (per infill point)
        al_pred_mean=np.array(al_pred_mean), al_pred_std=np.array(al_pred_std),
        al_true=np.array(al_true), al_iter=np.array(al_iter), al_fem=np.array(al_fem),
        # held-out parity / prediction-interval data
        test_true=y_true_test, test_pred=y_pred_test, test_std=y_std_test,
        test_pi_lo=pi_lo, test_pi_hi=pi_hi, test_coverage90=coverage90,
    )

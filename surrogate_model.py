"""
surrogate_model.py
==================
Failure-aware, physics-informed surrogate for the discontinuous R_redundancy
objective.

R_redundancy is a mixed function: a finite fraction of designs collapse under the
governing single-member removal (value exactly 0), the rest vary smoothly in
(0, ~0.8). A single regressor cannot capture both regimes, so we use a two-stage
model that is standard for functions with failure states:

  Stage 1 (classifier): predict collapse vs non-collapse.
  Stage 2 (GP regressor): on non-collapse designs only, predict R_redundancy.
  Prediction: 0 if classified collapse, else GP mean (with GP std for infill).

Physics-informed features: in addition to the design variables (areas), we append
the intact-state member stress utilisations |sigma_i|/Fy and the intact max
displacement. These encode *why* a removal collapses the system (a member carrying
high utilisation has little reserve), and they lift the surrogate from R^2~0 to
R^2~0.8 on the 10-bar benchmark.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.ensemble import GradientBoostingClassifier


def physics_features(p, X, Fy):
    """Append intact-state utilisation and max-disp to the design variables."""
    feats = []
    for x in np.atleast_2d(X):
        res = p.solve(x)
        if res is None or not np.isfinite(res["max_disp"]):
            feats.append(np.concatenate([x, np.ones(p.n_groups), [10.0]]))
        else:
            util = np.abs(res["stress"]) / Fy
            # collapse group stress -> per-group max for grouped trusses
            if hasattr(p, "element_groups") and len(util) != p.n_groups:
                g = p.element_groups
                util = np.array([util[g == k].max() if np.any(g == k) else 0.0
                                 for k in range(p.n_groups)])
            feats.append(np.concatenate([x, util[:p.n_groups], [res["max_disp"]]]))
    return np.array(feats)


class FailureAwareSurrogate:
    """Two-stage collapse-classifier + GP-regressor surrogate for R_redundancy."""

    def __init__(self, p, Fy, collapse_tol=1e-6):
        self.p = p
        self.Fy = Fy
        self.tol = collapse_tol
        self.Xm = self.Xs = None
        self.clf = None
        self.gp = None
        self.ym = self.ys = 0.0

    def _feat(self, X):
        return physics_features(self.p, X, self.Fy)

    def fit(self, X, y):
        Xf = self._feat(X)
        self.Xm, self.Xs = Xf.mean(0), Xf.std(0) + 1e-9
        Xn = (Xf - self.Xm) / self.Xs
        coll = (np.asarray(y) < self.tol).astype(int)

        # stage 1: classifier (handle degenerate single-class case)
        if coll.min() == coll.max():
            self.clf = ("const", int(coll[0]))
        else:
            self.clf = GradientBoostingClassifier(random_state=0).fit(Xn, coll)

        # stage 2: GP on non-collapse
        nc = coll == 0
        if nc.sum() >= 5:
            ker = (C(1.0) * Matern(length_scale=np.ones(Xn.shape[1]), nu=2.5)
                   + WhiteKernel(1e-2))
            self.gp = GaussianProcessRegressor(kernel=ker, alpha=1e-6,
                                               n_restarts_optimizer=2)
            self.ym, self.ys = y[nc].mean(), y[nc].std() + 1e-9
            self.gp.fit(Xn[nc], (np.asarray(y)[nc] - self.ym) / self.ys)
        else:
            self.gp = None
            self.ym = float(np.mean(y[nc])) if nc.any() else 0.0
        return self

    def _is_collapse(self, Xn):
        if isinstance(self.clf, tuple):
            return np.full(len(Xn), self.clf[1])
        return self.clf.predict(Xn)

    def predict(self, X, return_std=False):
        Xf = self._feat(X)
        Xn = (Xf - self.Xm) / self.Xs
        coll = self._is_collapse(Xn)
        if self.gp is not None:
            m, s = self.gp.predict(Xn, return_std=True)
            mean = m * self.ys + self.ym
            std = s * self.ys
        else:
            mean = np.full(len(Xn), self.ym)
            std = np.full(len(Xn), self.ys)
        mean = np.where(coll == 1, 0.0, mean)
        mean = np.clip(mean, 0.0, 1.0)
        if return_std:
            # collapse-predicted points carry classifier uncertainty -> keep std
            return mean, std
        return mean


def loo_cv(p, Fy, X, y):
    """Leave-one-out CV R^2 / RMSE for the two-stage surrogate."""
    n = len(X)
    preds = np.zeros(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        s = FailureAwareSurrogate(p, Fy).fit(X[m], np.asarray(y)[m])
        preds[i] = float(np.ravel(s.predict(X[i:i+1]))[0])
    y = np.asarray(y)
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    return 1 - ss_res / ss_tot, float(np.sqrt(np.mean((y - preds) ** 2))), preds

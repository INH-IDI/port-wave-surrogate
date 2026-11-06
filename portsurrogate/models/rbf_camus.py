"""Radial Basis Function (RBF) surrogate based on Camus et al. (2011, 2013).

Optimizes multiquadric kernel parameters (epsilon, smoothing) on training centroids.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


class CamusRBFInterpolator:
    """Hybrid downscaling using RBF interpolation for discrete harbor control stations."""

    def __init__(self, epsilon_init: float = 1.0, smooth_init: float = 0.0):
        self.epsilon_init = epsilon_init
        self.smooth_init = smooth_init
        self.rbf_models: Dict[int, RBFInterpolator] = {}
        self.optimized_params: Dict[int, np.ndarray] = {}
        self.minis: Optional[np.ndarray] = None
        self.maxis: Optional[np.ndarray] = None

    def normalize(
        self,
        X: np.ndarray,
        ix_scalar: List[int],
        ix_directional: List[int],
        minis: Optional[np.ndarray] = None,
        maxis: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize scalar parameters to [0, 1] and directions to radians."""
        X_norm = np.zeros_like(X, dtype=np.float64)
        if minis is None or maxis is None:
            min_l, max_l = [], []
            for ix in ix_scalar:
                v = X[:, ix]
                mi, ma = float(np.amin(v)), float(np.amax(v))
                if abs(ma - mi) < 1e-12:
                    ma = mi + 1.0
                X_norm[:, ix] = (v - mi) / (ma - mi)
                min_l.append(mi)
                max_l.append(ma)
            minis = np.array(min_l)
            maxis = np.array(max_l)
        else:
            for c, ix in enumerate(ix_scalar):
                v = X[:, ix]
                mi, ma = minis[c], maxis[c]
                if abs(ma - mi) < 1e-12:
                    ma = mi + 1.0
                X_norm[:, ix] = (v - mi) / (ma - mi)

        for ix in ix_directional:
            X_norm[:, ix] = X[:, ix] * np.pi / 180.0

        return X_norm, minis, maxis

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        ix_scalar: Optional[List[int]] = None,
        ix_directional: Optional[List[int]] = None,
        verbose: bool = False,
    ):
        """Fit an optimized multiquadric RBF for each target variable/station in Y_train."""
        X_train = np.asarray(X_train, dtype=np.float64)
        Y_train = np.asarray(Y_train, dtype=np.float64)
        if Y_train.ndim == 1:
            Y_train = Y_train[:, None]

        if ix_directional is None:
            ix_directional = []
        if ix_scalar is None:
            ix_scalar = [i for i in range(X_train.shape[1]) if i not in ix_directional]

        self.ix_scalar = ix_scalar
        self.ix_directional = ix_directional

        X_norm, minis, maxis = self.normalize(X_train, ix_scalar, ix_directional)
        self.minis = minis
        self.maxis = maxis
        self.X_train_norm = X_norm

        n_targets = Y_train.shape[1]
        for i in range(n_targets):
            y_i = Y_train[:, i]

            def cost_fn(params):
                eps = max(1e-4, float(params[0]))
                sm = max(0.0, float(params[1]))
                rbf = RBFInterpolator(X_norm, y_i, kernel="multiquadric", epsilon=eps, smoothing=sm)
                pred = rbf(X_norm)
                return mean_squared_error(y_i, pred)

            init_p = [self.epsilon_init, self.smooth_init]
            res = minimize(cost_fn, init_p, method="Nelder-Mead")
            opt_p = res.x
            self.optimized_params[i] = opt_p

            rbf_opt = RBFInterpolator(
                X_norm, y_i, kernel="multiquadric", epsilon=float(opt_p[0]), smoothing=float(opt_p[1])
            )
            self.rbf_models[i] = rbf_opt

            if verbose:
                print(f"  RBF target {i}: epsilon={opt_p[0]:.4f}, smoothing={opt_p[1]:.4f}")

    def predict(self, X_eval: np.ndarray) -> np.ndarray:
        """Predict target wave agitation at control stations for arbitrary input forcing."""
        X_eval = np.asarray(X_eval, dtype=np.float64)
        X_norm, _, _ = self.normalize(X_eval, self.ix_scalar, self.ix_directional, self.minis, self.maxis)

        n_targets = len(self.rbf_models)
        preds = np.zeros((X_eval.shape[0], n_targets), dtype=np.float64)

        for i in range(n_targets):
            preds[:, i] = self.rbf_models[i](X_norm)

        return preds

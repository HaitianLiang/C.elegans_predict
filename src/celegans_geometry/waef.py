"""Weighted-Adjacency Effective-Field Regression (WAEF)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import mean_squared_error, r2_score

WAEF_COLUMNS = [
    "wa_mother_x",
    "wa_mother_y",
    "wa_mother_z",
    "wa_degree",
    "wa_delta_x",
    "wa_delta_y",
    "wa_delta_z",
    "wa_q_x",
    "wa_q_y",
    "wa_q_z",
]


def _rms_standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.sqrt(np.mean(X * X, axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    return X / scale, scale


def _outer_beta(base: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.dot(base, base))
    return float(np.dot(base, y) / denom) if denom > 1e-12 else 0.0


@dataclass
class WAEFRegressor:
    target: str
    n_starts: int = 12
    random_state: int = 20260711
    max_iter: int = 1200

    def _base(self, u: np.ndarray, extra: np.ndarray) -> np.ndarray:
        if self.target.endswith("_mean"):
            return u
        if self.target == "x_half":
            k = np.exp(extra[0])
            return np.exp(np.clip(k * u, -30.0, 30.0))
        if self.target == "y_half":
            k = np.exp(extra[0])
            c = extra[1]
            return expit(np.clip(k * (u + c), -30.0, 30.0))
        if self.target == "z_half":
            k = np.exp(extra[0])
            c = extra[1]
            return expit(np.clip(k * (np.abs(u) - c), -30.0, 30.0))
        raise ValueError(f"Unsupported WAEF target: {self.target}")

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "WAEFRegressor":
        values = X[WAEF_COLUMNS].to_numpy(float)
        y = np.asarray(y, dtype=float)
        Z, self.rms_scale_ = _rms_standardize(values)
        p = Z.shape[1]
        n_extra = 0 if self.target.endswith("_mean") else (1 if self.target == "x_half" else 2)
        rng = np.random.default_rng(self.random_state)

        def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            raw_w = theta[:p]
            norm = np.linalg.norm(raw_w)
            w = raw_w / max(norm, 1e-12)
            return w, theta[p:]

        def objective(theta: np.ndarray) -> float:
            w, extra = unpack(theta)
            u = Z @ w
            base = self._base(u, extra)
            beta = _outer_beta(base, y)
            pred = beta * base
            return float(mean_squared_error(y, pred))

        best = None
        bounds = [(None, None)] * p
        if n_extra >= 1:
            bounds.append((-4.0, 4.0))  # log k
        if n_extra == 2:
            bounds.append((-2.0, 2.0))  # c
        for s in range(self.n_starts):
            theta0 = np.r_[rng.normal(size=p), np.zeros(n_extra)]
            res = minimize(
                objective,
                theta0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.max_iter},
            )
            if best is None or res.fun < best.fun:
                best = res
        if best is None:
            raise RuntimeError("WAEF optimization failed")
        self.optimization_result_ = best
        self.weights_, self.extra_ = unpack(best.x)
        u = Z @ self.weights_
        base = self._base(u, self.extra_)
        self.beta_ = _outer_beta(base, y)
        pred = self.beta_ * base
        self.train_r2_ = float(r2_score(y, pred))
        self.train_mse_ = float(mean_squared_error(y, pred))
        return self

    def transform_field(self, X: pd.DataFrame) -> np.ndarray:
        Z = X[WAEF_COLUMNS].to_numpy(float) / self.rms_scale_
        return Z @ self.weights_

    def predict(self, X: pd.DataFrame, refit_outer_beta_y: np.ndarray | None = None) -> np.ndarray:
        u = self.transform_field(X)
        base = self._base(u, self.extra_)
        beta = self.beta_
        if refit_outer_beta_y is not None:
            beta = _outer_beta(base, np.asarray(refit_outer_beta_y, dtype=float))
        return beta * base

    def weight_table(self) -> pd.DataFrame:
        return pd.DataFrame({"primitive": WAEF_COLUMNS, "weight": self.weights_, "abs_weight": np.abs(self.weights_)})

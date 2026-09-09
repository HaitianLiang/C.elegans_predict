"""FixedTermOLS reference implementation: fixed-alpha Lasso + WAIC/R2 greedy selection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

EPS = 1e-12


def approx_waic(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
    """Gaussian plug-in WAIC proxy used by this public reference implementation.

    The project reports specify WAIC-based selection but do not document the
    exact numerical WAIC routine.  This function therefore makes the public
    implementation explicit rather than claiming byte-for-byte identity with
    an archived exploratory notebook.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = max(float(mean_squared_error(y_true, y_pred)), EPS)
    log_lik = -0.5 * (np.log(2.0 * np.pi * mse) + (y_true - y_pred) ** 2 / mse)
    return float(-2.0 * (np.sum(log_lik) - int(n_params)))


def fit_no_intercept(A: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return coef, A @ coef


def regression_metrics(y: np.ndarray, pred: np.ndarray, n_params: int) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "r2": float(r2_score(y, pred)),
        "mse": float(mean_squared_error(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "waic": approx_waic(y, pred, n_params),
    }


def rms_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    scale = np.sqrt(np.mean(X * X, axis=0))
    scale = np.where(scale > EPS, scale, 1.0)
    return X / scale, scale


def lasso_rank(
    X: pd.DataFrame,
    y: np.ndarray,
    alpha: float,
    top_k: int,
    rms_normalize: bool = True,
    max_iter: int = 50_000,
) -> list[str]:
    values = X.to_numpy(float)
    values_scaled = rms_scale(values)[0] if rms_normalize else values
    model = Lasso(
        alpha=float(alpha),
        fit_intercept=False,
        max_iter=max_iter,
        selection="cyclic",
    ).fit(values_scaled, np.asarray(y, dtype=float))
    coef = np.abs(model.coef_)
    # Stable order: coefficient magnitude, then original column position.
    order = np.lexsort((np.arange(len(coef)), -coef))
    return [X.columns[i] for i in order[: min(top_k, X.shape[1])]]


def _score_candidates(
    current: dict[str, float],
    trials: list[tuple[str, dict[str, float]]],
    r2_weight: float,
    waic_weight: float,
) -> list[tuple[str, float, float, float, dict[str, float]]]:
    dr2 = np.asarray([max(m["r2"] - current["r2"], 0.0) for _, m in trials])
    dwaic = np.asarray([max(current["waic"] - m["waic"], 0.0) for _, m in trials])
    max_r2 = max(float(dr2.max(initial=0.0)), EPS)
    max_waic = max(float(dwaic.max(initial=0.0)), EPS)
    scored = []
    for (term, metrics), a, b in zip(trials, dr2, dwaic):
        score = r2_weight * a / max_r2 + waic_weight * b / max_waic
        scored.append((term, float(score), float(a), float(b), metrics))
    return scored


@dataclass
class FixedTermOLS:
    """Sparse no-intercept regression with the documented two-stage selection rule."""

    lasso_alpha: float = 1e-3
    top_k: int = 40
    max_steps: int = 15
    r2_weight: float = 0.5
    waic_weight: float = 0.5
    rms_normalize_lasso: bool = True

    def fit(self, X: pd.DataFrame, y: Iterable[float]) -> "FixedTermOLS":
        if self.lasso_alpha <= 0:
            raise ValueError("lasso_alpha must be positive")
        if self.top_k < 1 or self.max_steps < 1:
            raise ValueError("top_k and max_steps must be positive")
        if self.r2_weight < 0 or self.waic_weight < 0 or (self.r2_weight + self.waic_weight) <= 0:
            raise ValueError("selection weights must be non-negative with positive total")
        X = X.copy()
        y = np.asarray(list(y), dtype=float)
        if len(X) != len(y):
            raise ValueError("X/y length mismatch")
        if X.isna().any().any() or not np.isfinite(X.to_numpy(float)).all():
            raise ValueError("X contains NaN/inf")
        if not np.isfinite(y).all():
            raise ValueError("y contains NaN/inf")
        self.feature_names_in_ = list(X.columns)
        self.candidate_terms_ = lasso_rank(
            X,
            y,
            alpha=self.lasso_alpha,
            top_k=self.top_k,
            rms_normalize=self.rms_normalize_lasso,
        )

        zero_pred = np.zeros_like(y)
        current = regression_metrics(y, zero_pred, 0)
        selected: list[str] = []
        remaining = list(self.candidate_terms_)
        rows: list[dict[str, object]] = []

        for step in range(1, min(self.max_steps, len(remaining)) + 1):
            trials: list[tuple[str, dict[str, float]]] = []
            trial_coef: dict[str, np.ndarray] = {}
            for term in remaining:
                terms = selected + [term]
                coef, pred = fit_no_intercept(X[terms].to_numpy(float), y)
                metrics = regression_metrics(y, pred, len(terms))
                trials.append((term, metrics))
                trial_coef[term] = coef
            scored = _score_candidates(
                current,
                trials,
                r2_weight=self.r2_weight,
                waic_weight=self.waic_weight,
            )
            # Deterministic tie-breaking.
            best = sorted(scored, key=lambda t: (-t[1], -t[2], -t[3], t[0]))[0]
            term, score, dr2, dwaic, metrics = best
            selected.append(term)
            remaining.remove(term)
            coef = trial_coef[term]
            rows.append(
                {
                    "step": step,
                    "added_term": term,
                    "selected_terms": ";".join(selected),
                    "score": score,
                    "delta_r2": dr2,
                    "delta_waic": dwaic,
                    **metrics,
                    "coefficients_json": json.dumps(
                        {t: float(c) for t, c in zip(selected, coef)}, sort_keys=True
                    ),
                }
            )
            current = metrics

        self.path_ = pd.DataFrame(rows)
        if self.path_.empty:
            raise RuntimeError("No terms selected")
        best_idx = int(self.path_["waic"].idxmin())
        best_step = int(self.path_.loc[best_idx, "step"])
        self.selected_terms_ = selected[:best_step]
        self.coef_, pred = fit_no_intercept(X[self.selected_terms_].to_numpy(float), y)
        self.train_metrics_ = regression_metrics(y, pred, len(self.selected_terms_))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        missing = [t for t in self.selected_terms_ if t not in X.columns]
        if missing:
            raise KeyError(f"Missing selected terms: {missing}")
        return X[self.selected_terms_].to_numpy(float) @ self.coef_

    def evaluate(self, X: pd.DataFrame, y: Iterable[float]) -> dict[str, float]:
        y = np.asarray(list(y), dtype=float)
        pred = self.predict(X)
        return regression_metrics(y, pred, len(self.selected_terms_))

    def equation(self, lhs: str = "y_hat", precision: int = 8) -> str:
        rhs = " + ".join(
            f"({c:.{precision}g})*{t}" for t, c in zip(self.selected_terms_, self.coef_)
        )
        return f"{lhs} = {rhs}"

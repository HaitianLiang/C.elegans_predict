"""Mother-by-target heteroscedastic residual utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .constants import HALF_TARGETS


def training_residual_scales(train_predictions: pd.DataFrame) -> pd.DataFrame:
    """Estimate mother-by-half-target scales from training residual RMSE."""
    rows: list[dict[str, object]] = []
    for mother, g in train_predictions.groupby("mother_name"):
        for target in HALF_TARGETS:
            true_col = f"true_{target}"
            if target not in g or true_col not in g:
                continue
            resid = g[true_col].to_numpy(float) - g[target].to_numpy(float)
            sigma = float(np.sqrt(np.mean(resid**2))) if len(resid) else np.nan
            if np.isfinite(sigma):
                sigma = max(sigma, 1e-12)
            rows.append(
                {
                    "mother_name": str(mother),
                    "target": target,
                    "sigma_train_rmse": sigma,
                }
            )
    return pd.DataFrame(rows)


def standardize_half_residuals(
    predictions: pd.DataFrame,
    scales: pd.DataFrame,
    *,
    split_name: str,
) -> pd.DataFrame:
    """Apply training-derived scales to train or held-out half residuals."""
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "mother_name",
                "target",
                "split",
                "residual",
                "sigma_train_rmse",
                "z_residual",
            ]
        )
    scale_map = {
        (str(r["mother_name"]), str(r["target"])): float(r["sigma_train_rmse"])
        for _, r in scales.iterrows()
        if np.isfinite(float(r["sigma_train_rmse"]))
    }
    rows: list[dict[str, object]] = []
    for mother, g in predictions.groupby("mother_name"):
        for target in HALF_TARGETS:
            true_col = f"true_{target}"
            if target not in g or true_col not in g:
                continue
            sigma = scale_map.get((str(mother), target))
            if sigma is None or sigma <= 0:
                continue
            residuals = g[true_col].to_numpy(float) - g[target].to_numpy(float)
            for event_id, residual in zip(g["event_id"], residuals):
                rows.append(
                    {
                        "event_id": event_id,
                        "mother_name": str(mother),
                        "target": target,
                        "split": split_name,
                        "residual": float(residual),
                        "sigma_train_rmse": float(sigma),
                        "z_residual": float(residual / sigma),
                    }
                )
    return pd.DataFrame(rows)


def fit_zero_center_student_t(z: np.ndarray) -> dict[str, float]:
    """Fit a Student-t distribution with location fixed at zero."""
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if len(z) < 10:
        return {"df": np.nan, "loc": 0.0, "scale": np.nan, "n": int(len(z))}
    df, loc, scale = stats.t.fit(z, floc=0.0)
    return {"df": float(df), "loc": float(loc), "scale": float(scale), "n": int(len(z))}


def reported_residual_layer() -> dict[str, float]:
    """Residual parameters reported in the 2026-09-07 project analysis."""
    return {"df": 13.0, "loc": 0.0, "scale_multiplier": 1.08}


def axis_residual_correlations(z_residuals: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix of standardized x/y/z half residuals by event."""
    if z_residuals.empty:
        return pd.DataFrame()
    wide = z_residuals.pivot_table(index="event_id", columns="target", values="z_residual")
    cols = [c for c in HALF_TARGETS if c in wide]
    return wide[cols].corr() if cols else pd.DataFrame()

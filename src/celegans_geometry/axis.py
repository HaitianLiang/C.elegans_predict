"""Continuous undirected 3D division-axis statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def unit_axis(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    v = np.asarray(d1, dtype=float) - np.asarray(d2, dtype=float)
    n = np.linalg.norm(v)
    if n <= 1e-12:
        return np.full(3, np.nan)
    return v / n


def orientation_tensor(axes: np.ndarray) -> np.ndarray:
    axes = np.asarray(axes, dtype=float)
    axes = axes[np.isfinite(axes).all(axis=1)]
    if len(axes) == 0:
        return np.full((3, 3), np.nan)
    return np.einsum("ni,nj->ij", axes, axes) / len(axes)


def dominant_axis(axes: np.ndarray) -> tuple[np.ndarray, float, float]:
    M = orientation_tensor(axes)
    if not np.isfinite(M).all():
        return np.full(3, np.nan), np.nan, np.nan
    vals, vecs = np.linalg.eigh(M)
    i = int(np.argmax(vals))
    axis = vecs[:, i]
    lam1 = float(vals[i])
    strength = float((3.0 * lam1 - 1.0) / 2.0)
    return axis, lam1, strength


def undirected_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return np.nan
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return np.nan
    c = abs(float(np.dot(a, b) / (na * nb)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def division_length(d1: np.ndarray, d2: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(d1, dtype=float) - np.asarray(d2, dtype=float)))


def summarize_mother_axes(
    frame: pd.DataFrame,
    d1_cols: tuple[str, str, str],
    d2_cols: tuple[str, str, str],
) -> pd.DataFrame:
    rows = []
    for mother, g in frame.groupby("mother_name"):
        axes = np.vstack(
            [
                unit_axis(r[list(d1_cols)].to_numpy(float), r[list(d2_cols)].to_numpy(float))
                for _, r in g.iterrows()
            ]
        )
        dom, lam1, strength = dominant_axis(axes)
        deviations = np.array([undirected_angle_deg(u, dom) for u in axes], dtype=float)
        lengths = np.array(
            [
                division_length(r[list(d1_cols)].to_numpy(float), r[list(d2_cols)].to_numpy(float))
                for _, r in g.iterrows()
            ]
        )
        rows.append(
            {
                "mother_name": mother,
                "n": len(g),
                "axis_x": dom[0],
                "axis_y": dom[1],
                "axis_z": dom[2],
                "lambda1": lam1,
                "axial_strength": strength,
                "median_axis_deviation_deg": float(np.nanmedian(deviations)),
                "median_division_length": float(np.nanmedian(lengths)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_true_pred_axes(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate event-level and mother-level true/predicted axes.

    Required columns: daughter1_x/y/z, daughter2_x/y/z,
    pred_daughter1_x/y/z, pred_daughter2_x/y/z.
    """
    rows = []
    for _, r in frame.iterrows():
        t1 = r[["daughter1_x", "daughter1_y", "daughter1_z"]].to_numpy(float)
        t2 = r[["daughter2_x", "daughter2_y", "daughter2_z"]].to_numpy(float)
        p1 = r[["pred_daughter1_x", "pred_daughter1_y", "pred_daughter1_z"]].to_numpy(float)
        p2 = r[["pred_daughter2_x", "pred_daughter2_y", "pred_daughter2_z"]].to_numpy(float)
        ut = unit_axis(t1, t2)
        up = unit_axis(p1, p2)
        lt = division_length(t1, t2)
        lp = division_length(p1, p2)
        rows.append(
            {
                "event_id": r["event_id"],
                "mother_name": r["mother_name"],
                "axis_angle_deg": undirected_angle_deg(ut, up),
                "length_ratio": float(lp / lt) if lt > 1e-12 else np.nan,
            }
        )
    event_metrics = pd.DataFrame(rows)

    mother_rows = []
    for mother, g in frame.groupby("mother_name"):
        true_axes = np.vstack(
            [
                unit_axis(
                    r[["daughter1_x", "daughter1_y", "daughter1_z"]].to_numpy(float),
                    r[["daughter2_x", "daughter2_y", "daughter2_z"]].to_numpy(float),
                )
                for _, r in g.iterrows()
            ]
        )
        pred_axes = np.vstack(
            [
                unit_axis(
                    r[["pred_daughter1_x", "pred_daughter1_y", "pred_daughter1_z"]].to_numpy(float),
                    r[["pred_daughter2_x", "pred_daughter2_y", "pred_daughter2_z"]].to_numpy(float),
                )
                for _, r in g.iterrows()
            ]
        )
        true_dom, _, true_strength = dominant_axis(true_axes)
        pred_dom, _, pred_strength = dominant_axis(pred_axes)
        mother_rows.append(
            {
                "mother_name": mother,
                "n": len(g),
                "dominant_axis_angle_deg": undirected_angle_deg(true_dom, pred_dom),
                "true_axial_strength": true_strength,
                "pred_axial_strength": pred_strength,
                "true_median_deviation_deg": float(np.nanmedian([undirected_angle_deg(u, true_dom) for u in true_axes])),
                "pred_median_deviation_deg": float(np.nanmedian([undirected_angle_deg(u, pred_dom) for u in pred_axes])),
            }
        )
    return event_metrics, pd.DataFrame(mother_rows)

"""3D reconstruction and geometry metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


@dataclass
class AxisSignTemplate:
    """Training-only daughter-label sign template for absolute-half reconstruction.

    The six-target representation stores absolute half-distances. To recover
    labeled daughter coordinates, each mother needs a sign convention. This
    helper estimates the per-axis sign of daughter1-daughter2 from training
    events only. Undirected division-axis metrics are invariant to swapping the
    two daughters globally, but not to independently changing axis signs.
    """

    signs_by_mother: dict[str, np.ndarray]

    @classmethod
    def fit(cls, train_events: pd.DataFrame) -> "AxisSignTemplate":
        signs: dict[str, np.ndarray] = {}
        for mother, g in train_events.groupby("mother_name"):
            diffs = np.column_stack(
                [
                    g[f"daughter1_{a}"].to_numpy(float) - g[f"daughter2_{a}"].to_numpy(float)
                    for a in "xyz"
                ]
            )
            med = np.median(diffs, axis=0)
            s = np.sign(med)
            s[s == 0] = 1.0
            signs[str(mother)] = s.astype(float)
        return cls(signs)

    def sign(self, mother: str) -> np.ndarray:
        return self.signs_by_mother.get(str(mother), np.ones(3, dtype=float))


def reconstruct_daughters(
    pred: pd.DataFrame,
    mothers: pd.Series | list[str],
    sign_template: AxisSignTemplate,
) -> tuple[np.ndarray, np.ndarray]:
    centers = pred[["x_mean", "y_mean", "z_mean"]].to_numpy(float)
    halves = pred[["x_half", "y_half", "z_half"]].to_numpy(float)
    signs = np.vstack([sign_template.sign(m) for m in mothers])
    signed_half = halves * signs
    return centers + signed_half, centers - signed_half


def daughter_position_rmse(
    true_events: pd.DataFrame,
    pred: pd.DataFrame,
    sign_template: AxisSignTemplate,
) -> float:
    d1p, d2p = reconstruct_daughters(pred, true_events["mother_name"], sign_template)
    d1t = true_events[["daughter1_x", "daughter1_y", "daughter1_z"]].to_numpy(float)
    d2t = true_events[["daughter2_x", "daughter2_y", "daughter2_z"]].to_numpy(float)
    return float(np.sqrt(mean_squared_error(np.vstack([d1t, d2t]), np.vstack([d1p, d2p]))))

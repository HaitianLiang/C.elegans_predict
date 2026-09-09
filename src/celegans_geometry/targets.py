"""Six-target daughter-pair representation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def daughter_pair_targets(d1: np.ndarray, d2: np.ndarray) -> dict[str, float]:
    """Return mean/absolute-half targets for two 3D daughter coordinates."""
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)
    if d1.shape != (3,) or d2.shape != (3,):
        raise ValueError("d1 and d2 must be 3-vectors")
    mean = (d1 + d2) / 2.0
    half = np.abs(d1 - d2) / 2.0
    return {
        "x_mean": float(mean[0]),
        "x_half": float(half[0]),
        "y_mean": float(mean[1]),
        "y_half": float(half[1]),
        "z_mean": float(mean[2]),
        "z_half": float(half[2]),
    }


def attach_targets(events: pd.DataFrame) -> pd.DataFrame:
    """Attach six targets to a canonical event table.

    Expected daughter columns are daughter1_x/y/z and daughter2_x/y/z.
    """
    out = events.copy()
    for axis in "xyz":
        a = out[f"daughter1_{axis}"].astype(float)
        b = out[f"daughter2_{axis}"].astype(float)
        out[f"{axis}_mean"] = (a + b) / 2.0
        out[f"{axis}_half"] = (a - b).abs() / 2.0
    return out

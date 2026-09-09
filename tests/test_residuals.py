import numpy as np
import pandas as pd

from celegans_geometry.residuals import (
    standardize_half_residuals,
    training_residual_scales,
)


def _frame(split):
    return pd.DataFrame(
        {
            "event_id": [f"{split}1", f"{split}2"],
            "mother_name": ["ABa", "ABa"],
            "x_half": [1.0, 2.0],
            "true_x_half": [2.0, 4.0],
            "y_half": [1.0, 1.0],
            "true_y_half": [1.0, 3.0],
            "z_half": [2.0, 2.0],
            "true_z_half": [2.0, 4.0],
        }
    )


def test_training_scales_are_reused_for_heldout_standardization():
    train = _frame("tr")
    test = _frame("te")
    scales = training_residual_scales(train)
    z = standardize_half_residuals(test, scales, split_name="test")
    sx = scales.loc[scales.target == "x_half", "sigma_train_rmse"].iloc[0]
    expected = np.sqrt((1.0**2 + 2.0**2) / 2.0)
    assert abs(sx - expected) < 1e-12
    assert set(z["split"]) == {"test"}
    assert np.isfinite(z["z_residual"]).all()

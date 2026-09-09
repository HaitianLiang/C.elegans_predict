"""Random-forest feature audit utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .features import feature_block


def random_forest_feature_audit(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    n_estimators: int = 500,
    random_state: int = 20260711,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        oob_score=True,
        bootstrap=True,
    ).fit(X, Y)
    detail = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    detail["block"] = detail["feature"].map(feature_block)
    by_block = detail.groupby("block", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    return detail, by_block

"""Mother-specific 70/30 FixedTermOLS fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import TARGETS
from .features import fixedterm_feature_columns
from .fixedterm import FixedTermOLS


@dataclass
class MotherFitResult:
    models: dict[str, dict[str, FixedTermOLS]]
    metrics: pd.DataFrame
    equations: pd.DataFrame
    ranked_terms: pd.DataFrame
    predictions: pd.DataFrame
    train_predictions: pd.DataFrame
    splits: pd.DataFrame


def fit_mother_specific(
    events: pd.DataFrame,
    features: pd.DataFrame,
    *,
    lasso_alpha: float = 1e-3,
    top_k: int = 40,
    max_steps: int = 15,
    test_size: float = 0.30,
    random_state: int = 20260907,
    min_train: int = 8,
    rms_normalize_lasso: bool = True,
) -> MotherFitResult:
    """Fit 20 (or available) mother-specific six-target equations.

    The split is performed independently within each mother and is shared by
    all six targets for that mother.
    """
    if not events.index.equals(features.index):
        features = features.reindex(events.index)
    feature_cols = fixedterm_feature_columns(features)
    if not feature_cols:
        raise ValueError("No FixedTermOLS candidate features were found")
    models: dict[str, dict[str, FixedTermOLS]] = {}
    metric_rows: list[dict[str, Any]] = []
    equation_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    train_pred_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    for mother, g in events.groupby("mother_name", sort=True):
        idx = g.index.to_numpy()
        if len(idx) < max(min_train + 2, 6):
            continue
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=random_state, shuffle=True
        )
        if len(train_idx) < min_train:
            continue
        models[str(mother)] = {}
        for i in train_idx:
            split_rows.append({"event_id": events.loc[i, "event_id"], "mother_name": mother, "split": "train"})
        for i in test_idx:
            split_rows.append({"event_id": events.loc[i, "event_id"], "mother_name": mother, "split": "test"})

        # One wide prediction record per event, filled target by target.
        test_wide = {
            int(i): {
                "event_id": events.loc[i, "event_id"],
                "mother_name": mother,
                "split": "test",
            }
            for i in test_idx
        }
        train_wide = {
            int(i): {
                "event_id": events.loc[i, "event_id"],
                "mother_name": mother,
                "split": "train",
            }
            for i in train_idx
        }

        for target in TARGETS:
            model = FixedTermOLS(
                lasso_alpha=lasso_alpha,
                top_k=top_k,
                max_steps=max_steps,
                rms_normalize_lasso=rms_normalize_lasso,
            ).fit(features.loc[train_idx, feature_cols], events.loc[train_idx, target])
            models[str(mother)][target] = model
            test_metrics = model.evaluate(
                features.loc[test_idx, feature_cols], events.loc[test_idx, target]
            )
            metric_rows.append(
                {
                    "mother_name": mother,
                    "stage_block": g["stage_block"].iloc[0],
                    "mother_stage_count": int(g["mother_stage_count"].iloc[0]),
                    "target": target,
                    "n_total": len(idx),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_terms": len(model.selected_terms_),
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                    **{f"train_{k}": v for k, v in model.train_metrics_.items()},
                }
            )
            equation_rows.append(
                {
                    "mother_name": mother,
                    "target": target,
                    "n_terms": len(model.selected_terms_),
                    "selected_terms": ";".join(model.selected_terms_),
                    "equation": model.equation(lhs=f"{target}_hat"),
                }
            )
            for rank, term in enumerate(model.path_["added_term"].tolist(), start=1):
                rank_rows.append(
                    {
                        "mother_name": mother,
                        "target": target,
                        "rank": rank,
                        "term": term,
                    }
                )

            test_pred = model.predict(features.loc[test_idx, feature_cols])
            train_pred = model.predict(features.loc[train_idx, feature_cols])
            for i, yhat in zip(test_idx, test_pred):
                test_wide[int(i)][target] = float(yhat)
                test_wide[int(i)][f"true_{target}"] = float(events.loc[i, target])
            for i, yhat in zip(train_idx, train_pred):
                train_wide[int(i)][target] = float(yhat)
                train_wide[int(i)][f"true_{target}"] = float(events.loc[i, target])

        pred_rows.extend(test_wide.values())
        train_pred_rows.extend(train_wide.values())

    return MotherFitResult(
        models=models,
        metrics=pd.DataFrame(metric_rows),
        equations=pd.DataFrame(equation_rows),
        ranked_terms=pd.DataFrame(rank_rows),
        predictions=pd.DataFrame(pred_rows),
        train_predictions=pd.DataFrame(train_pred_rows),
        splits=pd.DataFrame(split_rows),
    )

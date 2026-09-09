"""Stage-level descriptive fits, transfer, and shared-dictionary pooling."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from .constants import HALF_TARGETS, MEAN_TARGETS, TARGETS
from .features import fixedterm_feature_columns
from .fixedterm import FixedTermOLS, fit_no_intercept


def fit_stage_descriptive(
    events: pd.DataFrame,
    features: pd.DataFrame,
    lasso_alpha: float = 1e-3,
    top_k: int = 20,
    max_steps: int = 20,
    rms_normalize_lasso: bool = True,
) -> tuple[dict[str, dict[str, FixedTermOLS]], pd.DataFrame]:
    cols = fixedterm_feature_columns(features)
    if not cols:
        raise ValueError("No FixedTermOLS candidate features were found")
    models: dict[str, dict[str, FixedTermOLS]] = {}
    rows = []
    for stage, g in events.groupby("stage_block", sort=True):
        idx = g.index
        models[str(stage)] = {}
        for target in TARGETS:
            model = FixedTermOLS(
                lasso_alpha=lasso_alpha,
                top_k=top_k,
                max_steps=max_steps,
                rms_normalize_lasso=rms_normalize_lasso,
            ).fit(
                features.loc[idx, cols], events.loc[idx, target]
            )
            models[str(stage)][target] = model
            rows.append({"stage": stage, "target": target, "n": len(idx), "n_terms": len(model.selected_terms_), **model.train_metrics_})
    return models, pd.DataFrame(rows)


def cross_stage_transfer(
    events: pd.DataFrame,
    features: pd.DataFrame,
    models: dict[str, dict[str, FixedTermOLS]],
) -> pd.DataFrame:
    cols = fixedterm_feature_columns(features)
    rows = []
    stages = sorted(models)
    for src in stages:
        for dst in stages:
            idx = events.index[events["stage_block"].astype(str) == dst]
            for target in TARGETS:
                y = events.loc[idx, target].to_numpy(float)
                src_model = models[src][target]
                # Direct coefficient transfer.
                pred_direct = src_model.predict(features.loc[idx, cols])
                rows.append({"source": src, "destination": dst, "target": target, "mode": "direct_coefficient", "r2": float(r2_score(y, pred_direct))})
                # Structure transfer + destination OLS refit.
                terms = src_model.selected_terms_
                coef, pred_refit = fit_no_intercept(features.loc[idx, terms].to_numpy(float), y)
                rows.append({"source": src, "destination": dst, "target": target, "mode": "structure_refit", "r2": float(r2_score(y, pred_refit))})
    return pd.DataFrame(rows)


def summarize_transfer(transfer: pd.DataFrame) -> pd.DataFrame:
    use = transfer[transfer["source"] != transfer["destination"]].copy()
    use["group"] = np.where(use["target"].isin(MEAN_TARGETS), "mean", "half")
    rows = []
    for mode, g in use.groupby("mode"):
        rows.append(
            {
                "mode": mode,
                "mean_r2": float(g.loc[g.group == "mean", "r2"].mean()),
                "half_r2": float(g.loc[g.group == "half", "r2"].mean()),
                "all_r2": float(g["r2"].mean()),
                "all_median_r2": float(g["r2"].median()),
            }
        )
    return pd.DataFrame(rows)


def shared_dictionary_pooling(
    events: pd.DataFrame,
    features: pd.DataFrame,
    stage_models: dict[str, dict[str, FixedTermOLS]],
    ridge_alpha: float = 1e-8,
) -> pd.DataFrame:
    """Reference implementation of all non-empty stage-subset pooling.

    For each subset, a target-specific dictionary is the union of terms selected
    by the participating stage models. Three pooled parameterizations are
    compared: shared coefficients, shared slopes + stage indicators, and
    stage-specific coefficients over the shared dictionary.
    """
    stages = sorted(stage_models)
    rows = []
    for size in range(1, len(stages) + 1):
        for combo in combinations(stages, size):
            combo = tuple(combo)
            combo_mask = events["stage_block"].astype(str).isin(combo)
            combo_idx = events.index[combo_mask]
            for target in TARGETS:
                dictionary = sorted(set().union(*(set(stage_models[s][target].selected_terms_) for s in combo)))
                if not dictionary:
                    continue
                X = features.loc[combo_idx, dictionary].to_numpy(float)
                y = events.loc[combo_idx, target].to_numpy(float)
                stage_labels = events.loc[combo_idx, "stage_block"].astype(str).to_numpy()

                # Self-stage baseline: each row is predicted by the descriptive
                # model fitted on its own stage with its own selected structure.
                pred_self = np.zeros_like(y, dtype=float)
                for s in combo:
                    local = stage_labels == s
                    local_idx = combo_idx[local]
                    pred_self[local] = stage_models[s][target].predict(
                        features.loc[local_idx, fixedterm_feature_columns(features)]
                    )

                # Shared coefficients.
                shared = Ridge(alpha=ridge_alpha, fit_intercept=False).fit(X, y)
                pred_shared = shared.predict(X)

                # Shared slopes + one-hot stage shifts.
                onehot = np.column_stack([(stage_labels == s).astype(float) for s in combo])
                Xi = np.column_stack([X, onehot])
                indicator = Ridge(alpha=ridge_alpha, fit_intercept=False).fit(Xi, y)
                pred_indicator = indicator.predict(Xi)

                # Stage-specific coefficients over the same dictionary.
                blocks = []
                for s in combo:
                    blocks.append(X * (stage_labels == s)[:, None])
                Xs = np.column_stack(blocks)
                specific = Ridge(alpha=ridge_alpha, fit_intercept=False).fit(Xs, y)
                pred_specific = specific.predict(Xs)

                for model_name, pred in [
                    ("SelfStageFixedTerm", pred_self),
                    ("ComboSharedCoeff", pred_shared),
                    ("ComboStageIndicator", pred_indicator),
                    ("ComboStageSpecificCoeff", pred_specific),
                ]:
                    rows.append(
                        {
                            "combo": "+".join(combo),
                            "combo_size": size,
                            "target": target,
                            "model": model_name,
                            "r2": float(r2_score(y, pred)),
                            "dictionary_size": len(dictionary),
                        }
                    )
    return pd.DataFrame(rows)

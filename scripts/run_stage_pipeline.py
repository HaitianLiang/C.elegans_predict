#!/usr/bin/env python3
"""Run the stage-level FixedTermOLS structural analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from celegans_geometry.adjacency import WeightedAdjacencyStore
from celegans_geometry.audit import random_forest_feature_audit
from celegans_geometry.constants import TARGETS
from celegans_geometry.features import FeatureBuilder, fixedterm_feature_columns
from celegans_geometry.io import load_celldata, reconstruct_division_events, save_canonical_tables
from celegans_geometry.stage_models import (
    cross_stage_transfer,
    fit_stage_descriptive,
    shared_dictionary_pooling,
    summarize_transfer,
)
from celegans_geometry.waef import WAEFRegressor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-level FixedTermOLS structural analysis")
    p.add_argument("--cell-data-root", required=True)
    p.add_argument("--adj-dir", required=True)
    p.add_argument("--out", default="results/stage")
    p.add_argument(
        "--lasso-alpha",
        type=float,
        required=True,
        help="Fixed Lasso alpha; set explicitly for every biological rerun.",
    )
    p.add_argument("--lasso-scale", choices=["rms", "none"], default="rms")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--pool-ridge-alpha", type=float, default=1e-8)
    p.add_argument("--waef", action="store_true", help="Also run within-stage and cross-stage WAEF checks")
    p.add_argument("--waef-n-starts", type=int, default=12)
    p.add_argument("--waef-max-iter", type=int, default=1200)
    p.add_argument("--rf-audit", action="store_true", help="Also export stage-wise RandomForest feature importance")
    return p.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.lasso_alpha <= 0:
        raise ValueError("--lasso-alpha must be positive")
    if args.top_k < 1 or args.max_steps < 1:
        raise ValueError("--top-k and --max-steps must be positive")
    if args.pool_ridge_alpha < 0:
        raise ValueError("--pool-ridge-alpha must be non-negative")
    if args.waef_n_starts < 1 or args.waef_max_iter < 1:
        raise ValueError("WAEF optimization settings must be positive")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    positions = load_celldata(args.cell_data_root)
    events = reconstruct_division_events(positions)
    save_canonical_tables(positions, events, out / "canonical")

    adj = WeightedAdjacencyStore.from_directory(args.adj_dir)
    features = FeatureBuilder(positions, adj).transform(events)
    features.to_csv(out / "features.csv", index=False)
    feature_cols = fixedterm_feature_columns(features)
    pd.DataFrame({"feature": feature_cols}).to_csv(
        out / "fixedterm_candidate_columns.csv", index=False
    )

    models, within = fit_stage_descriptive(
        events,
        features,
        lasso_alpha=args.lasso_alpha,
        top_k=args.top_k,
        max_steps=args.max_steps,
        rms_normalize_lasso=(args.lasso_scale == "rms"),
    )
    within.to_csv(out / "stage_within_fit_metrics.csv", index=False)

    equation_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    for stage, by_target in models.items():
        for target, model in by_target.items():
            equation_rows.append(
                {
                    "stage": stage,
                    "target": target,
                    "n_terms": len(model.selected_terms_),
                    "selected_terms": ";".join(model.selected_terms_),
                    "equation": model.equation(lhs=f"{target}_hat"),
                }
            )
            for rank, term in enumerate(model.path_["added_term"].tolist(), start=1):
                rank_rows.append(
                    {"stage": stage, "target": target, "rank": rank, "term": term}
                )
    pd.DataFrame(equation_rows).to_csv(out / "stage_equations.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(out / "stage_ranked_terms.csv", index=False)

    transfer = cross_stage_transfer(events, features, models)
    transfer.to_csv(out / "cross_stage_transfer_long.csv", index=False)
    transfer_summary = summarize_transfer(transfer)
    transfer_summary.to_csv(out / "cross_stage_transfer_summary.csv", index=False)

    pooling = shared_dictionary_pooling(
        events, features, models, ridge_alpha=args.pool_ridge_alpha
    )
    pooling.to_csv(out / "stage_subset_pooling.csv", index=False)
    if len(pooling):
        typed = pooling.assign(
            target_group=lambda d: d["target"].map(
                lambda t: "mean" if t.endswith("_mean") else "half"
            )
        )
        pool_summary = (
            typed.groupby(["combo", "combo_size", "model"], as_index=False)
            .agg(
                all_r2=("r2", "mean"),
                dictionary_size=("dictionary_size", "max"),
            )
        )
        group_means = (
            typed.groupby(
                ["combo", "combo_size", "model", "target_group"], as_index=False
            )["r2"]
            .mean()
            .pivot(
                index=["combo", "combo_size", "model"],
                columns="target_group",
                values="r2",
            )
            .reset_index()
            .rename(columns={"mean": "mean_r2", "half": "half_r2"})
        )
        pool_summary.merge(
            group_means,
            on=["combo", "combo_size", "model"],
            how="left",
        ).to_csv(out / "stage_subset_pooling_summary.csv", index=False)

    if args.waef:
        waef_models: dict[str, dict[str, WAEFRegressor]] = {}
        within_rows: list[dict[str, object]] = []
        weights: list[pd.DataFrame] = []
        for stage, g in events.groupby("stage_block", sort=True):
            stage_key = str(stage)
            idx = g.index
            waef_models[stage_key] = {}
            for target in TARGETS:
                model = WAEFRegressor(
                    target=target,
                    n_starts=args.waef_n_starts,
                    max_iter=args.waef_max_iter,
                ).fit(
                    features.loc[idx], events.loc[idx, target].to_numpy(float)
                )
                waef_models[stage_key][target] = model
                pred = model.predict(features.loc[idx])
                within_rows.append(
                    {
                        "stage": stage_key,
                        "target": target,
                        "r2": float(r2_score(events.loc[idx, target], pred)),
                        "mse": float(mean_squared_error(events.loc[idx, target], pred)),
                        "beta": float(model.beta_),
                        "extra_parameters": json.dumps(model.extra_.tolist()),
                    }
                )
                w = model.weight_table()
                w["stage"] = stage_key
                w["target"] = target
                weights.append(w)
        pd.DataFrame(within_rows).to_csv(out / "waef_within_stage.csv", index=False)
        pd.concat(weights, ignore_index=True).to_csv(out / "waef_weights.csv", index=False)

        cross_rows: list[dict[str, object]] = []
        stages = sorted(waef_models)
        for src in stages:
            for dst in stages:
                idx = events.index[events["stage_block"].astype(str) == dst]
                for target in TARGETS:
                    y = events.loc[idx, target].to_numpy(float)
                    model = waef_models[src][target]
                    pred_direct = model.predict(features.loc[idx])
                    pred_beta_refit = model.predict(
                        features.loc[idx], refit_outer_beta_y=y
                    )
                    for mode, pred in [
                        ("direct", pred_direct),
                        ("refit_outer_beta", pred_beta_refit),
                    ]:
                        cross_rows.append(
                            {
                                "source": src,
                                "destination": dst,
                                "target": target,
                                "mode": mode,
                                "r2": float(r2_score(y, pred)),
                                "mse": float(mean_squared_error(y, pred)),
                            }
                        )
        pd.DataFrame(cross_rows).to_csv(out / "waef_cross_stage.csv", index=False)

    if args.rf_audit:
        for stage, g in events.groupby("stage_block", sort=True):
            idx = g.index
            detail, block = random_forest_feature_audit(
                features.loc[idx, feature_cols], events.loc[idx, TARGETS]
            )
            safe = str(stage).replace("-", "_")
            detail.to_csv(out / f"rf_importance_{safe}.csv", index=False)
            block.to_csv(out / f"rf_blocks_{safe}.csv", index=False)

    summary = {
        "n_events": int(len(events)),
        "n_mothers": int(events["mother_name"].nunique()),
        "stage_counts": {
            str(k): int(v)
            for k, v in events["stage_block"].value_counts().sort_index().items()
        },
        "lasso_alpha": args.lasso_alpha,
        "lasso_scale": args.lasso_scale,
        "top_k": args.top_k,
        "max_steps": args.max_steps,
        "pool_ridge_alpha": args.pool_ridge_alpha,
        "ran_waef": bool(args.waef),
        "waef_n_starts": args.waef_n_starts if args.waef else None,
        "waef_max_iter": args.waef_max_iter if args.waef else None,
        "ran_rf_audit": bool(args.rf_audit),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved stage-level results to: {out.resolve()}")


if __name__ == "__main__":
    main()

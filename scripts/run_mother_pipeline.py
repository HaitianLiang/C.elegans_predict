#!/usr/bin/env python3
"""Run the mother-specific weighted-adjacency FixedTermOLS analysis."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from celegans_geometry.adjacency import WeightedAdjacencyStore
from celegans_geometry.axis import evaluate_true_pred_axes, summarize_mother_axes
from celegans_geometry.features import FeatureBuilder, fixedterm_feature_columns
from celegans_geometry.io import load_celldata, reconstruct_division_events, save_canonical_tables
from celegans_geometry.lineage import benjamini_hochberg, lineage_jaccard_summary, permutation_test_lineage
from celegans_geometry.metrics import AxisSignTemplate, reconstruct_daughters
from celegans_geometry.mother_models import fit_mother_specific
from celegans_geometry.residuals import (
    axis_residual_correlations,
    fit_zero_center_student_t,
    reported_residual_layer,
    standardize_half_residuals,
    training_residual_scales,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mother-specific 70/30 weighted-adjacency FixedTermOLS analysis"
    )
    p.add_argument("--cell-data-root", required=True, help="Directory or ZIP containing CellData_*.csv")
    p.add_argument("--adj-dir", required=True, help="Directory containing weighted stage adjacency CSVs")
    p.add_argument("--out", default="results/mother", help="Output directory")
    p.add_argument(
        "--lasso-alpha",
        type=float,
        required=True,
        help=(
            "Fixed Lasso alpha. The public project reports do not record its exact "
            "numeric value, so biological reruns must set it explicitly."
        ),
    )
    p.add_argument("--lasso-scale", choices=["rms", "none"], default="rms")
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--test-size", type=float, default=0.30)
    p.add_argument(
        "--seed",
        type=int,
        default=20260907,
        help="Reference split seed; replace with the archived split seed for an exact rerun.",
    )
    p.add_argument("--permutations", type=int, default=20000)
    p.add_argument(
        "--allow-all-context-fallback",
        action="store_true",
        help="Diagnostic fallback only; the reported analysis uses weighted adjacency.",
    )
    return p.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.lasso_alpha <= 0:
        raise ValueError("--lasso-alpha must be positive")
    if args.top_k < 1 or args.max_steps < 1:
        raise ValueError("--top-k and --max-steps must be positive")
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must lie in (0,1)")
    if args.permutations < 1:
        raise ValueError("--permutations must be positive")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    positions = load_celldata(args.cell_data_root)
    events = reconstruct_division_events(positions)
    save_canonical_tables(positions, events, out / "canonical")

    adjacency = WeightedAdjacencyStore.from_directory(
        args.adj_dir, fallback_all_context=args.allow_all_context_fallback
    )
    features = FeatureBuilder(positions, adjacency).transform(events)
    features.to_csv(out / "features.csv", index=False)

    candidate_cols = fixedterm_feature_columns(features)
    pd.DataFrame({"feature": candidate_cols}).to_csv(
        out / "fixedterm_candidate_columns.csv", index=False
    )

    fit = fit_mother_specific(
        events,
        features,
        lasso_alpha=args.lasso_alpha,
        top_k=args.top_k,
        max_steps=args.max_steps,
        test_size=args.test_size,
        random_state=args.seed,
        rms_normalize_lasso=(args.lasso_scale == "rms"),
    )
    fit.metrics.to_csv(out / "mother_target_metrics.csv", index=False)
    fit.equations.to_csv(out / "final_equations.csv", index=False)
    fit.ranked_terms.to_csv(out / "ranked_terms_top15.csv", index=False)
    fit.predictions.to_csv(out / "heldout_target_predictions.csv", index=False)
    fit.train_predictions.to_csv(out / "train_target_predictions.csv", index=False)
    fit.splits.to_csv(out / "splits.csv", index=False)

    # Residual layer. Scales are estimated on training residuals and then applied
    # separately to training and held-out predictions.
    scales = training_residual_scales(fit.train_predictions)
    z_train = standardize_half_residuals(fit.train_predictions, scales, split_name="train")
    z_test = standardize_half_residuals(fit.predictions, scales, split_name="test")
    scales.to_csv(out / "mother_target_residual_scales.csv", index=False)
    z_train.to_csv(out / "standardized_half_residuals_train.csv", index=False)
    z_test.to_csv(out / "standardized_half_residuals_heldout.csv", index=False)
    corr_train = axis_residual_correlations(z_train)
    corr_test = axis_residual_correlations(z_test)
    corr_train.to_csv(out / "half_residual_correlations_train.csv")
    corr_test.to_csv(out / "half_residual_correlations_heldout.csv")
    tfit_train = fit_zero_center_student_t(z_train["z_residual"].to_numpy(float)) if len(z_train) else {}
    tfit_test = fit_zero_center_student_t(z_test["z_residual"].to_numpy(float)) if len(z_test) else {}

    # Lineage structural dictionary statistics at all profile depths.
    lineage_rows: list[dict[str, object]] = []
    pvals: list[float] = []
    for k in [5, 8, 12, 15]:
        _, summary = lineage_jaccard_summary(fit.ranked_terms, k=k)
        perm = permutation_test_lineage(
            fit.ranked_terms, k=k, n_perm=args.permutations, random_state=args.seed
        )
        p = float(perm["p_one_sided"])
        pvals.append(p)
        lineage_rows.append({"top_k": k, **summary, **perm})
    finite_positions = [i for i, p in enumerate(pvals) if np.isfinite(p)]
    if finite_positions:
        qvals = benjamini_hochberg([pvals[i] for i in finite_positions])
        for i, q in zip(finite_positions, qvals):
            lineage_rows[i]["bh_q"] = q
    pd.DataFrame(lineage_rows).to_csv(out / "lineage_dictionary_stats.csv", index=False)

    # Reconstruct held-out daughter coordinates using a training-only sign
    # convention for the absolute-half representation.
    train_ids = set(fit.splits.loc[fit.splits["split"] == "train", "event_id"])
    train_events = events[events["event_id"].isin(train_ids)]
    sign_template = AxisSignTemplate.fit(train_events)

    held = events.merge(
        fit.predictions,
        on=["event_id", "mother_name"],
        how="inner",
        suffixes=("", "_predrow"),
    )
    if len(held):
        pred_target_frame = held[
            [
                "x_mean_predrow",
                "x_half_predrow",
                "y_mean_predrow",
                "y_half_predrow",
                "z_mean_predrow",
                "z_half_predrow",
            ]
        ].copy()
        pred_target_frame.columns = [
            "x_mean",
            "x_half",
            "y_mean",
            "y_half",
            "z_mean",
            "z_half",
        ]
        d1p, d2p = reconstruct_daughters(
            pred_target_frame, held["mother_name"], sign_template
        )
        for j, axis_name in enumerate("xyz"):
            held[f"pred_daughter1_{axis_name}"] = d1p[:, j]
            held[f"pred_daughter2_{axis_name}"] = d2p[:, j]
        held.to_csv(out / "heldout_daughter_predictions.csv", index=False)
        event_axis, mother_axis = evaluate_true_pred_axes(held)
        event_axis.to_csv(out / "heldout_axis_event_metrics.csv", index=False)
        mother_axis.to_csv(out / "heldout_axis_mother_metrics.csv", index=False)
    else:
        event_axis, mother_axis = pd.DataFrame(), pd.DataFrame()

    true_axis = summarize_mother_axes(
        events,
        ("daughter1_x", "daughter1_y", "daughter1_z"),
        ("daughter2_x", "daughter2_y", "daughter2_z"),
    )
    true_axis.to_csv(out / "all_event_true_axis_summary.csv", index=False)

    summary: dict[str, object] = {
        "n_positions": int(len(positions)),
        "n_events": int(len(events)),
        "n_mothers": int(events["mother_name"].nunique()),
        "n_final_equations": int(len(fit.equations)),
        "n_fixedterm_candidates": int(len(candidate_cols)),
        "lasso_alpha": args.lasso_alpha,
        "lasso_scale": args.lasso_scale,
        "top_k": args.top_k,
        "max_steps": args.max_steps,
        "test_size": args.test_size,
        "seed": args.seed,
        "fitted_student_t_train": tfit_train,
        "fitted_student_t_heldout": tfit_test,
        "reported_residual_layer_reference": reported_residual_layer(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    if len(event_axis):
        summary["heldout_event_axis_median_deg"] = float(event_axis["axis_angle_deg"].median())
        summary["heldout_length_ratio_median"] = float(event_axis["length_ratio"].median())
    if len(mother_axis):
        summary["heldout_mother_dominant_axis_median_deg"] = float(
            mother_axis["dominant_axis_angle_deg"].median()
        )
    (out / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print(f"\nSaved results to: {out.resolve()}")


if __name__ == "__main__":
    main()

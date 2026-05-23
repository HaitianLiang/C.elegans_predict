# Cell Division Geometry Prediction Pipeline

This repository contains a small-data regression pipeline for predicting early embryonic cell-division geometry from raw `CellData` tables and stage-specific cell-contact matrices.

## Overview

The workflow has two main stages:

1. **Feature discovery / term selection**
   - Build a feature library from mother-cell coordinates, polynomial self-features, and adjacency-aware features.
   - Fit sparse models such as Lasso to identify a compact set of candidate fixed terms for each target.
   - Export selected terms or equations for downstream evaluation.

2. **Evaluation of FixedTerm models and baselines**
   - Construct mother-to-daughter division events from raw `CellData_*.csv` files.
   - Match adjacency files `G1.csv`--`G5.csv` to mother-cell stages `4/6/7/8/12`.
   - Evaluate several models over random training sizes, including FixedTerm OLS variants, simple baselines, Random Forest, and MLP.
   - Save scalar metrics, geometric metrics, model predictions, fitted coefficients, audit tables, and plots.

## Data Assumptions

Each `CellData_*.csv` file should contain at least the following columns:

```text
CellName, T, X, Y, Z
```

For each division event, the code searches within the same CSV file:

- mother cell at time `T`;
- two daughter cells at time `T + 1`;
- all same-time cells at `T` for context-based baselines.

The built-in lineage map covers early divisions such as:

```text
ABa -> ABal, ABar
ABp -> ABpl, ABpr
EMS -> MS, E
P2  -> C, P3
```

A custom lineage map can be supplied with `--lineage-json`.

## Model Variants

The evaluation script compares:

- `FixedTerm_OLS`: original FixedTerm model using contact strengths from `Gi.csv`.
- `FixedTerm_OLS_binaryAdj`: same FixedTerm library, but every nonzero contact is treated as weight `1`.
- `MotherCopyBaseline`: copies mother coordinates for mean-position targets and uses training means for split-size targets.
- `LinearRegression`: direct six-target regression from same-time cell context.
- `RandomForest`: small-data conservative Random Forest configuration.
- `MLP`: one-hidden-layer neural network baseline.

All six target columns are fitted and evaluated independently:

```text
x_mean, x_half_absdiff,
y_mean, y_half_absdiff,
z_mean, z_half_absdiff
```

## Key Outputs

A typical run produces:

```text
summary_metrics_mean_std.csv          # mean/std of R2, RMSE, MAE
per_repeat_task_metrics.csv           # per-split scalar metrics
split_vector_error_summary.csv        # geometric split-vector summary
split_vector_error_by_split.csv       # per-split geometric errors
fixed_term_refit_coefficients.csv     # refitted FixedTerm coefficients
source_event_lineage_audit.csv        # constructed source events for inspection
eval_event_lineage_audit_*.csv        # constructed eval events for inspection
detailed_split_outputs/               # y_true, metadata, and predictions per split
plots/                                # default trend plots and boxplots
```

## Example Usage

You can run `pipeline.ipynb` for the result of the method `FixedTerm_OLS` or `FixedTerm_OLS_binaryAdj`. The pipeline scripts are also provided in `cell_pipeline/`.

And then run the evaluation part `evaluate.ipynb` step by step.


## Custom Plotting

After evaluation, you can see the figures from saved CSV outputs. 

It supports:

- configurable train-size range;
- selectable model list;
- scalar line plots for `R2`, `RMSE`, and `MAE`;
- boxplots with overlaid scatter points;
- `split_vector_error` line plots and boxplots.

```

## Notes

- `Gi.csv` files are matched by the number of cells present at the mother time point, not by the local time index.
- Context features are built strictly from the same CSV and same `T` as the current division event; daughter coordinates at `T + 1` are not used as inputs.
- The macro score should be interpreted together with target-group scores, because mean-position prediction and split-geometry prediction can behave very differently.

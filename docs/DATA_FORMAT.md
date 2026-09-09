# Data format

## 1. CellData coordinate input

The loader scans recursively for files named `CellData_*.csv`. It also accepts a single ZIP file or a directory containing ZIP archives.

Each CSV must contain:

| Column | Meaning |
|---|---|
| `CellName` | canonical cell identity |
| `T` | within-embryo snapshot index |
| `X` | x coordinate |
| `Y` | y coordinate |
| `Z` | z coordinate |

A CSV is treated as one embryo input unit. If multiple source archives contain files with identical names, the loader preserves archive/subdirectory identity when constructing `embryo_id` values.

For each snapshot, `stage_count` is computed from the number of distinct cell names.

## 2. Event reconstruction

For each supported mother and embryo:

1. find the earliest snapshot containing both daughters;
2. find the latest earlier snapshot containing the mother;
3. use that earlier snapshot as the mother/context state;
4. create one event for the embryo-mother pair.

The canonical event table includes mother coordinates, daughter coordinates, stage information, and all six mean/half targets.

## 3. Weighted adjacency input

Adjacency matrices are square labeled CSV files. Supported filename conventions include:

- `G1.csv` through `G7.csv`, mapped to 4, 6, 7, 8, 12, 14, and 15 cells;
- explicit stage names such as `4.csv`, `stage_12.csv`, or other names containing one supported cell count.

The first column may contain row labels. Row and column labels should be canonical cell names.

Weights are read as floating-point values and are not binarized. A nonzero diagonal is set to zero with a warning. A nonsymmetric matrix is retained but produces a warning.

## 4. Excel adjacency workbook conversion

If adjacency information is stored in a workbook whose sheet names contain stage counts:

```bash
python scripts/convert_adjacency_workbook.py adjacency.xlsx --out data/adj
```

The converter recognizes the supported counts `4`, `6`, `7`, `8`, `12`, `14`, and `15` and writes one CSV per recognized sheet.

## 5. Mother-specific outputs

`run_mother_pipeline.py` writes the following main files.

| Output | Contents |
|---|---|
| `canonical/cell_positions.csv` | normalized cell-coordinate table |
| `canonical/division_events.csv` | reconstructed mother-division events and six targets |
| `features.csv` | full feature table, including separate FixedTerm and WAEF namespaces |
| `fixedterm_candidate_columns.csv` | exact FixedTerm candidate columns used in the run |
| `splits.csv` | mother-specific train/test event IDs |
| `final_equations.csv` | Final-WAIC equation per mother and target |
| `ranked_terms_top15.csv` | greedy ranking profile per mother and target |
| `mother_target_metrics.csv` | train and held-out target metrics |
| `train_target_predictions.csv` | training predictions and true targets |
| `heldout_target_predictions.csv` | held-out predictions and true targets |
| `mother_target_residual_scales.csv` | training-RMSE scale for each mother × half target |
| `standardized_half_residuals_train.csv` | training standardized half residuals |
| `standardized_half_residuals_heldout.csv` | held-out residuals standardized by training scale |
| `lineage_dictionary_stats.csv` | Top-5/8/12/15 lineage Jaccard/permutation summaries |
| `heldout_daughter_predictions.csv` | labeled daughter reconstruction under the training-only sign convention |
| `all_event_true_axis_summary.csv` | true dominant-axis statistics per mother |
| `heldout_axis_event_metrics.csv` | held-out event axis-angle and length ratio |
| `heldout_axis_mother_metrics.csv` | held-out mother-level true/predicted dominant-axis statistics |
| `run_summary.json` | run parameters and compact summary |

## 6. Stage-level outputs

`run_stage_pipeline.py` writes:

| Output | Contents |
|---|---|
| `canonical/` | normalized coordinates and reconstructed events |
| `features.csv` | feature table |
| `fixedterm_candidate_columns.csv` | FixedTerm candidate list |
| `stage_within_fit_metrics.csv` | stage × target descriptive fit metrics |
| `stage_equations.csv` | final stage × target equations |
| `stage_ranked_terms.csv` | full greedy term ranking per stage × target |
| `cross_stage_transfer_long.csv` | source × destination × target transfer results |
| `cross_stage_transfer_summary.csv` | off-diagonal transfer summary |
| `stage_subset_pooling.csv` | all subset × target × parameterization results |
| `stage_subset_pooling_summary.csv` | subset-level mean/half/all R2 summary |
| `waef_within_stage.csv` | optional WAEF within-stage metrics |
| `waef_weights.csv` | optional WAEF primitive weights |
| `waef_cross_stage.csv` | optional WAEF direct/refit-outer-beta cross-stage metrics |
| `rf_importance_*.csv` | optional stage-specific RandomForest feature importance |
| `rf_blocks_*.csv` | optional RandomForest importance aggregated by structural block |
| `run_summary.json` | run parameters and dataset counts |

## 7. Data not included in the repository

The repository does not redistribute the biological source data or project-private intermediate files. Keep such files in local ignored directories such as `data/raw/` or `data/private/` unless redistribution is explicitly permitted.

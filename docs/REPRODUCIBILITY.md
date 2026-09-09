# Reproducibility notes

## 1. Two different reproducibility claims

This repository distinguishes two levels of reproducibility.

### Method reproducibility

The public code makes the current workflow explicit:

- event reconstruction;
- stage-matched weighted adjacency;
- FixedTerm candidate construction;
- fixed-alpha Lasso screening;
- WAIC/R2 greedy ranking;
- no-intercept OLS refitting;
- mother-specific 70/30 evaluation;
- lineage dictionary statistics;
- training-scaled residual analysis;
- continuous undirected division-axis statistics;
- stage-level transfer and pooling;
- optional WAEF and RandomForest audits.

These steps can be rerun on any compatible data with explicit command-line parameters.

### Exact archival result reproduction

Reproducing the exact numerical tables and historical selected equations requires additional project-archive information that is not fully recorded in the two current reports.

The most important archive-dependent items are:

1. the exact fixed Lasso alpha;
2. the exact Lasso feature-scaling convention used in the historical run;
3. the exact random split seed or frozen train/test event IDs;
4. the exact low-level WAIC numerical routine;
5. the exact input-file snapshot and adjacency labels/weights;
6. the exact daughter-sign convention used for historical labeled 3D reconstruction.

The public code therefore does not hard-code guessed values for these items and does not claim byte-for-byte identity with every historical notebook.

## 2. What the current project reports establish

The mother-specific analysis reports:

- 4,056 mother-event observations;
- 20 mother division types;
- 120 Final-WAIC equations;
- 70%/30% splitting within each mother;
- fixed-alpha Lasso Top-40 screening;
- a maximum 15-step greedy structural profile;
- training-WAIC selection of the final equation;
- stage-matched weighted adjacency for the event system;
- the reported lineage, residual, and continuous-axis summaries listed in `RESULTS.md`.

The stage-level analysis reports:

- five stage blocks;
- fixed-alpha Lasso Top-20 screening;
- FixedTerm stage equations;
- direct coefficient transfer versus structure transfer with destination refit;
- all 31 non-empty stage-subset pooling experiments;
- WAEF within-stage/cross-stage analyses;
- RandomForest feature audit;
- few-shot coefficient calibration.

## 3. Explicit public implementation conventions

The current repository makes several choices explicit because they are not fully specified at low level in the reports.

### Lasso scaling

The CLI accepts:

```bash
--lasso-scale rms
```

or

```bash
--lasso-scale none
```

`rms` means no-centering RMS scaling before Lasso screening. The biological-analysis command still requires an explicit `--lasso-alpha`.

### WAIC calculation

`celegans_geometry.fixedterm.approx_waic()` implements a Gaussian plug-in WAIC proxy with variance estimated by MSE and a parameter-count penalty. This is a transparent public convention, not a claim about the exact historical notebook internals.

### Absolute-half daughter reconstruction

The public code estimates a mother-specific per-axis sign template using training events only. This satisfies the report-level requirement that daughter direction correspondence be determined from training samples, while making the exact public convention inspectable.

### Residual standardization

The public code estimates mother-by-half-target RMSE scales from training residuals, then exports standardized training and held-out residuals separately using the same training-derived scale.

## 4. Exact-rerun checklist

Before describing a run as an exact reproduction of the reported biological results, verify all of the following.

- [ ] Original CellData inputs are identical.
- [ ] Original weighted adjacency matrices are identical, including row/column labels and continuous weights.
- [ ] Event reconstruction returns 4,056 events and 20 mother types.
- [ ] The original fixed Lasso alpha is restored.
- [ ] The original Lasso scaling convention is restored.
- [ ] The original mother-level train/test split IDs are restored or regenerated with the original seed.
- [ ] The exact historical WAIC routine is confirmed.
- [ ] 120 final mother-specific equations are recovered.
- [ ] Top-5/8/12/15 lineage statistics reproduce the recorded values.
- [ ] Held-out continuous-axis summaries reproduce the recorded values.
- [ ] Environment versions and input hashes are archived with the rerun.

If any of these items is unknown, describe the run as a **reference-method rerun**, not an exact archival reproduction.

## 5. Local software validation

Install the package and run the unit tests:

```bash
pip install -e ".[dev]"
pytest
```

The tests cover:

- six-target construction;
- weighted averaging;
- FixedTerm sparse-signal recovery;
- undirected-axis sign invariance;
- dominant-axis concentration;
- FixedTerm/WAEF feature-namespace separation;
- training-derived residual scaling;
- Benjamini-Hochberg correction for the reported four-depth lineage example.

## 6. Synthetic smoke test

The synthetic dataset tests software plumbing only. It must not be interpreted as biological validation.

```bash
python scripts/make_synthetic_demo.py --out data/demo

python scripts/run_mother_pipeline.py \
  --cell-data-root data/demo/CellData \
  --adj-dir data/demo/adj \
  --out results/demo_mother \
  --lasso-alpha 0.001
```

The value `0.001` in this command is an explicit synthetic-test parameter, not a statement about the historical biological run.

## 7. Data handling

Raw biological source files are not included in the repository. The `.gitignore` excludes common local raw-data paths, ZIP archives, Excel workbooks, generated results, virtual environments, and cache files.

Derived result tables can be committed separately when their provenance and redistribution status are clear.

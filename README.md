# Interpretable *C. elegans* Cell-Division Geometry

Research code for modeling early *C. elegans* cell-division geometry with **FixedTermOLS**, stage-matched **weighted adjacency**, mother-specific structural dictionaries, residual uncertainty, and continuous 3D division-axis statistics.

## Project status

This repository is a **research-project codebase**. It contains the current method implementation and Markdown documentation used to organize the project.

The numerical results summarized in `docs/RESULTS.md` come from the current project analyses. The public code is written to reproduce the documented workflow, but an **exact archival rerun** additionally requires the original fixed Lasso $\alpha$, the original train/test split seed or split table, and the original biological input files. Those numerical settings are not fully recorded in the two project reports, so the command-line tools require the Lasso $\alpha$ to be supplied explicitly.

## Scope

The repository keeps two related analyses separate because they answer different statistical questions.

| Resolution | Statistical unit | Evaluation | Main purpose |
|---|---|---|---|
| **Mother-specific** | one mother-division type across repeated embryos | independent 70% train / 30% held-out test within each mother | estimate mother-specific geometry and residual variability |
| **Stage-level** | all division events in a developmental block | within-stage descriptive fits, cross-stage transfer, and stage pooling | study which geometric structures are reusable across stages |

The current mother-specific analysis contains **4,056 mother-event observations**, **20 mother-division types**, and **120 final equations**. The stage-level analysis groups the same event system into five developmental blocks: `4-8`, `8-12`, `12-14`, `14-15`, and `15-24`.

## Daughter-pair representation

For daughter coordinates

$$
d_1=(x_1,y_1,z_1),\qquad d_2=(x_2,y_2,z_2),
$$

the six regression targets are

$$
x_{\mathrm{mean}}=\frac{x_1+x_2}{2},\qquad
x_{\mathrm{half}}=\frac{|x_1-x_2|}{2},
$$

with the same definitions for $y$ and $z$.

The three `mean` targets describe the daughter-pair center. The three `half` targets describe coordinate-wise separation amplitude.

## Method overview

The implemented workflow is deliberately explicit and sequential:

1. reconstruct mother $\rightarrow$ two-daughter division events from CellData coordinate tables;
2. match each event to the weighted adjacency matrix for the mother-cell stage;
3. construct the FixedTerm candidate library from mother-coordinate polynomial terms, weighted neighbor moments, weighted relative displacements, and mother-neighbor couplings;
4. run fixed-$\alpha$ Lasso screening;
5. rank screened terms by the documented WAIC/$R^2$ greedy rule;
6. choose the final no-intercept FixedTermOLS equation by the minimum training-set WAIC along the greedy path;
7. retain deeper Top-5/8/12/15 rankings separately for structural-dictionary comparisons;
8. evaluate mother-specific held-out predictions, residual scales, lineage dictionary similarity, and continuous 3D division axes;
9. run the stage-level transfer, pooling, WAEF, and RandomForest audit analyses when requested.

The FixedTerm and WAEF namespaces are separated in code. WAEF primitives are **not** silently included in the FixedTerm candidate pool.

## Current reported findings

The main conclusions supported by the current analyses are summarized below. Detailed tables and qualifications are in `docs/RESULTS.md` and `docs/REPORT_CN.md`.

### Mother-specific geometry

Across the 60 final half equations, weighted-adjacency terms account for approximately **70.8% of selected terms** and approximately **62.0% of standardized contribution** using $|\beta_j|\,\mathrm{SD}(\phi_j)$.

The Top-15 half-target union dictionary shows a modest same-root lineage similarity signal: mean Jaccard **0.3918** within the same root lineage versus **0.3694** across different roots. The nominal permutation value is $p=0.0304$, but after correction across four profile depths $q\approx0.1216$. This is therefore treated as **exploratory structural evidence**, not as a lineage-specific marker claim.

For continuous division axes, the reported mother-level true axial-strength median is **0.955**. On 1,224 held-out events, the event-level true/predicted axis-angle median is **7.44°**, and the predicted/true division-length ratio has median **0.997**. After aggregating held-out events by mother, the reported true/predicted dominant-axis angle median is **1.39°**.

### Stage-level structure

The stage-level transfer experiment distinguishes coefficient transfer from structural transfer. The reported off-diagonal all-target average $R^2$ is **-72.357** for direct coefficient transfer and **0.514** when the selected source structure is transferred and the coefficients are refit in the destination stage.

The five-stage pooling analysis gives the same qualitative conclusion: a shared structural dictionary with stage-specific coefficients performs better than forcing all stages to share one coefficient vector. The reported five-stage all-target $R^2$ values are **0.897** for the self-stage baseline and **0.904** for the shared-dictionary/stage-specific-coefficient model.

WAEF is retained as a complementary low-dimensional effective-field model. It can approach FixedTermOLS within several stages, but its cross-stage effective-field direction is not stable enough to replace the explicit FixedTerm equations.

## Documentation

The Markdown files are organized by purpose rather than as parallel versions of the same text.

| File | Role |
|---|---|
| [`docs/REPORT_CN.md`](docs/REPORT_CN.md) | complete Chinese project report; scientific narrative and interpretation |
| [`docs/METHODS.md`](docs/METHODS.md) | mathematical definitions and implementation-level method specification |
| [`docs/RESULTS.md`](docs/RESULTS.md) | reported numerical results, tables, and interpretation boundaries |
| [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) | input schema, adjacency format, and generated outputs |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | what is reproduced by the public code, what remains archive-dependent, and exact-rerun checklist |
| [`data/README.md`](data/README.md) | local data placement and synthetic-data instructions |

A reader who wants the scientific story should read `README.md` and then `docs/REPORT_CN.md`. A reader who wants to rerun the code should use `docs/METHODS.md`, `docs/DATA_FORMAT.md`, and `docs/REPRODUCIBILITY.md`.

## Repository contents

| Path | Contents |
|---|---|
| `src/celegans_geometry/` | model, feature, lineage, residual, and axis utilities |
| `scripts/run_mother_pipeline.py` | mother-specific 70/30 analysis |
| `scripts/run_stage_pipeline.py` | stage-level fitting, transfer, pooling, optional WAEF/RF analyses |
| `scripts/convert_adjacency_workbook.py` | convert stage-labeled adjacency workbooks to CSV |
| `scripts/make_synthetic_demo.py` | synthetic CellData-style smoke-test generator |
| `tests/` | numerical and structural unit tests |
| `docs/` | Markdown technical documentation |

No raw biological data are included in the repository.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For local tests:

```bash
pip install -e ".[dev]"
pytest
```

## Synthetic smoke test

The synthetic demo is only a software check; its metrics are **not** biological results.

```bash
python scripts/make_synthetic_demo.py --out data/demo

python scripts/run_mother_pipeline.py \
  --cell-data-root data/demo/CellData \
  --adj-dir data/demo/adj \
  --out results/demo_mother \
  --lasso-alpha 0.001
```

The explicit `0.001` above is a synthetic-test setting only. It is not presented as the archived biological-analysis value.

## Running the biological analysis

Place the CellData source files and the weighted adjacency matrices locally as described in `docs/DATA_FORMAT.md`.

Mother-specific analysis:

```bash
python scripts/run_mother_pipeline.py \
  --cell-data-root data/raw \
  --adj-dir data/adj \
  --out results/mother \
  --lasso-alpha <FIXED_ALPHA> \
  --seed <SPLIT_SEED>
```

Stage-level analysis:

```bash
python scripts/run_stage_pipeline.py \
  --cell-data-root data/raw \
  --adj-dir data/adj \
  --out results/stage \
  --lasso-alpha <FIXED_ALPHA> \
  --waef \
  --rf-audit
```

If the adjacency information is stored in an Excel workbook:

```bash
python scripts/convert_adjacency_workbook.py adjacency.xlsx --out data/adj
```

## Important implementation distinctions

The package makes several distinctions explicit because they affect interpretation.

- **Within-stage $R^2$ is not held-out mother-specific performance.** The stage-level scores are descriptive fits used for structure analysis.
- **Final equations and structural profiles are different objects.** Final prediction uses the training-WAIC-selected path depth; Top-5/8/12/15 profiles preserve deeper ranking information for dictionary comparisons.
- **Residual scales are training-derived.** The code exports training and held-out standardized residuals separately using the same mother-by-target training RMSE scale.
- **Absolute half-targets do not retain daughter-label signs.** The reference 3D reconstruction uses a mother-specific sign convention estimated only from training events.
- **The public WAIC routine is explicit but not claimed to be byte-identical to an archived notebook.** The project reports specify WAIC-based selection but do not document the exact low-level WAIC calculation.

## Limitations

1. Stage-level contact matrices describe stage-matched weighted contact structure, not embryo-specific dynamic contact-area fluctuations.
2. The 14-15 stage is a short, atypical transition window and should not be used as the main source of general structural claims.
3. The lineage Top-15 similarity signal does not remain conventionally significant after correction across profile depths.
4. Predicted division-axis clouds are more concentrated than the real event clouds, so embryo-to-embryo directional variation is underestimated.
5. Exact reproduction of the reported numerical tables requires the original fixed Lasso $\alpha$, split definition, and input snapshot.

## License

The code is distributed under the MIT License. Source biological data are not redistributed here and remain subject to their original data-use terms.

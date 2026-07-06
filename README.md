# Cell Division Geometry Prediction Pipeline

This repository contains an interpretable small-data regression pipeline for predicting early embryonic daughter-cell positions from mother-cell and neighborhood geometry.

The main idea is that **FixedTermOLS learns a reusable structural dictionary**, while the numerical coefficients are **stage-specific** and should be calibrated for each developmental stage. The repository contains both the original notebooks and a complete result package under `results/`.

---

## Repository layout

```text
.
├── cell_pipeline/              # Pipeline scripts and helper code
├── results/                    # Main result package: reports, figures, tables, equations
│   ├── README.md               # Full English result overview
│   ├── assets/figures/         # Main figures used in the report
│   ├── docs/                   # Detailed explanations and equation documents
│   └── tables/                 # Exported metrics, equations, paths, and audit tables
├── README.md                   # This project-level overview
├── pipeline.ipynb              # Feature construction, Lasso screening, FixedTermOLS fitting
└── evaluate.ipynb              # Few-shot and baseline evaluation workflow
```

Most detailed results are documented in `results/`. In particular:

- [`results/README.md`](results/README.md): full result overview.
- [`results/docs/all_fixedterm_equations.md`](results/docs/all_fixedterm_equations.md): all stage-specific FixedTermOLS equations.
- [`results/docs/key_results_tables.md`](results/docs/key_results_tables.md): important result tables.
- [`results/docs/non_main_explorations.md`](results/docs/non_main_explorations.md): geometry-aware and other non-main explorations.

---

## Problem formulation

For each division event, the model predicts two daughter-cell positions. Instead of predicting the two daughters directly, we use a center/half-vector target representation:

$$
x_{\mathrm{mean}}=\frac{x_1+x_2}{2},
\qquad
x_{\mathrm{half}}=\frac{|x_1-x_2|}{2},
$$

and analogously for the $y$ and $z$ coordinates. The six targets are

```text
x_mean, x_half, y_mean, y_half, z_mean, z_half
```

This separation is important:

- **mean targets** describe daughter-center transport;
- **half targets** describe local division scale and separation geometry.

---

## Main modeling pipeline

### 1. Feature construction

The feature library is built from:

- mother-cell coordinates, such as $m_x,m_y,m_z$;
- mother-coordinate polynomial terms;
- neighbor raw moments, such as $\langle x_j\rangle_N$ and $\langle x_j^2\rangle_N$;
- relative displacement and local-spread terms, such as $\langle (y_j-m_y)^2\rangle_N$;
- mother-neighbor coupling terms, such as $\langle m_xx_j\rangle_N$.

Here $\langle\cdot\rangle_N$ denotes a neighborhood-weighted average.

### 2. Two-stage selection: Lasso screening then FixedTermOLS reorder

The final equations are not obtained by blindly taking a Lasso order. The selection has two explicit stages.

**Step 1: fixed-alpha Lasso screening.** Lasso screens a Top-20 candidate list from the full feature library. For path visualization, the Lasso $\alpha$ is fixed to avoid artificial oscillations caused by changing regularization strength.

**Step 2: FixedTermOLS reorder.** The Top-20 candidates are reordered using a stepwise FixedTermOLS path, while tracking $R^2$, MSE, and WAIC as terms are added.

<p align="center">
  <img src="results/assets/figures/lasso_screening_R2_overview_all_stages.png" width="760">
</p>

<p align="center">
  <img src="results/assets/figures/lasso_reorder_R2_overview_all_stages.png" width="760">
</p>

A stage-level example shows the two stages side by side:

<p align="center">
  <img src="results/assets/figures/lasso_greedy_path_4_6_7_8.png" width="820">
</p>

---

## Final model: FixedTermOLS

For each developmental stage $s$ and each target $t$, the final model is a sparse no-intercept equation:

$$
\widehat q_t^{(s)}=\sum_{k\in S_{s,t}}\beta_{s,t,k}\phi_k.
$$

FixedTermOLS is kept as the final model because it is:

- sparse;
- interpretable;
- stable in 3D morphology checks;
- compatible with a structural equation interpretation;
- useful as a low-data prior in few-shot calibration.

### Stage-wise performance summary

| stage | events | all-target mean $R^2$ | mean-target mean $R^2$ | half-target mean $R^2$ | total selected terms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4-8 | 888 | 0.883 | 0.981 | 0.784 | 34 |
| 8-12 | 872 | 0.896 | 0.986 | 0.806 | 43 |
| 12-14 | 436 | 0.546 | 0.936 | 0.156 | 36 |
| 14-15 | 186 | 0.170 | 0.248 | 0.092 | 34 |
| 15-24 | 1674 | 0.853 | 0.980 | 0.727 | 45 |

The 14-15 stage is treated as a short transition window rather than as a representative stage: it contains fewer events and only one mother-cell type.

---

## 3D morphology checks

The following figures compare true daughter-cell positions with FixedTermOLS predictions. Colors indicate mother-cell type.

<p align="center">
  <img src="results/assets/figures/fig3d_4_8.png" width="760">
</p>

<p align="center">
  <img src="results/assets/figures/fig3d_8_12.png" width="760">
</p>

<p align="center">
  <img src="results/assets/figures/fig3d_15_24.png" width="760">
</p>

Full 3D figures for all stages are available in [`results/assets/figures/`](results/assets/figures/).

---

## Structural equation interpretation

Although the six targets are fitted separately, the selected equations can be organized into interpretable structural blocks:

- mother-coordinate polynomial terms;
- neighbor raw moments;
- local spread and relative displacement terms;
- mother-neighbor coupling terms;
- mother-axis coupling terms.

The most useful structural reformulation is for the three half-coordinate equations. Define

$$
h=(x_{\mathrm{half}},y_{\mathrm{half}},z_{\mathrm{half}})^\top,
$$

$$
Q_\alpha=\langle(\alpha_j-m_\alpha)^2\rangle_N,
\qquad
\mu_\alpha=\langle\alpha_j\rangle_N,
\qquad
M_{\alpha,k}=\langle\alpha_j^k\rangle_N.
$$

Then the three half equations can be written as a unified half-vector structure:

$$
\widehat h^{(s)}=A_sQ+B_s(m)\mu+C_s(m)M+p_s(m).
$$

This is not a new model. It is a structural reorganization of the already fitted FixedTermOLS equations.

<p align="center">
  <img src="results/assets/figures/half_block_lines.png" width="760">
</p>

The half-vector structure suggests a stage-wise transition:

- early stages are more local-neighborhood driven;
- intermediate stages rely more on neighbor shape moments;
- later stages show stronger mother-axis coupling.

---

## Stable local-spread motif: $Q_y \to x_{\rm half}$

The clearest target-specific local-spread motif is

$$
Q_y=\langle(y_j-m_y)^2\rangle_N \quad \longrightarrow \quad x_{\mathrm{half}}.
$$

This should be interpreted as a **conditional local-spread channel**, not as a universal univariate law. It is repeatedly selected by FixedTermOLS in multiple stages, and it is most clearly visible in the non-transition stages.

<p align="center">
  <img src="results/assets/figures/q_local_spread_heatmap.png" width="760">
</p>

<p align="center">
  <img src="results/assets/figures/qy_scatter_allstages.png" width="760">
</p>

---

## Cross-stage transfer: structures transfer better than coefficients

Cross-stage transfer experiments separate two questions.

### Direct coefficient transfer

Apply source-stage coefficients directly to a destination stage:

$$
\widehat q_t^{(d)}=\sum_{k\in S_s}\beta_{s,t,k}\phi_k^{(d)}.
$$

This mostly fails off-diagonal, which means that numerical coefficients are not stable across stages.

### Feature transfer with destination refit

Transfer only the selected feature structure, then refit coefficients on the destination stage:

$$
\widehat q_t^{(d)}=\sum_{k\in S_s}\beta_{d,t,k}^{\mathrm{refit}}\phi_k^{(d)}.
$$

This works much better. The conclusion is:

> selected structures are partially transferable, but coefficients are stage-specific.

| transfer mode | off-diagonal mean-target $R^2$ | off-diagonal half-target $R^2$ | off-diagonal all-target $R^2$ |
| --- | ---: | ---: | ---: |
| direct coefficients | -7.263 | -137.451 | -72.357 |
| feature transfer + refit | 0.784 | 0.244 | 0.514 |

<p align="center">
  <img src="results/assets/figures/transfer_direct_all.png" width="740">
</p>

<p align="center">
  <img src="results/assets/figures/transfer_refit_all.png" width="740">
</p>

---

## RandomForest feature audit

RandomForest is used only as a nonlinear feature audit, not as the final interpretable model. It checks whether the structural feature blocks selected by FixedTermOLS are also important to a nonlinear model.

<p align="center">
  <img src="results/assets/figures/rf_block_importance.png" width="740">
</p>

The audit supports the significance of broad structural blocks such as mother-coordinate polynomial terms, neighbor moments, relative displacement features, and mother-neighbor couplings. However, RF importance is stage-level and multi-output, so target-specific channels such as $Q_y\to x_{\rm half}$ should still be interpreted through FixedTermOLS equations.

---

## Few-shot coefficient calibration

The transfer results imply that the selected structure dictionary is more stable than coefficients. We therefore test a few-shot coefficient calibration setting:

1. sample a small number of destination-stage events;
2. keep a sparse FixedTerm structure fixed;
3. refit only the coefficients;
4. evaluate on the remaining events.

The tested training sizes are

$$
n_{\rm train}\in\{5,10,20,40,80\}.
$$

The key result is that FixedTerm structures are effective low-data priors. Compared with full-feature linear baselines, sparse FixedTerm structures are much more stable in low-data regimes.

<p align="center">
  <img src="results/assets/figures/fewshot_global_panel.png" width="860">
</p>

<p align="center">
  <img src="results/assets/figures/fewshot_stage_allR2.png" width="860">
</p>

<p align="center">
  <img src="results/assets/figures/fewshot_stage_rmse.png" width="860">
</p>

### Example: global ranking at train size 40

| model | all-target $R^2$ | mean-target $R^2$ | half-target $R^2$ | position RMSE |
| --- | ---: | ---: | ---: | ---: |
| Same-stage FixedTerm | 0.590 | 0.776 | 0.405 | 1.782 |
| RF quick | 0.536 | 0.725 | 0.347 | 2.861 |
| Union FixedTerm | 0.511 | 0.740 | 0.282 | 1.831 |
| Prev-stage FixedTerm | 0.465 | 0.748 | 0.181 | 2.195 |
| 15-24 FixedTerm | 0.445 | 0.712 | 0.178 | 1.946 |
| Full ridge | 0.341 | 0.610 | 0.072 | 2.057 |

---

## Non-main explorations

Several geometry-aware corrections were explored, including joint six-target corrections, direction models, radial calibration, and geometry-weighted variants. These experiments are not used as the final main model because they did not consistently improve both quantitative metrics and 3D morphology.

See [`results/docs/non_main_explorations.md`](results/docs/non_main_explorations.md) for details.

---

## How to run

The repository includes two main notebooks.

### 1. Fit the main pipeline

```bash
jupyter notebook pipeline.ipynb
```

This notebook builds features, performs Lasso screening, constructs FixedTermOLS equations, and exports selected terms and metrics.

### 2. Run evaluation and baselines

```bash
jupyter notebook evaluate.ipynb
```

This notebook evaluates FixedTerm variants and baselines under different train sizes, including RandomForest and other regression baselines.

Scripts and reusable pipeline code are stored under [`cell_pipeline/`](cell_pipeline/).

---

## Main takeaway

This project is not just a coordinate prediction benchmark. The main contribution is an interpretable stage-dependent equation framework:

> FixedTermOLS learns a transferable structural dictionary for daughter-cell positioning, while the coefficients are stage-specific and should be calibrated with a small amount of destination-stage data.


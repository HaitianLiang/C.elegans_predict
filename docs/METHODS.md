# Methods

This document specifies the mathematical objects used by the public code and separates **reported project method choices** from **reference implementation conventions** that are not fully specified in the project reports.

## 1. Canonical mother-division event

For each embryo and mother cell, the loader finds the earliest snapshot containing both daughters and the latest earlier snapshot containing the mother. One event therefore contains:

- mother identity and 3D coordinate;
- two daughter identities and 3D coordinates;
- mother-stage cell count;
- stage block;
- embryo/source identifier.

The built-in mother/daughter map contains the 20 mother division types used by the current 4–24-cell analysis.

## 2. Six regression targets

For daughters

$$
d_1=(x_1,y_1,z_1),\qquad d_2=(x_2,y_2,z_2),
$$

define

$$
c=\frac{d_1+d_2}{2},
\qquad
h=\frac{|d_1-d_2|}{2}.
$$

The six scalar targets are

$$
(x_{\mathrm{mean}},x_{\mathrm{half}},
 y_{\mathrm{mean}},y_{\mathrm{half}},
 z_{\mathrm{mean}},z_{\mathrm{half}}).
$$

The absolute-half representation is invariant to swapping the two daughters but does not retain the per-axis daughter-label signs.

## 3. Stage-matched weighted adjacency

For mother $m$ and neighbor $j$, let $w_{mj}\ge 0$ be the stage-matched adjacency weight. The public code preserves the continuous weights.

Weighted moments use

$$
\langle f_j\rangle_w
=
\frac{\sum_j w_{mj}f_j}{\sum_j w_{mj}}.
$$

Representative FixedTerm neighborhood features are

$$
\langle x_j^k\rangle_w,
\qquad
\langle(x_j-x_m)^k\rangle_w,
\qquad
\langle(x_mx_j)^k\rangle_w,
\qquad
\langle x_mx_j^2\rangle_w.
$$

The public `FeatureBuilder` exports both FixedTerm and WAEF features. The namespaces are deliberately separated:

- `mother_*`: FixedTerm candidate library;
- `wa_*`: WAEF primitive library;
- `adjacency_degree` and `adjacency_neighbor_count`: diagnostics only.

`fixedterm_feature_columns()` returns only `mother_*` features so that WAEF primitives cannot enter FixedTerm screening accidentally.

## 4. Mother-coordinate terms

The FixedTerm mother-only library contains the polynomial/interaction families represented in the project equations:

- first-, second-, and third-order single-axis terms;
- pairwise products and squared differences;
- pairwise polynomial couplings such as $x_m^2y_m$, $x_my_m^2$, and $(x_my_m)^k$;
- the three-axis product $x_my_mz_m$;
- the higher-order three-axis couplings that recur in the archived equation family.

The public implementation uses an explicit finite list rather than an unconstrained generic polynomial expansion.

## 5. FixedTermOLS

### 5.1 Model

For a selected term set $S$, the model is no-intercept ordinary least squares:

$$
\widehat y
=
\sum_{k\in S}\beta_k\phi_k(X).
$$

### 5.2 Fixed-alpha Lasso screening

The project reports specify fixed-$\alpha$ Lasso screening. The mother-specific analysis retains Top-40 candidates; the stage-level analysis retains Top-20.

The public implementation exposes two Lasso-scaling choices:

- `--lasso-scale rms`: no-centering RMS scaling before Lasso screening;
- `--lasso-scale none`: use raw feature scales.

The project reports do not record the exact low-level scaling convention, so this option is explicit rather than hidden.

### 5.3 WAIC/R2 greedy ranking

Starting from the screened candidate set, the next term is chosen by

$$
S=
\frac12
\frac{\max(\Delta R^2,0)}{\max \Delta R^2+10^{-12}}
+
\frac12
\frac{\max(\Delta \mathrm{WAIC},0)}{\max \Delta \mathrm{WAIC}+10^{-12}}.
$$

The public code refits no-intercept OLS for every candidate at every greedy step and applies deterministic tie-breaking.

### 5.4 Public WAIC convention

The project reports specify WAIC-based selection but do not document the exact computational formula. The public function `approx_waic()` therefore uses an explicit Gaussian plug-in likelihood with residual variance estimated by MSE and a parameter-count penalty.

This is a **reference implementation convention**. It should not be described as byte-identical to the historical exploratory notebook unless that notebook is checked directly.

### 5.5 Final equation versus structural profile

Mother-specific paths are retained for at most 15 steps. The prediction equation uses the path depth with minimum training-set WAIC.

The Top-5/8/12/15 ranking profiles are kept separately for structural-dictionary comparisons. They do not automatically define the final predictive model.

## 6. Mother-specific train/test design

Each mother is split independently into 70% training and 30% held-out events. The same split is shared by all six targets for that mother.

Training events are used for:

- Lasso screening;
- greedy term ranking;
- Final-WAIC depth selection;
- OLS coefficient fitting;
- mother-by-target residual RMSE scale;
- the reference daughter-sign template used for labeled 3D reconstruction.

Held-out events are not used in these fitting steps.

## 7. Mother-specific structural contribution

For a selected term $\beta_j\phi_j$, the project report measures standardized contribution by

$$
C_j=|\beta_j|\,\mathrm{SD}(\phi_j).
$$

This avoids interpreting coefficient magnitude without accounting for feature scale.

## 8. Lineage dictionary comparison

For each mother, take the union of the Top-$k$ terms across the three half targets. For mothers $a,b$, define

$$
J(a,b)=\frac{|D_a\cap D_b|}{|D_a\cup D_b|}.
$$

The project compares mean Jaccard similarity for mother pairs with the same root family and different root families. The public code uses a one-sided permutation test on shuffled family labels and Benjamini-Hochberg correction across the four depths $k\in\{5,8,12,15\}$.

## 9. Residual layer

For mother $m$, half target $a$, and event $i$,

$$
\epsilon_{i,m,a}
=y_{i,m,a}-\widehat y_{i,m,a},
\qquad
\sigma_{m,a}=\mathrm{RMSE}_{\mathrm{train}}(m,a).
$$

The project report summarizes standardized residuals with

$$
\frac{\epsilon_{i,m,a}}{\sigma_{m,a}}
\sim t_{13}(0,1.08).
$$

The public code makes the data split explicit:

1. estimate $\sigma_{m,a}$ from training residuals only;
2. standardize training residuals with that scale;
3. standardize held-out residuals with the same scale;
4. export both tables separately;
5. optionally fit a zero-centered Student-t to either table.

The reported $t_{13}(0,1.08)$ parameters are retained as a reference result, not hard-coded as the fit produced by every rerun.

## 10. Continuous undirected division axis

For one event,

$$
v=d_1-d_2,
\qquad
u=\frac{v}{\|v\|}.
$$

Because $u\equiv -u$, the mother orientation tensor is

$$
M=\frac1n\sum_i u_i u_i^\top.
$$

Let $\lambda_1$ be its largest eigenvalue. The dominant eigenvector is the mother dominant axis and

$$
A=\frac{3\lambda_1-1}{2}
$$

is the axial strength.

The undirected angle between two axes is

$$
\theta
=
\arccos\left(
\left|
\frac{u_1^\top u_2}{\|u_1\|\|u_2\|}
\right|
\right).
$$

The event-level division-length ratio is

$$
\rho=
\frac{\|d_{1,\mathrm{pred}}-d_{2,\mathrm{pred}}\|}
     {\|d_{1,\mathrm{true}}-d_{2,\mathrm{true}}\|}.
$$

## 11. Labeled 3D daughter reconstruction

Because the half targets are absolute, the six-target representation alone cannot determine a labeled daughter vector. The public reference code estimates, for each mother and axis, the median sign of `daughter1 - daughter2` using training events only and applies that sign to held-out half predictions.

This is an explicit **public-code reconstruction convention**. The project reports state that daughter direction correspondence must be determined from training samples but do not specify the exact sign-estimation algorithm.

Undirected axis statistics are insensitive to a global daughter swap, but they are not insensitive to arbitrary independent sign flips across axes.

## 12. Stage-level half-vector organization

The stage-level half equations are reorganized as

$$
\widehat h^{(s)}
=
A_sQ
+
B_s(x_m,y_m,z_m)\mu
+
C_s(x_m,y_m,z_m)M
+
p_s(x_m,y_m,z_m),
$$

where $Q$ denotes local spread, $\mu$ neighborhood centers, $M$ higher-order neighborhood moments, and $p_s$ mother-coordinate polynomial/coupling terms.

This organization does not change predictions; it is an interpretive regrouping of selected FixedTerm terms.

## 13. Cross-stage transfer

For source stage $s$ and destination stage $t$, the public stage pipeline evaluates:

1. **direct coefficient transfer** — apply the complete source equation to destination events;
2. **structure transfer + destination refit** — retain the source selected terms and refit no-intercept OLS in the destination stage.

The off-diagonal average is reported separately from the diagonal self-stage entries.

## 14. Multi-stage shared dictionary

For every non-empty subset of the five stage blocks, the target-specific dictionary is the union of the participating stage-selected terms. The public reference implementation compares:

- `SelfStageFixedTerm`;
- `ComboSharedCoeff`;
- `ComboStageIndicator`;
- `ComboStageSpecificCoeff`.

A small Ridge penalty is used for numerical stability in the pooled linear solves. The exact ridge value is a public implementation parameter and should not be mistaken for a biological conclusion.

## 15. WAEF

WAEF uses the ten primitives

$$
(x_m,y_m,z_m,d,\Delta_x,\Delta_y,\Delta_z,Q_x,Q_y,Q_z).
$$

Within each stage, each primitive is RMS-standardized without centering. A unit-norm effective-field direction satisfies

$$
u=w^\top v^*,
\qquad
\|w\|_2=1.
$$

The response families implemented are:

$$
\widehat q_{\mathrm{mean}}=\beta u,
$$

$$
\widehat x_{\mathrm{half}}=\beta\exp(ku),
$$

$$
\widehat y_{\mathrm{half}}
=\beta\,\mathrm{logistic}(k(u+c)),
$$

$$
\widehat z_{\mathrm{half}}
=\beta\,\mathrm{logistic}(k(|u|-c)).
$$

The field direction and nonlinear parameters are optimized numerically; the outer coefficient $\beta$ is solved analytically conditional on the response shape.

The stage script can export within-stage WAEF fits and cross-stage direct/refit-outer-$\beta$ checks.

## 16. RandomForest audit

RandomForest is used as a nonparametric feature-importance audit, not as the main interpretable model. The audit is run on the FixedTerm candidate library only. Feature importance is aggregated into mother-coordinate, mother-polynomial, neighbor-moment, relative-displacement, and mother-neighbor-coupling blocks.

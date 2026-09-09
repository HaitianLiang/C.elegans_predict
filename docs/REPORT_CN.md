# 线虫空间分裂几何项目报告

## 1. 项目定位

本项目研究 *C. elegans* 早期胚胎中 mother cell 分裂后两个 daughter cell 的三维空间几何。当前项目同时保留两个统计分辨率：

- **stage-level 分析**：把事件按发育阶段聚合，研究几何结构在不同 stage 间的稳定性、结构迁移和系数迁移；
- **mother-specific 分析**：以单个 mother division type 为统计单位，在同一 mother 的重复胚胎观测中独立划分训练集和测试集，研究具体分裂事件的可预测性、结构字典、残差不确定性和连续三维分裂轴。

两种分析回答的问题不同，因此本项目不把它们的指标直接混为同一类泛化结果。stage-level 的 within-stage 指标主要用于结构描述；mother-specific 结果才对应同一 mother 内 70% 训练、30% held-out 测试的事件级评价。

本报告中的数值均来自当前项目的两份分析报告。仓库中的公开代码实现了对应的方法结构，但若要逐项复现历史数值，还需要恢复原始 fixed Lasso $\alpha$、原始 train/test split 以及完全一致的输入文件。

## 2. 数据与事件单位

mother-specific 分析覆盖 4–24 细胞阶段，共重构得到 **4056 条 mother-event observations**，包含 20 个 mother division types：

`ABa`, `ABp`, `EMS`, `P2`, `ABal`, `ABar`, `ABpl`, `ABpr`, `MS`, `E`, `C`, `ABala`, `ABalp`, `ABara`, `ABarp`, `ABpla`, `ABplp`, `ABpra`, `ABprp`, `P3`。

stage-level 分组如下。

| Stage block | Event rows | Mother types |
|---|---:|---:|
| 4-8 | 888 | 4 |
| 8-12 | 872 | 4 |
| 12-14 | 436 | 2 |
| 14-15 | 186 | 1 |
| 15-24 | 1674 | 9 |
| **Total** | **4056** | **20** |

mother-specific 分析对每个 mother 独立进行 70%/30% 随机划分，六个 target 共用同一组 mother-level train/test split。其他 mother、其他 stage 和测试事件不参与该 mother 方程的 term selection 或系数拟合。

## 3. 六目标表示

设一次分裂的两个 daughter 坐标为

$$
d_1=(x_1,y_1,z_1),\qquad d_2=(x_2,y_2,z_2).
$$

定义 daughter pair 中心

$$
c=\frac{d_1+d_2}{2}
$$

以及逐坐标绝对 half-vector

$$
h=\frac{|d_1-d_2|}{2}.
$$

因此六个回归目标为

$$
(x_{\mathrm{mean}},y_{\mathrm{mean}},z_{\mathrm{mean}},
 x_{\mathrm{half}},y_{\mathrm{half}},z_{\mathrm{half}}).
$$

`mean` 描述 daughter pair 的中心位置；`half` 描述沿三个坐标轴的分离幅度。由于 half 使用绝对值，其本身不保留 daughter label 在每个坐标轴上的符号信息，因此从六目标恢复有标签的 3D daughter 坐标时需要额外的方向约定。

## 4. 加权邻接与特征库

项目使用与 mother 所处发育阶段对应的加权邻接矩阵。矩阵元素作为连续接触权重使用，不进行二值化。

对 mother $m$ 与邻居 $j$ 的权重 $w_{mj}$，定义加权邻域矩

$$
\langle x_j^k\rangle_w
=
\frac{\sum_j w_{mj}x_j^k}{\sum_j w_{mj}},
$$

相对位移矩

$$
\langle (x_j-x_m)^k\rangle_w
=
\frac{\sum_j w_{mj}(x_j-x_m)^k}{\sum_j w_{mj}},
$$

并构造 mother-neighbor coupling，例如

$$
\langle (z_mz_j)^k\rangle_w,
\qquad
\langle x_mx_j^2\rangle_w,
\qquad
\langle y_m^2y_j\rangle_w.
$$

FixedTermOLS 的候选库由以下四类来源构成：

1. mother coordinate 及其低阶多项式、轴间耦合；
2. weighted neighbor moments；
3. weighted relative displacement；
4. mother-neighbor coupling。

WAEF 使用的 degree、weighted displacement 和 local spread primitive 在代码中使用独立 namespace，不自动进入 FixedTermOLS candidate pool。

## 5. FixedTermOLS

### 5.1 无截距模型

对 mother-specific 模型，设 mother 为 $m$、target 为 $a$，最终方程写为

$$
\widehat y_{m,a}
=
\sum_{k\in S_{m,a}}\beta_{m,a,k}\phi_k(X).
$$

stage-level 模型同样采用无截距稀疏线性方程，但以 stage 为分组单位。

### 5.2 两阶段选择

mother-specific 流程首先通过 fixed-$\alpha$ Lasso 保留 Top-40 候选项；stage-level 分析使用 Top-20 候选项。随后在候选池中逐步加入项，mother-specific 路径最多保留 15 步。

每一步对剩余候选项同时考虑 $R^2$ 改善和 WAIC 改善，综合排序规则为

$$
S=
\frac12
\frac{\max(\Delta R^2,0)}{\max\Delta R^2+10^{-12}}
+
\frac12
\frac{\max(\Delta \mathrm{WAIC},0)}{\max\Delta \mathrm{WAIC}+10^{-12}}.
$$

Final-WAIC equation 与 Top-k structural profile 承担不同作用：

- **Final-WAIC equation**：在训练集 greedy path 上选择 WAIC 最优项数，用于最终点预测；
- **Top-5/8/12/15 profile**：保留更深的有序 term dictionary，用于 mother/lineage 间结构比较。

因此，结构比较不等价于把 Top-15 的全部项强制塞入预测方程。

## 6. Mother-specific FixedTermOLS 结果

### 6.1 120 个 mother-specific 方程

20 个 mother 分别拟合 6 个 target，共得到 **120 个 Final-WAIC 方程**。每个方程的结构选择与系数估计只使用对应 mother 的训练样本。

### 6.2 Half 方程主要来自加权邻域几何

最终 60 个 half 方程中：

- weighted-adjacency terms 平均约占 **70.8%**；
- 按

$$
C_j=|\beta_j|\,\mathrm{SD}(\phi_j)
$$

计算标准化贡献时，加权邻接项约占总贡献的 **62.0%**。

Top-15 中跨 mother 高频出现的结构包括：

| Structural term | Mothers containing the term |
|---|---:|
| $\langle z_j^2\rangle_w$ | 19/20 |
| $\langle x_j^3\rangle_w$ | 17/20 |
| $\langle y_j^2\rangle_w$ | 16/20 |
| $\langle z_j^3\rangle_w$ | 16/20 |
| $\langle (z_mz_j)^3\rangle_w$ | 16/20 |

这些结果支持“多个 mother 重复调用相似邻域几何来源”的结构字典解释，但不同 mother 的最终稀疏组合和数值系数仍然不同。

### 6.3 Lineage 结构字典相似性

将 20 个 mother 分为 ABa、ABp、EMS、P2 四个 root families，对三个 half target 的 term union 计算 Jaccard，相应结果为：

| Top-k | Same-root Jaccard | Different-root Jaccard | Difference | Permutation $p$ | BH-corrected $q$ |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.1803 | 0.1740 | 0.0063 | 0.3098 | 0.3118 |
| 8 | 0.2467 | 0.2408 | 0.0059 | 0.3118 | 0.3118 |
| 12 | 0.3250 | 0.3184 | 0.0066 | 0.2746 | 0.3118 |
| 15 | 0.3918 | 0.3694 | 0.0224 | 0.0304 | 0.1216 |

Top-15 的 same-root dictionary overlap 最大，但四个 profile depth 校正后 $q\approx0.1216$。因此当前证据支持的是**探索性的 lineage-associated structural signal**，不足以把单一 term 声称为稳定的 lineage-specific marker。

### 6.4 Mean 方程与中心位置运输

三个 mean target 中，同方向一次 mother coordinate 反复进入 Final-WAIC equation：

| Target | 同轴一次项进入 Final 的比例 | 该同轴一次项平均标准化贡献 | 三个一次 mother-coordinate 合计贡献 |
|---|---:|---:|---:|
| $x_{\mathrm{mean}}$ | 60% | 15.77% | 16.91% |
| $y_{\mathrm{mean}}$ | 40% | 9.32% | 9.81% |
| $z_{\mathrm{mean}}$ | 40% | 8.60% | 10.77% |

因此 `mean` 更适合解释为 daughter pair 中心位置相对 mother absolute position 的运输和校准；`half` 更直接反映局部分裂幅度及邻域几何调节。

## 7. Mother-specific 残差概率层

对 half target 定义

$$
\epsilon_{i,m,a}
=y_{i,m,a}-\widehat y_{i,m,a},
\qquad
\sigma_{m,a}=\mathrm{RMSE}_{\mathrm{train}}(m,a).
$$

报告中的标准化 half residual 使用零中心 Student-t 形式概括：

$$
\frac{\epsilon_{i,m,a}}{\sigma_{m,a}}
\sim t_{13}(0,1.08).
$$

因此可写成

$$
Y_{i,m,a}\mid X_i
\sim
t_{13}\left(
 f^{\mathrm{FixedTerm}}_{m,a}(X_i),
 1.08\,\sigma_{m,a}
\right).
$$

三个 half 方向的标准化残差相关约为

$$
\rho_{xy}\approx0.026,
\qquad
\rho_{xz}\approx0.082,
\qquad
\rho_{yz}\approx-0.001.
$$

当前代码明确分开导出 training standardized residual 和 held-out standardized residual，并统一使用训练集估计的 $\sigma_{m,a}$，避免把残差尺度估计和 held-out 评价混在一起。

## 8. 连续三维分裂轴

### 8.1 无向轴定义

对每次分裂定义

$$
v=d_1-d_2,
\qquad
u=\frac{v}{\|v\|}.
$$

由于 $u$ 和 $-u$ 表示同一条无向分裂轴，对同一 mother 的事件构造 orientation tensor

$$
M=\frac1n\sum_{i=1}^n u_i u_i^\top.
$$

最大特征向量定义 dominant division axis。若最大特征值为 $\lambda_1$，定义 axial strength

$$
A=\frac{3\lambda_1-1}{2}\in[0,1].
$$

真实轴与预测轴的夹角为

$$
\theta
=
\arccos\left(
|u_{\mathrm{true}}^\top u_{\mathrm{pred}}|
\right)
\in[0^\circ,90^\circ],
$$

分裂长度比为

$$
\rho
=
\frac{\|v_{\mathrm{pred}}\|}{\|v_{\mathrm{true}}\|}.
$$

### 8.2 真实主轴稳定性

4056 条真实事件中，20 个 mother 的 axial strength 中位数为 **0.955**，最低约为 **0.864**；各 mother 的 event-level axis 相对自身 dominant axis 的中位偏离角总体中位数约为 **7.84°**。

这表明 dominant division axis 是高度重复的 mother-specific 几何量。

### 8.3 Held-out 轴方向与分裂长度

在 **1224 条 held-out 事件**上：

- event-level 轴角误差中位数：**7.44°**；
- IQR：**4.66°–11.38°**；
- $68.7\%$ 不超过 10°；
- $86.6\%$ 不超过 15°；
- $94.4\%$ 不超过 20°；
- division-length ratio 中位数：**0.997**；
- IQR：**0.927–1.073**；
- $90.5\%$ 位于 $[0.8,1.2]$ 内。

把同一 mother 的 held-out events 汇总后，20 个 mother 的 true/predicted dominant-axis angle 中位数为 **1.39°**，最大为 **5.08°**，其中 **19/20** 小于 5°。

### 8.4 预测轴云过于集中

held-out true axis 的 axial strength 中位数为 **0.955**，predicted axis 为 **0.991**；event 相对主轴的 mother-level 中位偏离角总体中位数，真实值约为 **7.68°**，预测值约为 **3.23°**。

因此模型能够稳定恢复 mother 的典型主轴，但会低估 embryo-to-embryo 的方向波动。这个现象与 residual layer 中仍然存在个体不确定性的结论一致。

## 9. Stage-level FixedTermOLS

### 9.1 Within-stage 描述性拟合

下表是每个 stage 自己的数据拟合并在同一 stage 上评价的 descriptive fit；它不是 cross-stage 迁移结果，也不是 mother-specific held-out 结果。

| Stage | Events | Mean $R^2$ | Half $R^2$ | All $R^2$ |
|---|---:|---:|---:|---:|
| 4-8 | 888 | 0.981 | 0.784 | 0.883 |
| 8-12 | 872 | 0.986 | 0.806 | 0.896 |
| 12-14 | 436 | 0.936 | 0.156 | 0.546 |
| 14-15 | 186 | 0.248 | 0.092 | 0.170 |
| 15-24 | 1674 | 0.980 | 0.727 | 0.853 |

4-8、8-12 和 15-24 的 stage-level 方程较稳定；12-14 和 14-15 的 half target 明显更困难。14-15 只有一个 mother type，应当作为短过渡窗口单独解释，而不作为一般规律的主要来源。

### 9.2 Half-vector 结构化表达

三个 half 方程可以统一写成

$$
\widehat h^{(s)}
=
A_sQ
+
B_s(x_m,y_m,z_m)\mu
+
C_s(x_m,y_m,z_m)M
+
p_s(x_m,y_m,z_m).
$$

其中：

- $Q$：local spread；
- $\mu$：neighbor centroid；
- $M$：neighbor higher moments；
- $p_s$：mother-axis polynomial / coupling。

按每一项的实际样本贡献强度 $\mathrm{SD}(\beta\phi(X))$ 归一化后，half-vector 结构块占比如下。

| Stage | Mother coupling | Mother-neighbor moment | Mother polynomial | Neighbor moment | Relative spread |
|---|---:|---:|---:|---:|---:|
| 4-8 | 0.168 | 0.143 | 0.130 | 0.364 | 0.195 |
| 8-12 | 0.257 | 0.141 | 0.282 | 0.171 | 0.149 |
| 12-14 | 0.101 | 0.015 | 0.175 | 0.550 | 0.159 |
| 14-15 | 0.584 | 0.000 | 0.280 | 0.076 | 0.060 |
| 15-24 | 0.426 | 0.182 | 0.193 | 0.157 | 0.042 |

总体上，早期 half-vector 更受邻域矩和相对扩散控制；12-14 的邻域高阶矩作用最突出；后期 mother coupling 增强。

### 9.3 $Q_y\rightarrow x_{\mathrm{half}}$ 条件性 motif

定义

$$
Q_y=\langle(y_j-y_m)^2\rangle_N.
$$

在多个非过渡阶段，$Q_y$ 反复进入 $x_{\mathrm{half}}$ 方程：

| Stage | Pearson $r$ | Spearman $r$ | Univariate $R^2$ | Selected by FixedTerm | Coefficient |
|---|---:|---:|---:|---|---:|
| 4-8 | -0.094 | 0.051 | 0.009 | yes | 0.1089 |
| 8-12 | 0.405 | 0.288 | 0.164 | yes | 0.1205 |
| 12-14 | 0.437 | 0.399 | 0.191 | yes | 0.0312 |
| 14-15 | 0.148 | 0.147 | 0.022 | no | — |
| 15-24 | 0.581 | 0.617 | 0.338 | yes | 0.0287 |

该结果应解释为多变量方程中重复出现的 **conditional geometric motif**，而不是所有 stage 都成立的简单单变量因果律。

## 10. 跨 stage 迁移

跨 stage 迁移用于区分“系数是否稳定”和“结构是否稳定”。

- **Direct coefficient transfer**：把 source stage 的结构和数值系数一起搬到 destination；
- **Structure transfer + destination refit**：只迁移 selected term set，在 destination stage 重新拟合系数。

reported off-diagonal summary 为：

| Transfer mode | Mean $R^2$ | Half $R^2$ | All $R^2$ | All-target median $R^2$ |
|---|---:|---:|---:|---:|
| Direct coefficient transfer | -7.263 | -137.451 | -72.357 | -8.118 |
| Structure transfer + destination refit | 0.784 | 0.244 | 0.514 | 0.571 |

这一结果支持 stage-level 分析的核心结论：**跨阶段更稳定的是几何结构字典，而不是固定的数值系数。**

## 11. 多阶段结构字典共享

5 个 stage 的所有非空 subset 共计

$$
\binom51+\binom52+\binom53+\binom54+\binom55=31
$$

个组合。组合内比较：

- SelfStageFixedTerm；
- ComboSharedCoeff；
- ComboStageIndicator；
- ComboStageSpecificCoeff。

按组合大小平均的结果如下。

| Combo size | Model | All $R^2$ | Mean $R^2$ | Half $R^2$ | Position RMSE | $\Delta$ all $R^2$ vs self |
|---:|---|---:|---:|---:|---:|---:|
| 1 | SelfStageFixedTerm | 0.628 | 0.798 | 0.459 | 1.571 | 0.000 |
| 2 | ComboSharedCoeff | 0.812 | 0.977 | 0.647 | 1.819 | -0.036 |
| 2 | ComboStageIndicator | 0.838 | 0.979 | 0.698 | 1.691 | -0.009 |
| 2 | ComboStageSpecificCoeff | 0.853 | 0.983 | 0.723 | 1.559 | +0.006 |
| 3 | ComboSharedCoeff | 0.834 | 0.972 | 0.696 | 1.968 | -0.051 |
| 3 | ComboStageIndicator | 0.866 | 0.977 | 0.755 | 1.790 | -0.019 |
| 3 | ComboStageSpecificCoeff | 0.890 | 0.985 | 0.796 | 1.555 | +0.006 |
| 4 | ComboSharedCoeff | 0.824 | 0.967 | 0.681 | 2.098 | -0.069 |
| 4 | ComboStageIndicator | 0.865 | 0.973 | 0.757 | 1.884 | -0.028 |
| 4 | ComboStageSpecificCoeff | 0.899 | 0.985 | 0.813 | 1.553 | +0.006 |
| 5 | ComboSharedCoeff | 0.810 | 0.960 | 0.660 | 2.254 | -0.087 |
| 5 | ComboStageIndicator | 0.859 | 0.969 | 0.750 | 1.980 | -0.038 |
| 5 | ComboStageSpecificCoeff | 0.904 | 0.985 | 0.822 | 1.549 | +0.006 |

强制多个 stage 共用同一套系数会降低性能；共享结构字典并保留 stage-specific coefficients 则能稳定获得小幅收益，尤其在 3D position RMSE 上更明显。

## 12. WAEF Regression

Weighted-Adjacency Effective-Field Regression 使用 primitive

$$
(x_m,y_m,z_m,d,\Delta_x,\Delta_y,\Delta_z,Q_x,Q_y,Q_z).
$$

stage 内对 primitive 进行不中心化 RMS 标准化，构造单位范数有效场

$$
u=\sum_r w_rv_r^*,
\qquad
\sum_r w_r^2=1.
$$

mean target 使用线性有效场；三个 half target 分别使用指数或 Sigmoid 类型响应。

### 12.1 Within-stage 对比

| Stage | WAEF all $R^2$ | FixedTermOLS all $R^2$ | WAEF position RMSE | FixedTermOLS position RMSE |
|---|---:|---:|---:|---:|
| 4-8 | 0.884 | 0.883 | 1.345 | 1.339 |
| 8-12 | 0.899 | 0.896 | 1.681 | 1.584 |
| 12-14 | 0.561 | 0.546 | 1.526 | 1.398 |
| 14-15 | 0.184 | 0.170 | 1.333 | 1.333 |
| 15-24 | 0.819 | 0.853 | 2.117 | 1.791 |

WAEF 在若干早中期 stage 内可以接近 FixedTermOLS，但在 15-24 的压缩损失更明显。因此它适合作为低维结构压缩和动力学解释工具，而不是 FixedTermOLS 的替代主模型。

### 12.2 Cross-stage WAEF

| Mode | All $R^2$ | Mean $R^2$ | Half $R^2$ | MSE |
|---|---:|---:|---:|---:|
| Direct transfer of field and coefficient | -12.356 | -19.499 | -5.212 | 17.836 |
| Transfer field, refit outer coefficient | -2.090 | -3.175 | -1.006 | 10.233 |

说明 source stage 学到的有效场方向本身具有明显 stage dependence。

## 13. RandomForest 审计

RandomForest 在项目中只作为非线性 feature audit，不作为主预测模型。RF top-20 与 FixedTerm selected terms 的重合为：

| Stage | FixedTerm selected | RF top-20 | Overlap | Overlap ratio |
|---|---:|---:|---:|---:|
| 4-8 | 21 | 20 | 7 | 0.333 |
| 8-12 | 30 | 20 | 12 | 0.400 |
| 12-14 | 26 | 20 | 11 | 0.423 |
| 14-15 | 22 | 20 | 13 | 0.591 |
| 15-24 | 28 | 20 | 8 | 0.286 |

该结果说明母细胞多项式、邻域矩、相对位移和 mother-neighbor coupling 同样受到非线性模型重视。但 RF importance 是 stage-level multi-output importance，不能单独证明某个 target-specific 通道，例如 $Q_y\rightarrow x_{\mathrm{half}}$。

## 14. 少样本系数标定

stage-level 结构迁移还进行了少样本标定实验：在 destination stage 抽取少量事件，只重新拟合已有结构的系数，不重新做 term selection。

三个代表训练规模的全局结果如下。

### $n=20$

| Model | All $R^2$ | Position RMSE |
|---|---:|---:|
| RF quick | 0.458 | 4.202 |
| Prev-stage FixedTerm | 0.361 | 2.485 |
| Same-stage FixedTerm | 0.339 | 2.224 |
| 15-24 FixedTerm | 0.323 | 2.194 |
| Union FixedTerm | 0.108 | 2.521 |
| Full ridge | -0.082 | 3.014 |
| Mother copy | -0.127 | 3.328 |
| Full linear | -3724.407 | 54.010 |

### $n=40$

| Model | All $R^2$ | Position RMSE |
|---|---:|---:|
| Same-stage FixedTerm | 0.590 | 1.782 |
| RF quick | 0.536 | 2.861 |
| Union FixedTerm | 0.511 | 1.831 |
| Prev-stage FixedTerm | 0.465 | 2.195 |
| 15-24 FixedTerm | 0.445 | 1.946 |
| Full ridge | 0.341 | 2.057 |
| Mother copy | -0.091 | 3.307 |
| Full linear | -88.211 | 17.988 |

### $n=80$

| Model | All $R^2$ | Position RMSE |
|---|---:|---:|
| Same-stage FixedTerm | 0.612 | 1.663 |
| RF quick | 0.609 | 2.005 |
| Union FixedTerm | 0.595 | 1.657 |
| Full ridge | 0.537 | 1.768 |
| Prev-stage FixedTerm | 0.510 | 2.088 |
| 15-24 FixedTerm | 0.482 | 1.855 |
| Mother copy | -0.062 | 3.295 |
| Full linear | -6.763 | 4.651 |

结论不是“某一个模型在所有样本数都最优”，而是：在小样本环境下，稀疏结构先验通常比直接拟合完整特征库更稳定；随着样本数增加，full ridge 会逐渐追近。RF 在部分坐标 $R^2$ 上具有竞争力，但 3D position error 并不总是同样稳定。

## 15. 两个分析分辨率如何统一

当前项目可以用三个层次概括：

1. **Stage-level**：说明不同发育阶段可以重复使用相似几何信息来源，但数值强度是阶段化的；
2. **Mother-specific**：说明具体 mother 的分裂几何具有高度重复的 dominant axis，同时 half 方程主要由加权邻域结构解释；
3. **Residual layer**：说明即使典型结构和主轴可以被恢复，同一 mother 不同 embryo 之间仍有异方差和轻度重尾的个体偏差。

因此目前最稳妥的项目结论不是“所有 stage 共享一条固定方程”，而是：

> **几何结构字典可以在不同尺度上重复出现，但具体数值系数和个体波动具有明显的 stage/mother dependence。**

## 16. 适用范围与限制

- 加权邻接矩阵描述 stage-matched contact structure，不等价于每个 embryo 的动态 contact area；
- half target 是绝对半间距，有标签的 3D daughter reconstruction 需要额外的训练期方向约定；
- 14-15 是特殊短窗口，样本数和 mother type 都较少，不宜从中提取普适结构规律；
- lineage Top-15 Jaccard 在多 depth 校正后未达到传统显著性标准；
- predicted axis cloud 比 true axis cloud 更集中，说明模型低估 individual embryo 的方向离散度；
- 当前公开实现中的 Lasso scaling 和 WAIC 数值例程是显式 reference convention；若要逐项复现历史方程，需要恢复原始 archive 中的具体实现与配置。

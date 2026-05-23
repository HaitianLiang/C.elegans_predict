
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


TARGET_NAMES = [
    "(x1+x2)/2",
    "(x1-x2).abs/2",
    "(y1+y2)/2",
    "(y1-y2).abs/2",
    "(z1+z2)/2",
    "(z1-z2).abs/2",
]


def align_feature_and_target_tables(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    key: str = "sample_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if key not in features_df.columns or key not in targets_df.columns:
        raise KeyError("features/targets 都必须包含 {}".format(key))

    feat = features_df.copy()
    targ = targets_df.copy()

    if feat[key].duplicated().any():
        raise ValueError("features 表中 sample_id 不唯一")
    if targ[key].duplicated().any():
        raise ValueError("targets 表中 sample_id 不唯一")

    common = sorted(set(feat[key]) & set(targ[key]))
    if not common:
        raise ValueError("features 与 targets 没有共同 sample_id")

    feat = feat[feat[key].isin(common)].sort_values(key).reset_index(drop=True)
    targ = targ[targ[key].isin(common)].sort_values(key).reset_index(drop=True)

    if not feat[key].equals(targ[key]):
        mismatch = pd.DataFrame({
            "feature_id": feat[key],
            "target_id": targ[key],
        })
        raise ValueError("按 sample_id 对齐失败，请检查输入数据。\n" + mismatch.head().to_string(index=False))

    return feat, targ


def l1_normalize_matrix(data: np.ndarray, target_sum: int) -> Tuple[np.ndarray, np.ndarray]:
    col_abs_sum = np.sum(np.abs(data), axis=0)
    col_abs_sum[col_abs_sum == 0] = 1.0
    normalized = (data / col_abs_sum) * float(target_sum)
    return normalized, col_abs_sum


def fit_initial_lasso(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    cv: int = 5,
    random_state: Optional[int] = None,
) -> Dict:
    X_mat = X_df.values.astype(float)
    y_mat = y_df.values.astype(float)
    L = X_mat.shape[0]

    X_norm, X_abs_sum_before = l1_normalize_matrix(X_mat, L)
    y_norm, y_abs_sum_before = l1_normalize_matrix(y_mat, L)

    X_norm_df = pd.DataFrame(X_norm, columns=X_df.columns)
    y_norm_df = pd.DataFrame(y_norm, columns=y_df.columns)

    models = []
    preds = []
    alphas = []
    n_samples = X_norm_df.shape[0]
    eff_cv = min(cv, n_samples)
    for i in range(y_norm.shape[1]):
        if eff_cv >= 2:
            reg = LassoCV(
                cv=eff_cv,
                fit_intercept=False,
                n_jobs=-1,
                max_iter=10000,
                random_state=random_state,
            ).fit(X_norm_df.values, y_norm[:, i])
            alpha = float(reg.alpha_)
        else:
            reg = LinearRegression(fit_intercept=False).fit(X_norm_df.values, y_norm[:, i])
            alpha = 0.0
        models.append(reg)
        preds.append(reg.predict(X_norm_df.values))
        alphas.append(alpha)

    pred_mat = np.column_stack(preds)

    adjusted_coef_list = []
    X_cols = X_df.columns.tolist()
    for task_idx in range(y_norm.shape[1]):
        coef_task = models[task_idx].coef_
        adjusted = []
        for feat_idx, coef in enumerate(coef_task):
            adjusted_coef = coef * (y_abs_sum_before[task_idx] / X_abs_sum_before[feat_idx])
            adjusted.append(adjusted_coef)
        adjusted_coef_list.append(pd.Series(adjusted, index=X_cols, name=y_df.columns[task_idx]))

    return {
        "X_normalized": X_norm_df,
        "y_normalized": y_norm_df,
        "models": models,
        "y_pred": pd.DataFrame(pred_mat, columns=y_df.columns),
        "alphas": alphas,
        "X_abs_sum_before": X_abs_sum_before,
        "y_abs_sum_before": y_abs_sum_before,
        "adjusted_coef_list": adjusted_coef_list,
    }


def select_top_features(
    adjusted_coef_list: List[pd.Series],
    top_k: int = 20,
    manual_indices_per_task: Optional[Dict[int, List[int]]] = None,
) -> Dict[int, List[str]]:
    feature_orders: Dict[int, List[str]] = {}

    for task_idx, coef_series in enumerate(adjusted_coef_list):
        ranked = coef_series.abs().sort_values(ascending=False).index.tolist()[:top_k]
        if manual_indices_per_task and task_idx in manual_indices_per_task:
            indices = manual_indices_per_task[task_idx]
            if set(indices) != set(range(1, len(ranked) + 1)):
                raise ValueError("task {} 的手动排序索引不合法".format(task_idx))
            ranked = [ranked[i - 1] for i in indices]
        feature_orders[task_idx] = ranked
    return feature_orders


def compute_waic_from_residuals(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
    mse = mean_squared_error(y_true, y_pred)
    mse = max(float(mse), 1e-12)
    log_lik = -0.5 * (np.log(2 * np.pi * mse) + (y_true - y_pred) ** 2 / mse)
    lppd = np.sum(log_lik)
    return float(-2 * (lppd - n_params))


def _fit_regression(
    X_mat: np.ndarray,
    y_vec: np.ndarray,
    use_lasso: bool = True,
    cv: int = 5,
    random_state: Optional[int] = None,
):
    eff_cv = min(cv, X_mat.shape[0])
    if use_lasso and eff_cv >= 2:
        reg = LassoCV(
            cv=eff_cv,
            fit_intercept=False,
            n_jobs=-1,
            max_iter=10000,
            random_state=random_state,
        ).fit(X_mat, y_vec)
    else:
        reg = LinearRegression(fit_intercept=False).fit(X_mat, y_vec)

    y_pred = reg.predict(X_mat)
    r2 = r2_score(y_vec, y_pred)
    mse = mean_squared_error(y_vec, y_pred)
    waic = compute_waic_from_residuals(y_vec, y_pred, n_params=X_mat.shape[1])
    return reg, y_pred, float(r2), float(mse), float(waic)


def evaluate_topk_prefix_metrics(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    feature_orders: Dict[int, List[str]],
    max_k: int = 20,
    use_lasso: bool = True,
    cv: int = 5,
    random_state: Optional[int] = None,
) -> Dict[int, pd.DataFrame]:
    results: Dict[int, pd.DataFrame] = {}
    for task_idx, feature_order in feature_orders.items():
        y_task = y_df.iloc[:, task_idx].values.astype(float)
        rows = []
        for k in range(1, min(max_k, len(feature_order)) + 1):
            feats = feature_order[:k]
            X_k = X_df[feats].values.astype(float)

            if X_k.shape[0] != len(y_task):
                raise ValueError(
                    "task {} 在 k={} 时样本数不匹配: X={}, y={}".format(
                        task_idx, k, X_k.shape[0], len(y_task)
                    )
                )

            _, _, r2, mse, waic = _fit_regression(
                X_k, y_task, use_lasso=use_lasso, cv=cv, random_state=random_state
            )

            rows.append(
                {
                    "task_idx": task_idx,
                    "target_name": TARGET_NAMES[task_idx],
                    "k": k,
                    "features": ",".join(feats),
                    "R2": r2,
                    "MSE": mse,
                    "WAIC": waic,
                }
            )
        results[task_idx] = pd.DataFrame(rows)
    return results


def plot_feature_metrics_single_task(
    Score_list,
    MSE_list,
    WAIC_list,
    imp_coef_list=None,
    target_names=None,
    save_dir: str = ".",
    prefix: str = "feature_metrics_task",
    star_positions: Optional[Dict[int, int]] = None,
):
    Score_list = np.asarray(Score_list)
    MSE_list = np.asarray(MSE_list)
    WAIC_list = np.asarray(WAIC_list)

    n_feature_exp = Score_list.shape[0]
    n_targets = Score_list.shape[1]
    Index = np.arange(1, n_feature_exp + 1)

    if target_names is None:
        target_names = ["Target_{}".format(i + 1) for i in range(n_targets)]
    else:
        assert len(target_names) == n_targets, "目标名称数量不匹配"

    os.makedirs(save_dir, exist_ok=True)

    for task_idx in range(n_targets):
        colors = ["blue", "red", "green"]
        markers = ["o", "s", "^"]

        plt.rcParams["font.sans-serif"] = ["Arial", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig = plt.figure(figsize=(12, 10))
        fig.suptitle(
            "Feature Count vs. Model Metrics - {} (Single-Task {})".format(
                target_names[task_idx],
                "Lasso" if "lasso" in prefix.lower() else "OLS",
            ),
            fontsize=14,
            y=0.95,
        )

        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(
            Index, Score_list[:, task_idx],
            marker=markers[0], linestyle="-", color=colors[0],
            linewidth=2, markersize=6, label="R² Score"
        )
        ax1.set_xlabel("Number of Features (Top-K)", fontsize=11)
        ax1.set_ylabel("R² Score", fontsize=11)
        ax1.set_xlim(0.5, n_feature_exp + 0.5)
        ax1.set_ylim(
            max(0, float(np.min(Score_list[:, task_idx])) - 0.05),
            min(1.05, float(np.max(Score_list[:, task_idx])) + 0.05)
        )
        ax1.legend(loc="lower right", fontsize=10)
        ax1.grid(alpha=0.3)
        ax1.set_title("R² Score: Higher = Better", fontsize=12, pad=15)

        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(
            Index, MSE_list[:, task_idx],
            marker=markers[1], linestyle="-", color=colors[1],
            linewidth=2, markersize=6, label="MSE"
        )
        ax2.set_xlabel("Number of Features", fontsize=11)
        ax2.set_ylabel("MSE", fontsize=11)
        ax2.set_xlim(0.5, n_feature_exp + 0.5)
        ax2.set_ylim(
            max(0, float(np.min(MSE_list[:, task_idx])) - 0.01),
            float(np.max(MSE_list[:, task_idx])) + 0.01
        )
        ax2.legend(loc="upper right", fontsize=10)
        ax2.grid(alpha=0.3)
        ax2.set_title("MSE: Lower = Better", fontsize=12, pad=15)

        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(
            Index, WAIC_list[:, task_idx],
            marker=markers[2], linestyle="-", color=colors[2],
            linewidth=2, markersize=6, label="WAIC"
        )
        ax3.set_xlabel("Number of Features", fontsize=11)
        ax3.set_ylabel("WAIC Value", fontsize=11)
        ax3.set_xlim(0.5, n_feature_exp + 0.5)
        ax3.set_ylim(
            float(np.min(WAIC_list[:, task_idx])) - 10,
            float(np.max(WAIC_list[:, task_idx])) + 10
        )
        ax3.legend(loc="upper right", fontsize=10)
        ax3.grid(alpha=0.3)
        ax3.set_title("WAIC: Lower = Better", fontsize=12, pad=15)

        star_k = None
        if star_positions is not None:
            star_k = star_positions.get(task_idx)
        if star_k is not None and 1 <= int(star_k) <= n_feature_exp:
            star_idx = int(star_k) - 1
            star_x = Index[star_idx]
            if not np.isnan(Score_list[star_idx, task_idx]):
                ax1.scatter([star_x], [Score_list[star_idx, task_idx]], marker='*', s=220, c='gold', edgecolors='black', zorder=5)
            if not np.isnan(MSE_list[star_idx, task_idx]):
                ax2.scatter([star_x], [MSE_list[star_idx, task_idx]], marker='*', s=220, c='gold', edgecolors='black', zorder=5)
            if not np.isnan(WAIC_list[star_idx, task_idx]):
                ax3.scatter([star_x], [WAIC_list[star_idx, task_idx]], marker='*', s=220, c='gold', edgecolors='black', zorder=5)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        save_path = os.path.join(save_dir, "{}_{}.png".format(prefix, task_idx + 1))
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def metrics_dict_to_matrices(
    metric_results: Dict[int, pd.DataFrame],
    n_targets: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_len = 0
    for _, df in metric_results.items():
        max_len = max(max_len, len(df))

    if max_len == 0:
        return (
            np.zeros((0, n_targets), dtype=float),
            np.zeros((0, n_targets), dtype=float),
            np.zeros((0, n_targets), dtype=float),
        )

    score = np.full((max_len, n_targets), np.nan, dtype=float)
    mse = np.full((max_len, n_targets), np.nan, dtype=float)
    waic = np.full((max_len, n_targets), np.nan, dtype=float)

    for task_idx, df in metric_results.items():
        vals_r2 = df["R2"].values.astype(float)
        vals_mse = df["MSE"].values.astype(float)
        vals_waic = df["WAIC"].values.astype(float)
        score[:len(vals_r2), task_idx] = vals_r2
        mse[:len(vals_mse), task_idx] = vals_mse
        waic[:len(vals_waic), task_idx] = vals_waic

        # 用最后一个值填充，便于和 ipynb 一样画连续图
        if len(vals_r2) < max_len and len(vals_r2) > 0:
            score[len(vals_r2):, task_idx] = vals_r2[-1]
            mse[len(vals_mse):, task_idx] = vals_mse[-1]
            waic[len(vals_waic):, task_idx] = vals_waic[-1]

    return score, mse, waic


def evaluate_stepwise_metrics(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    feature_orders: Dict[int, List[str]],
    max_k: int = 20,
    use_lasso: bool = True,
    cv: int = 5,
    random_state: Optional[int] = None,
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, Dict]]:
    """
    两阶段贪心前向选择：

    阶段1（R2阶段）：
    - 在未入选特征中，贪心选择 delta_R2 最大的特征
    - 若最佳 delta_R2 >= 0.001，继续留在 R2 阶段
    - 若最佳 delta_R2 < 0.001，切换到 WAIC 阶段

    阶段2（WAIC阶段）：
    - 在未入选特征中，重新贪心选择 delta_WAIC 最大的特征
    - delta_WAIC = current_waic - new_waic（越大越好）
    - 若最佳 delta_WAIC >= 10，继续
    - 若最佳 delta_WAIC < 10，停止
    """
    results: Dict[int, pd.DataFrame] = {}
    models_by_task: Dict[int, Dict] = {}

    stop_delta_r2 = 0.001
    stop_delta_waic = 20.0

    for task_idx, feature_order in feature_orders.items():
        y_task = y_df.iloc[:, task_idx].values.astype(float)

        candidate_features = list(feature_order[: min(max_k, len(feature_order))])
        selected_features: List[str] = []
        remaining_features = candidate_features.copy()

        rows = []
        best_model = None
        best_k = 0
        best_r2 = -np.inf
        best_mse = None
        best_waic = None

        current_r2 = None
        current_mse = None
        current_waic = None
        phase = "r2"

        while len(remaining_features) > 0:
            # 第一轮：直接选单特征 R2 最好的
            if current_r2 is None:
                step_best_feat = None
                step_best_reg = None
                step_best_r2 = -np.inf
                step_best_mse = None
                step_best_waic = None

                for feat in remaining_features:
                    X_trial = X_df[[feat]].values.astype(float)
                    if X_trial.shape[0] != len(y_task):
                        raise ValueError(
                            "task {} 在尝试首个特征 {} 时样本数不匹配: X={}, y={}".format(
                                task_idx, feat, X_trial.shape[0], len(y_task)
                            )
                        )

                    reg, _, r2, mse, waic = _fit_regression(
                        X_trial, y_task, use_lasso=use_lasso, cv=cv, random_state=random_state
                    )
                    if r2 > step_best_r2:
                        step_best_feat = feat
                        step_best_reg = reg
                        step_best_r2 = r2
                        step_best_mse = mse
                        step_best_waic = waic

                if step_best_feat is None:
                    break

                selected_features.append(step_best_feat)
                remaining_features.remove(step_best_feat)
                current_r2 = step_best_r2
                current_mse = step_best_mse
                current_waic = step_best_waic

                rows.append(
                    {
                        "task_idx": task_idx,
                        "target_name": TARGET_NAMES[task_idx],
                        "phase": "r2",
                        "k": 1,
                        "new_feature": step_best_feat,
                        "features": ",".join(selected_features),
                        "delta_R2": np.nan,
                        "delta_WAIC": np.nan,
                        "R2": current_r2,
                        "MSE": current_mse,
                        "WAIC": current_waic,
                    }
                )

                best_model = step_best_reg
                best_k = 1
                best_r2 = current_r2
                best_mse = current_mse
                best_waic = current_waic
                continue

            # 阶段1：按 delta_R2 贪心
            if phase == "r2":
                step_best_feat = None
                step_best_reg = None
                step_best_r2 = -np.inf
                step_best_mse = None
                step_best_waic = None
                step_best_delta_r2 = None
                step_best_delta_waic = None

                for feat in remaining_features:
                    trial_features = selected_features + [feat]
                    X_trial = X_df[trial_features].values.astype(float)

                    if X_trial.shape[0] != len(y_task):
                        raise ValueError(
                            "task {} 在尝试特征 {} 时样本数不匹配: X={}, y={}".format(
                                task_idx, feat, X_trial.shape[0], len(y_task)
                            )
                        )

                    reg, _, r2, mse, waic = _fit_regression(
                        X_trial, y_task, use_lasso=use_lasso, cv=cv, random_state=random_state
                    )
                    delta_r2 = r2 - current_r2
                    delta_waic = current_waic - waic

                    if step_best_delta_r2 is None:
                        better = True
                    else:
                        better = (delta_r2 > step_best_delta_r2) or (
                            np.isclose(delta_r2, step_best_delta_r2) and r2 > step_best_r2
                        )

                    if better:
                        step_best_feat = feat
                        step_best_reg = reg
                        step_best_r2 = r2
                        step_best_mse = mse
                        step_best_waic = waic
                        step_best_delta_r2 = delta_r2
                        step_best_delta_waic = delta_waic

                if step_best_feat is None:
                    break

                if step_best_delta_r2 is not None and step_best_delta_r2 >= stop_delta_r2:
                    selected_features.append(step_best_feat)
                    remaining_features.remove(step_best_feat)
                    current_r2 = step_best_r2
                    current_mse = step_best_mse
                    current_waic = step_best_waic

                    k = len(selected_features)
                    rows.append(
                        {
                            "task_idx": task_idx,
                            "target_name": TARGET_NAMES[task_idx],
                            "phase": "r2",
                            "k": k,
                            "new_feature": step_best_feat,
                            "features": ",".join(selected_features),
                            "delta_R2": step_best_delta_r2,
                            "delta_WAIC": step_best_delta_waic,
                            "R2": current_r2,
                            "MSE": current_mse,
                            "WAIC": current_waic,
                        }
                    )

                    best_model = step_best_reg
                    best_k = k
                    best_r2 = current_r2
                    best_mse = current_mse
                    best_waic = current_waic
                    continue

                phase = "waic"

            # 阶段2：按 delta_WAIC 贪心
            if phase == "waic":
                step_best_feat = None
                step_best_reg = None
                step_best_r2 = -np.inf
                step_best_mse = None
                step_best_waic = None
                step_best_delta_r2 = None
                step_best_delta_waic = None

                for feat in remaining_features:
                    trial_features = selected_features + [feat]
                    X_trial = X_df[trial_features].values.astype(float)

                    if X_trial.shape[0] != len(y_task):
                        raise ValueError(
                            "task {} 在 WAIC 阶段尝试特征 {} 时样本数不匹配: X={}, y={}".format(
                                task_idx, feat, X_trial.shape[0], len(y_task)
                            )
                        )

                    reg, _, r2, mse, waic = _fit_regression(
                        X_trial, y_task, use_lasso=use_lasso, cv=cv, random_state=random_state
                    )
                    delta_r2 = r2 - current_r2
                    delta_waic = current_waic - waic

                    if step_best_delta_waic is None:
                        better = True
                    else:
                        better = (delta_waic > step_best_delta_waic) or (
                            np.isclose(delta_waic, step_best_delta_waic) and waic < step_best_waic
                        )

                    if better:
                        step_best_feat = feat
                        step_best_reg = reg
                        step_best_r2 = r2
                        step_best_mse = mse
                        step_best_waic = waic
                        step_best_delta_r2 = delta_r2
                        step_best_delta_waic = delta_waic

                if step_best_feat is None:
                    break

                if step_best_delta_waic is None or step_best_delta_waic < stop_delta_waic:
                    break

                selected_features.append(step_best_feat)
                remaining_features.remove(step_best_feat)
                current_r2 = step_best_r2
                current_mse = step_best_mse
                current_waic = step_best_waic

                k = len(selected_features)
                rows.append(
                    {
                        "task_idx": task_idx,
                        "target_name": TARGET_NAMES[task_idx],
                        "phase": "waic",
                        "k": k,
                        "new_feature": step_best_feat,
                        "features": ",".join(selected_features),
                        "delta_R2": step_best_delta_r2,
                        "delta_WAIC": step_best_delta_waic,
                        "R2": current_r2,
                        "MSE": current_mse,
                        "WAIC": current_waic,
                    }
                )

                best_model = step_best_reg
                best_k = k
                best_r2 = current_r2
                best_mse = current_mse
                best_waic = current_waic
                continue

        final_feature_order = selected_features + remaining_features
        selection_stop_feature = selected_features[-1] if selected_features else None

        results[task_idx] = pd.DataFrame(rows)
        models_by_task[task_idx] = {
            "model": best_model,
            "best_k": best_k,
            "best_r2": best_r2,
            "best_mse": best_mse,
            "best_waic": best_waic,
            "features": selected_features.copy(),
            "final_feature_order": final_feature_order,
            "selection_stop_feature": selection_stop_feature,
        }

    return results, models_by_task


def fit_no_intercept_equations(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    best_models: Dict[int, Dict],
) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    summary_rows = []
    coef_tables: Dict[int, pd.DataFrame] = {}

    for task_idx in range(y_df.shape[1]):
        info = best_models.get(task_idx, {})
        feats = info.get("features", []) or []
        target_name = TARGET_NAMES[task_idx]
        y_task = y_df.iloc[:, task_idx].values.astype(float)

        if len(feats) == 0:
            summary_rows.append({
                "task_idx": task_idx,
                "target_name": target_name,
                "equation_r2": np.nan,
                "equation_mse": np.nan,
                "equation_waic": np.nan,
                "equation": "",
            })
            coef_tables[task_idx] = pd.DataFrame(columns=["feature_name", "coefficient"])
            continue

        X_task = X_df[feats].values.astype(float)
        reg = LinearRegression(fit_intercept=False).fit(X_task, y_task)
        y_pred = reg.predict(X_task)

        r2 = float(r2_score(y_task, y_pred))
        mse = float(mean_squared_error(y_task, y_pred))
        waic = float(compute_waic_from_residuals(y_task, y_pred, n_params=X_task.shape[1]))

        coef_df = pd.DataFrame({
            "feature_name": feats,
            "coefficient": reg.coef_.astype(float),
        })
        coef_tables[task_idx] = coef_df

        terms = []
        for feat, coef in zip(feats, reg.coef_):
            coef_str = "{:.10g}".format(float(coef))
            terms.append("{}*{}".format(coef_str, feat))
        equation = " + ".join(terms)

        summary_rows.append({
            "task_idx": task_idx,
            "target_name": target_name,
            "equation_r2": r2,
            "equation_mse": mse,
            "equation_waic": waic,
            "equation": equation,
        })

    return pd.DataFrame(summary_rows), coef_tables

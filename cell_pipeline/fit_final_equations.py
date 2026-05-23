from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


TARGET_NAMES = [
    "(x1+x2)/2",
    "(x1-x2).abs/2",
    "(y1+y2)/2",
    "(y1-y2).abs/2",
    "(z1+z2)/2",
    "(z1-z2).abs/2",
]

META_FEATURE_COLS = [
    "sample_id",
    "group_idx",
    "transition",
    "mother_t",
    "daughter_t",
    "mother_name",
    "daughter1_name",
    "daughter2_name",
]


def compute_waic_from_residuals(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
    mse = mean_squared_error(y_true, y_pred)
    mse = max(float(mse), 1e-12)
    log_lik = -0.5 * (np.log(2 * np.pi * mse) + (y_true - y_pred) ** 2 / mse)
    lppd = np.sum(log_lik)
    return float(-2 * (lppd - n_params))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "用 stage2_greedy_metrics_ols 的贪心顺序做最终 keep-k 截断拟合；"
            "图的数据也按 greedy 逻辑构造：前面是 greedy 入选顺序，后面补上 TopK 中未入选特征；"
            "图样式仿照 stage2_greedy_task_ols_*.png，星号位置由 --keep-k 控制。"
        )
    )
    parser.add_argument("--features-aligned", required=True, help="features_aligned.csv 路径")
    parser.add_argument("--targets-aligned", required=True, help="targets_aligned.csv 路径")
    parser.add_argument(
        "--stage2-greedy-ols-metrics-dir",
        required=True,
        help="stage2_greedy_metrics_ols 目录路径，里面应有 task_1_metrics.csv ... task_6_metrics.csv",
    )
    parser.add_argument(
        "--feature-order-dir",
        required=True,
        help="原始 TopK 特征顺序目录，例如 feature_orders/，里面应有 feature_order_task_1.csv ...",
    )
    parser.add_argument(
        "--keep-k",
        required=True,
        help="每个 target 保留特征数，逗号分隔，例如 5,6,4,7,3,8",
    )
    parser.add_argument(
        "--output-dir",
        default="final_equations_from_stage2_greedyplusremaining_ols",
        help="输出目录",
    )
    return parser.parse_args()


def parse_keep_k(text: str, n_targets: int) -> List[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != n_targets:
        raise ValueError(f"--keep-k 需要提供 {n_targets} 个整数，当前收到 {len(vals)} 个")
    if any(v <= 0 for v in vals):
        raise ValueError("--keep-k 中所有值都必须 > 0")
    return vals


def load_aligned_xy(features_path: str, targets_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feat = pd.read_csv(features_path)
    targ = pd.read_csv(targets_path)

    if "sample_id" not in feat.columns or "sample_id" not in targ.columns:
        raise KeyError("features_aligned.csv 和 targets_aligned.csv 都必须包含 sample_id")

    feat = feat.sort_values("sample_id").reset_index(drop=True)
    targ = targ.sort_values("sample_id").reset_index(drop=True)

    if not feat["sample_id"].equals(targ["sample_id"]):
        raise ValueError("features_aligned 和 targets_aligned 的 sample_id 未对齐")

    X_df = feat.drop(columns=[c for c in META_FEATURE_COLS if c in feat.columns]).copy()
    y_df = targ.drop(columns=["sample_id"]).copy()
    return feat, X_df, y_df


def load_feature_order(feature_order_csv: Path) -> List[str]:
    df = pd.read_csv(feature_order_csv)
    if df.empty:
        raise ValueError(f"{feature_order_csv} 为空")
    if "feature_name" in df.columns:
        feats = df["feature_name"].astype(str).tolist()
    else:
        feats = df.iloc[:, 0].astype(str).tolist()
    return [x.strip() for x in feats if str(x).strip()]


def load_greedy_selected_sequence(greedy_metrics_csv: Path) -> List[str]:
    df = pd.read_csv(greedy_metrics_csv)
    if df.empty:
        raise ValueError(f"{greedy_metrics_csv} 为空")
    required = {"k", "features"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{greedy_metrics_csv} 缺少列: {sorted(missing)}")

    df = df.sort_values("k").reset_index(drop=True)
    max_k = int(df["k"].max())
    row = df[df["k"] == max_k].iloc[-1]
    feats = [x.strip() for x in str(row["features"]).split(",") if x.strip()]
    if len(feats) != max_k:
        raise ValueError(f"{greedy_metrics_csv.name} 中 greedy 特征数与最大 k 不一致")
    return feats


def build_greedy_plus_remaining_order(greedy_selected: List[str], topk_order: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for f in greedy_selected:
        if f not in seen:
            ordered.append(f)
            seen.add(f)
    for f in topk_order:
        if f not in seen:
            ordered.append(f)
            seen.add(f)
    return ordered


def evaluate_prefix_metrics_ols(X_df: pd.DataFrame, y: np.ndarray, ordered_features: List[str]) -> pd.DataFrame:
    rows = []
    for k in range(1, len(ordered_features) + 1):
        feats = ordered_features[:k]
        missing = [f for f in feats if f not in X_df.columns]
        if missing:
            raise KeyError(f"以下特征不在 features_aligned 中: {missing}")
        X = X_df[feats].values.astype(float)
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        r2 = float(r2_score(y, y_pred))
        mse = float(mean_squared_error(y, y_pred))
        waic = float(compute_waic_from_residuals(y, y_pred, n_params=len(feats)))
        rows.append({
            "k": k,
            "features": ",".join(feats),
            "R2": r2,
            "MSE": mse,
            "WAIC": waic,
        })
    return pd.DataFrame(rows)


def fit_equation(X_df: pd.DataFrame, y: np.ndarray, features: List[str]):
    missing = [f for f in features if f not in X_df.columns]
    if missing:
        raise KeyError(f"以下特征不在 features_aligned 中: {missing}")
    X = X_df[features].values.astype(float)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    r2 = float(r2_score(y, y_pred))
    mse = float(mean_squared_error(y, y_pred))
    waic = float(compute_waic_from_residuals(y, y_pred, n_params=len(features)))
    return reg, y_pred, r2, mse, waic


def make_equation_string(features: List[str], coefs: np.ndarray) -> str:
    parts = []
    for coef, feat in zip(coefs, features):
        parts.append(f"({coef:.10g})*{feat}")
    return " + ".join(parts) if parts else "0"


def _annotate_star(ax, x_val: float, y_val: float) -> None:
    ax.scatter(
        [x_val], [y_val],
        marker="*",
        s=320,
        facecolor="#FFD700",
        edgecolor="black",
        linewidth=1.3,
        zorder=8,
    )


def plot_like_stage2_greedy(metrics_df: pd.DataFrame, keep_k: int, target_name: str, output_png: Path) -> None:
    plot_df = metrics_df.sort_values("k").reset_index(drop=True)
    x = plot_df["k"].astype(int).to_numpy()
    r2 = plot_df["R2"].astype(float).to_numpy()
    mse = plot_df["MSE"].astype(float).to_numpy()
    waic = plot_df["WAIC"].astype(float).to_numpy()

    if keep_k not in set(x.tolist()):
        raise ValueError(f"keep_k={keep_k} 不在可绘制的 k 范围内，当前可用: {x.tolist()}")
    star_idx = int(np.where(x == keep_k)[0][0])

    plt.rcParams["font.sans-serif"] = ["Arial", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(
        f"Feature Count vs. Model Metrics - {target_name} (Single-Task OLS)",
        fontsize=14,
        y=0.95,
    )

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(x, r2, marker="o", linestyle="-", color="blue", linewidth=2, markersize=6, label="R² Score")
    _annotate_star(ax1, x[star_idx], r2[star_idx])
    ax1.set_xlabel("Number of Features (Top-K)", fontsize=11)
    ax1.set_ylabel("R² Score", fontsize=11)
    ax1.set_xlim(0.5, np.max(x) + 0.5)
    ax1.set_ylim(max(0, np.min(r2) - 0.05), min(1.05, np.max(r2) + 0.05))
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_title("R² Score: Higher = Better", fontsize=12, pad=15)

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(x, mse, marker="s", linestyle="-", color="red", linewidth=2, markersize=6, label="MSE")
    _annotate_star(ax2, x[star_idx], mse[star_idx])
    ax2.set_xlabel("Number of Features", fontsize=11)
    ax2.set_ylabel("MSE", fontsize=11)
    ax2.set_xlim(0.5, np.max(x) + 0.5)
    ax2.set_ylim(max(0, np.min(mse) - 0.01), np.max(mse) + 0.01)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_title("MSE: Lower = Better", fontsize=12, pad=15)

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(x, waic, marker="^", linestyle="-", color="green", linewidth=2, markersize=6, label="WAIC")
    _annotate_star(ax3, x[star_idx], waic[star_idx])
    ax3.set_xlabel("Number of Features", fontsize=11)
    ax3.set_ylabel("WAIC Value", fontsize=11)
    ax3.set_xlim(0.5, np.max(x) + 0.5)
    ax3.set_ylim(np.min(waic) - 10, np.max(waic) + 10)
    ax3.legend(loc="upper right", fontsize=10)
    ax3.grid(alpha=0.3)
    ax3.set_title("WAIC: Lower = Better", fontsize=12, pad=15)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coef_dir = out_dir / "equations"
    coef_dir.mkdir(exist_ok=True)
    plots_dir = out_dir / "plots_keepk_star"
    plots_dir.mkdir(exist_ok=True)
    metrics_dir = out_dir / "metrics_keepk_star"
    metrics_dir.mkdir(exist_ok=True)

    _, X_df, y_df = load_aligned_xy(args.features_aligned, args.targets_aligned)
    n_targets = y_df.shape[1]
    keep_k_list = parse_keep_k(args.keep_k, n_targets)

    summary_rows = []

    for task_idx in range(n_targets):
        greedy_csv = Path(args.stage2_greedy_ols_metrics_dir) / f"task_{task_idx+1}.csv"
        feature_order_csv = Path(args.feature_order_dir) / f"feature_order_task_{task_idx+1}.csv"

        if not greedy_csv.exists():
            raise FileNotFoundError(f"未找到: {greedy_csv}")
        if not feature_order_csv.exists():
            raise FileNotFoundError(f"未找到: {feature_order_csv}")

        greedy_selected = load_greedy_selected_sequence(greedy_csv)
        topk_order = load_feature_order(feature_order_csv)
        full_order = build_greedy_plus_remaining_order(greedy_selected, topk_order)

        y = y_df.iloc[:, task_idx].values.astype(float)
        full_metrics_df = evaluate_prefix_metrics_ols(X_df, y, full_order)
        full_metrics_df.to_csv(metrics_dir / f"task_{task_idx+1}.csv", index=False, encoding="utf-8-sig")

        keep_k = keep_k_list[task_idx]
        if keep_k > len(full_order):
            raise ValueError(f"task {task_idx+1} 的 keep_k={keep_k} 超过可用特征数 {len(full_order)}")

        selected_features = full_order[:keep_k]
        reg, _, r2, mse, waic = fit_equation(X_df, y, selected_features)
        equation = make_equation_string(selected_features, reg.coef_)

        coef_df = pd.DataFrame({
            "task_idx": task_idx,
            "target_name": TARGET_NAMES[task_idx] if task_idx < len(TARGET_NAMES) else y_df.columns[task_idx],
            "order": np.arange(1, len(selected_features) + 1),
            "feature_name": selected_features,
            "coefficient": reg.coef_,
        })
        coef_df.to_csv(coef_dir / f"task_{task_idx+1}_equation_coefficients.csv", index=False, encoding="utf-8-sig")

        plot_like_stage2_greedy(
            metrics_df=full_metrics_df,
            keep_k=keep_k,
            target_name=TARGET_NAMES[task_idx] if task_idx < len(TARGET_NAMES) else y_df.columns[task_idx],
            output_png=plots_dir / f"feature_metrics_task_{task_idx+1}_keepk_star.png",
        )

        summary_rows.append({
            "task_idx": task_idx,
            "target_name": TARGET_NAMES[task_idx] if task_idx < len(TARGET_NAMES) else y_df.columns[task_idx],
            "keep_k": keep_k,
            "features": ",".join(selected_features),
            "equation_r2": r2,
            "equation_mse": mse,
            "equation_waic": waic,
            "equation": equation,
            "plot_file": f"plots_keepk_star/feature_metrics_task_{task_idx+1}_keepk_star.png",
            "metrics_file": f"metrics_keepk_star/task_{task_idx+1}.csv",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "equation_summary.csv", index=False, encoding="utf-8-sig")
    print(f"完成。输出目录: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

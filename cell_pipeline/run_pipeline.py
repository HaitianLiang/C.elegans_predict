
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .data_io import (
    build_timepoint_cells,
    discover_adjacency_files,
    discover_cell_data_files,
    infer_group_idx_from_path,
    load_cell_data_file,
)
from .event_builder import build_division_events
from .features import CellFeatureExtractor, build_feature_table_from_events
from .modeling import (
    TARGET_NAMES,
    align_feature_and_target_tables,
    evaluate_stepwise_metrics,
    evaluate_topk_prefix_metrics,
    fit_initial_lasso,
    fit_no_intercept_equations,
    metrics_dict_to_matrices,
    plot_feature_metrics_single_task,
    select_top_features,
)
from .relations import load_lineage_map, normalize_transitions
from .targets import build_target_table, export_legacy_compatible_vectors


DEFAULT_MANUAL_INDICES = {
    0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    4: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    5: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把原 notebook 重构成脚本化流水线：原始 CellData/adj -> 事件 -> 特征 -> 目标 -> Lasso/逐步指标"
    )
    parser.add_argument("--cell-data", required=True, help="单个 CellData_*.csv、文件夹、或 glob")
    parser.add_argument("--adjacency", required=True, help="单个 G*.csv、文件夹、或 glob")
    parser.add_argument("--lineage", required=True, help="cell_mother_daughter.txt")
    parser.add_argument("--transitions", default="T0-1", help="逗号分隔，例如 T0-1,T1-2；默认 T0-1")
    parser.add_argument("--output-dir", default="pipeline_output", help="输出目录")
    parser.add_argument("--top-k", type=int, default=20, help="候选特征数")
    parser.add_argument("--use-manual-order", action="store_true", help="沿用 notebook 里的手工特征顺序")
    parser.add_argument("--disable-adjacent-other-functions", action="store_true")
    parser.add_argument("--disable-self-other-functions", action="store_true")
    parser.add_argument("--stepwise-use-ols", action="store_true", help="第二阶段贪心筛选用 OLS；默认 LassoCV")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _save_metric_dict(metric_dict: Dict[int, pd.DataFrame], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for task_idx, df in metric_dict.items():
        df.to_csv(out_dir / "{}_{}.csv".format(stem, task_idx + 1), index=False, encoding="utf-8-sig")


def _plot_metric_dict(
    metric_dict: Dict[int, pd.DataFrame],
    out_dir: Path,
    prefix: str,
    star_positions: Dict[int, int] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    score_list, mse_list, waic_list = metrics_dict_to_matrices(metric_dict, n_targets=len(TARGET_NAMES))
    plot_feature_metrics_single_task(
        Score_list=score_list,
        MSE_list=mse_list,
        WAIC_list=waic_list,
        imp_coef_list=None,
        target_names=TARGET_NAMES,
        save_dir=str(out_dir),
        prefix=prefix,
        star_positions=star_positions,
    )


def _reorder_topk_after_stage2(
    feature_orders: Dict[int, List[str]],
    best_models: Dict[int, Dict],
) -> Dict[int, List[str]]:
    reordered: Dict[int, List[str]] = {}
    for task_idx, feats in feature_orders.items():
        selected = list(best_models.get(task_idx, {}).get("features", []) or [])
        selected_set = set(selected)
        remaining = [f for f in feats if f not in selected_set]
        reordered[task_idx] = selected + remaining
    return reordered


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transitions = normalize_transitions(args.transitions)
    lineage_map = load_lineage_map(args.lineage)
    adjacency_by_t = discover_adjacency_files(args.adjacency)

    cell_files = discover_cell_data_files(args.cell_data)
    if not cell_files:
        raise FileNotFoundError("没有找到 CellData 文件: {}".format(args.cell_data))

    all_events = []
    all_timepoint_cells: Dict[int, Dict[int, Dict[str, Dict]]] = {}

    for cell_path in cell_files:
        group_idx = infer_group_idx_from_path(cell_path)
        cell_df = load_cell_data_file(cell_path)
        timepoint_cells = build_timepoint_cells(cell_df, adjacency_by_t=adjacency_by_t)
        events_df = build_division_events(
            group_idx=group_idx,
            timepoint_cells=timepoint_cells,
            lineage_map=lineage_map,
            transitions=transitions,
        )
        if not events_df.empty:
            all_events.append(events_df)
            all_timepoint_cells[group_idx] = timepoint_cells

    if not all_events:
        raise ValueError("没有构造出任何分裂事件，请检查 transitions / lineage / CellData 是否匹配。")

    events_df = pd.concat(all_events, ignore_index=True)
    events_df.to_csv(out_dir / "division_events.csv", index=False, encoding="utf-8-sig")

    export_legacy_compatible_vectors(events_df, str(out_dir / "vectors"))

    feature_toggles = {
        "adjacent_other_functions": not args.disable_adjacent_other_functions,
        "self_other_functions": not args.disable_self_other_functions,
    }
    extractor = CellFeatureExtractor(feature_toggles=feature_toggles)

    features_df = build_feature_table_from_events(
        events_df=events_df,
        timepoint_cells_by_group=all_timepoint_cells,
        extractor=extractor,
    )
    extractor.export_feature_table(str(out_dir / "feature_catalog.md"))
    features_df.to_csv(out_dir / "features.csv", index=False, encoding="utf-8-sig")

    targets_df = build_target_table(events_df)
    targets_df.to_csv(out_dir / "targets_6cols.csv", index=False, encoding="utf-8-sig")

    aligned_features, aligned_targets = align_feature_and_target_tables(features_df, targets_df, key="sample_id")
    aligned_features.to_csv(out_dir / "features_aligned.csv", index=False, encoding="utf-8-sig")
    aligned_targets.to_csv(out_dir / "targets_aligned.csv", index=False, encoding="utf-8-sig")

    meta_cols = [
        "sample_id", "group_idx", "transition", "mother_t", "daughter_t",
        "mother_name", "daughter1_name", "daughter2_name",
    ]
    X_df = aligned_features.drop(columns=meta_cols)
    y_df = aligned_targets.drop(columns=["sample_id"])

    lasso_result = fit_initial_lasso(X_df, y_df, random_state=args.random_state)
    lasso_result["X_normalized"].to_csv(out_dir / "normalized_X.csv", index=False, encoding="utf-8-sig")
    lasso_result["y_normalized"].to_csv(out_dir / "normalized_y.csv", index=False, encoding="utf-8-sig")
    lasso_result["y_pred"].to_csv(out_dir / "lasso_y_pred.csv", index=False, encoding="utf-8-sig")

    adjusted_dir = out_dir / "adjusted_coefficients"
    adjusted_dir.mkdir(exist_ok=True)
    for task_idx, coef_series in enumerate(lasso_result["adjusted_coef_list"]):
        coef_series.rename("coefficient").to_csv(
            adjusted_dir / "task_{}_adjusted_coefficients.csv".format(task_idx + 1),
            encoding="utf-8-sig"
        )

    manual_order = DEFAULT_MANUAL_INDICES if args.use_manual_order else None
    feature_orders = select_top_features(
        lasso_result["adjusted_coef_list"],
        top_k=args.top_k,
        manual_indices_per_task=manual_order,
    )

    order_dir = out_dir / "feature_orders"
    order_dir.mkdir(exist_ok=True)
    for task_idx, feats in feature_orders.items():
        pd.DataFrame({"feature_name": feats}).to_csv(
            order_dir / "feature_order_task_{}.csv".format(task_idx + 1),
            index=False,
            encoding="utf-8-sig",
        )

    # ----------------------
    # Stage1：当前 TopK 顺序的连续曲线
    # ----------------------
    stage1_metrics_lasso = evaluate_topk_prefix_metrics(
        X_df=X_df,
        y_df=y_df,
        feature_orders=feature_orders,
        max_k=args.top_k,
        use_lasso=True,
        random_state=args.random_state,
    )
    _save_metric_dict(stage1_metrics_lasso, out_dir / "stage1_metrics_lasso", "task")
    _plot_metric_dict(stage1_metrics_lasso, out_dir / "stage1_plots_lasso", "feature_metrics_task_lasso")

    stage1_metrics_ols = evaluate_topk_prefix_metrics(
        X_df=X_df,
        y_df=y_df,
        feature_orders=feature_orders,
        max_k=args.top_k,
        use_lasso=False,
        random_state=args.random_state,
    )
    _save_metric_dict(stage1_metrics_ols, out_dir / "stage1_metrics_ols", "task")
    _plot_metric_dict(stage1_metrics_ols, out_dir / "stage1_plots_ols", "feature_metrics_task_ols")

    # ----------------------
    # Stage2：两步贪心筛选
    # - 同时跑 Lasso 版和 OLS 版，用于画图对比
    # - 最终 summary / 方程 / 入选特征 使用主方法（由 --stepwise-use-ols 控制）
    # ----------------------
    stage2_metric_results_lasso, stage2_best_models_lasso = evaluate_stepwise_metrics(
        X_df=X_df,
        y_df=y_df,
        feature_orders=feature_orders,
        max_k=args.top_k,
        use_lasso=True,
        random_state=args.random_state,
    )
    _save_metric_dict(stage2_metric_results_lasso, out_dir / "stage2_greedy_metrics_lasso", "task")
    _plot_metric_dict(
        stage2_metric_results_lasso,
        out_dir / "stage2_greedy_plots_lasso",
        "stage2_greedy_task_lasso",
        star_positions={k: v.get("best_k") for k, v in stage2_best_models_lasso.items()},
    )

    stage2_metric_results_ols, stage2_best_models_ols = evaluate_stepwise_metrics(
        X_df=X_df,
        y_df=y_df,
        feature_orders=feature_orders,
        max_k=args.top_k,
        use_lasso=False,
        random_state=args.random_state,
    )
    _save_metric_dict(stage2_metric_results_ols, out_dir / "stage2_greedy_metrics_ols", "task")
    _plot_metric_dict(
        stage2_metric_results_ols,
        out_dir / "stage2_greedy_plots_ols",
        "stage2_greedy_task_ols",
        star_positions={k: v.get("best_k") for k, v in stage2_best_models_ols.items()},
    )

    # 选择主结果：默认 Lasso；若指定 --stepwise-use-ols，则主结果切换为 OLS
    if args.stepwise_use_ols:
        metric_results = stage2_metric_results_ols
        best_models = stage2_best_models_ols
        main_stage2_method = "ols"
    else:
        metric_results = stage2_metric_results_lasso
        best_models = stage2_best_models_lasso
        main_stage2_method = "lasso"

    # 向后兼容：metrics/ 保存主结果
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    for task_idx, metric_df in metric_results.items():
        metric_df.to_csv(metrics_dir / "task_{}_metrics.csv".format(task_idx + 1), index=False, encoding="utf-8-sig")

    # 第二阶段最终入选顺序：已入选放前面，剩下保持原顺序；最后一个入选特征打星号
    reordered_feature_orders = _reorder_topk_after_stage2(feature_orders, best_models)
    reordered_dir = out_dir / "feature_orders_stage2_reordered"
    reordered_dir.mkdir(exist_ok=True)
    for task_idx, feats in reordered_feature_orders.items():
        selected = list(best_models.get(task_idx, {}).get("features", []) or [])
        last_selected = best_models.get(task_idx, {}).get("selection_stop_feature")
        selected_set = set(selected)

        rows = []
        for rank, feat in enumerate(feats, start=1):
            is_selected = feat in selected_set
            is_last_selected = (feat == last_selected)
            feat_display = feat + ("*" if is_last_selected else "")
            rows.append({
                "rank": rank,
                "feature_name": feat,
                "feature_name_display": feat_display,
                "is_selected": int(is_selected),
                "is_last_selected": int(is_last_selected),
            })
        pd.DataFrame(rows).to_csv(
            reordered_dir / "task_{}_feature_order_stage2_reordered.csv".format(task_idx + 1),
            index=False,
            encoding="utf-8-sig",
        )

    # 可选对照：按“第二阶段最终顺序重排后的完整 TopK”再画连续曲线（不影响最终结果）
    stage2_reordered_metrics_lasso = evaluate_topk_prefix_metrics(
        X_df=X_df,
        y_df=y_df,
        feature_orders=reordered_feature_orders,
        max_k=args.top_k,
        use_lasso=True,
        random_state=args.random_state,
    )
    _save_metric_dict(stage2_reordered_metrics_lasso, out_dir / "stage2_reordered_metrics_lasso", "task")
    _plot_metric_dict(stage2_reordered_metrics_lasso, out_dir / "stage2_reordered_plots_lasso", "stage2_reordered_task_lasso")

    stage2_reordered_metrics_ols = evaluate_topk_prefix_metrics(
        X_df=X_df,
        y_df=y_df,
        feature_orders=reordered_feature_orders,
        max_k=args.top_k,
        use_lasso=False,
        random_state=args.random_state,
    )
    _save_metric_dict(stage2_reordered_metrics_ols, out_dir / "stage2_reordered_metrics_ols", "task")
    _plot_metric_dict(stage2_reordered_metrics_ols, out_dir / "stage2_reordered_plots_ols", "stage2_reordered_task_ols")

    # 最终无截距方程拟合
    equation_summary_df, equation_coef_tables = fit_no_intercept_equations(
        X_df=X_df,
        y_df=y_df,
        best_models=best_models,
    )
    equation_summary_df.to_csv(out_dir / "equation_summary.csv", index=False, encoding="utf-8-sig")
    equations_dir = out_dir / "equations"
    equations_dir.mkdir(exist_ok=True)
    for task_idx, coef_df in equation_coef_tables.items():
        coef_df.to_csv(
            equations_dir / "task_{}_equation_coefficients.csv".format(task_idx + 1),
            index=False,
            encoding="utf-8-sig",
        )

    summary_rows = []
    for task_idx in range(len(TARGET_NAMES)):
        info = best_models.get(task_idx, {})
        eq_row = equation_summary_df[equation_summary_df["task_idx"] == task_idx]
        eq_dict = eq_row.iloc[0].to_dict() if not eq_row.empty else {}

        summary_rows.append({
            "task_idx": task_idx,
            "target_name": TARGET_NAMES[task_idx],
            "best_k": info.get("best_k"),
            "best_r2": info.get("best_r2"),
            "best_mse": info.get("best_mse"),
            "best_waic": info.get("best_waic"),
            "features": ",".join(info.get("features", []) or []),
            "selection_stop_feature": info.get("selection_stop_feature"),
            "stage2_method": main_stage2_method,
            "equation_r2": eq_dict.get("equation_r2"),
            "equation_mse": eq_dict.get("equation_mse"),
            "equation_waic": eq_dict.get("equation_waic"),
            "equation": eq_dict.get("equation", ""),
        })

    pd.DataFrame(summary_rows).to_csv(out_dir / "best_model_summary.csv", index=False, encoding="utf-8-sig")

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "cell_data": args.cell_data,
                "adjacency": args.adjacency,
                "lineage": args.lineage,
                "transitions": transitions,
                "top_k": args.top_k,
                "use_manual_order": args.use_manual_order,
                "stepwise_use_ols": args.stepwise_use_ols,
                "stage2_main_method": main_stage2_method,
                "feature_toggles": feature_toggles,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("完成。输出目录: {}".format(out_dir.resolve()))


if __name__ == "__main__":
    main()

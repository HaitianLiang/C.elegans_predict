
from __future__ import annotations

from typing import Tuple

import pandas as pd


TARGET_COLUMNS = [
    "x_mean",
    "x_half_absdiff",
    "y_mean",
    "y_half_absdiff",
    "z_mean",
    "z_half_absdiff",
]


def build_target_table(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    sum_*.csv 的正确含义：
    - 每一行是两个子细胞的原始坐标
    这里直接从 daughter1 / daughter2 构造 6 个目标列。
    """
    df = events_df[[
        "sample_id",
        "daughter1_x", "daughter2_x",
        "daughter1_y", "daughter2_y",
        "daughter1_z", "daughter2_z",
    ]].copy()

    target_df = pd.DataFrame({
        "sample_id": df["sample_id"],
        "x_mean": (df["daughter1_x"] + df["daughter2_x"]) / 2.0,
        "x_half_absdiff": (df["daughter1_x"] - df["daughter2_x"]).abs() / 2.0,
        "y_mean": (df["daughter1_y"] + df["daughter2_y"]) / 2.0,
        "y_half_absdiff": (df["daughter1_y"] - df["daughter2_y"]).abs() / 2.0,
        "z_mean": (df["daughter1_z"] + df["daughter2_z"]) / 2.0,
        "z_half_absdiff": (df["daughter1_z"] - df["daughter2_z"]).abs() / 2.0,
    })
    return target_df


def export_legacy_compatible_vectors(events_df: pd.DataFrame, output_dir: str) -> None:
    """
    导出两类文件：
    1. corrected: pipeline 真正使用的“按事件一行”的版本
    2. legacy_duplicated: 保留 notebook 旧格式，便于对照检查

    注意：
    - corrected mother_*.csv 不再重复两次
    - corrected sum_*.csv 每行两个 daughter 坐标
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # corrected
    for axis in ["x", "y", "z"]:
        events_df[[f"mother_{axis}"]].to_csv(
            os.path.join(output_dir, f"mother_{axis}_vector_corrected.csv"),
            index=False
        )
        events_df[[f"daughter1_{axis}", f"daughter2_{axis}"]].to_csv(
            os.path.join(output_dir, f"sum_{axis}_vector_corrected.csv"),
            index=False, header=False
        )

    # legacy duplicated mother
    legacy = pd.DataFrame({"sample_id": events_df["sample_id"]})
    for axis in ["x", "y", "z"]:
        mother_vals = events_df[f"mother_{axis}"].tolist()
        duplicated = []
        for v in mother_vals:
            duplicated.extend([v, v])
        pd.DataFrame({f"mother_{axis}": duplicated}).to_csv(
            os.path.join(output_dir, f"mother_{axis}_vector_legacy_duplicated.csv"),
            index=False
        )

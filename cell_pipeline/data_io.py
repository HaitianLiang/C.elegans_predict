
from __future__ import annotations

import os
import re
import glob
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

GI_TO_T = {"G1": 0, "G2": 1, "G3": 2, "G4": 3}
T_TO_GI = {v: k for k, v in GI_TO_T.items()}


def infer_group_idx_from_path(path: str) -> int:
    name = os.path.basename(path)
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def discover_cell_data_files(cell_data_input: str) -> List[str]:
    """
    支持:
    - 单个 CSV 文件
    - 文件夹（读取其中 CellData_*.csv）
    - glob pattern
    """
    if os.path.isfile(cell_data_input):
        return [cell_data_input]
    if os.path.isdir(cell_data_input):
        files = sorted(glob.glob(os.path.join(cell_data_input, "CellData_*.csv")))
        if not files:
            files = sorted(glob.glob(os.path.join(cell_data_input, "*.csv")))
        return files
    return sorted(glob.glob(cell_data_input))


def load_cell_data_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    required = {"CellName", "T", "X", "Y", "Z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少列: {missing}")
    df["CellName"] = df["CellName"].astype(str).str.replace("'", "", regex=False).str.strip()
    df["T"] = df["T"].astype(int)
    for col in ["X", "Y", "Z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["CellName", "T", "X", "Y", "Z"]).reset_index(drop=True)
    return df


def load_adjacency_csv(path: str) -> Dict[str, Dict[str, float]]:
    """
    读取 G1.csv / G2.csv ... 形式的邻接矩阵。
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.replace("'", "", regex=False).str.strip()

    first_col = df.columns[0]
    if first_col.lower() in {"cell identity", "cell_identity", "cellname", "cell"}:
        df[first_col] = df[first_col].str.replace("'", "", regex=False).str.strip()
        df = df.set_index(first_col)

    clean_cols = {}
    for c in df.columns:
        clean_cols[c] = str(c).replace("'", "").strip()
    df = df.rename(columns=clean_cols)

    for c in df.columns:
        df[c] = df[c].astype(str).str.replace("'", "", regex=False).str.strip()
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    result: Dict[str, Dict[str, float]] = {}
    for cell in df.index:
        cell_name = str(cell).replace("'", "").strip()
        row = {}
        for nb in df.columns:
            if nb == cell_name:
                continue
            val = float(df.loc[cell, nb])
            val = max(0.0, min(1.0, val))
            if val > 0:
                row[nb] = val
        result[cell_name] = row
    return result


def discover_adjacency_files(adjacency_input: str) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    支持:
    - 单个 G1.csv
    - 文件夹（读取 G*.csv）
    - glob / pattern
    返回: {t: adjacency_matrix}
    """
    files: List[str]
    if os.path.isfile(adjacency_input):
        files = [adjacency_input]
    elif os.path.isdir(adjacency_input):
        files = sorted(glob.glob(os.path.join(adjacency_input, "G*.csv")))
    else:
        files = sorted(glob.glob(adjacency_input))

    out: Dict[int, Dict[str, Dict[str, float]]] = {}
    for path in files:
        name = os.path.basename(path)
        m = re.search(r"(G\d+)", name, re.I)
        if not m:
            continue
        gi = m.group(1).upper()
        if gi not in GI_TO_T:
            continue
        out[GI_TO_T[gi]] = load_adjacency_csv(path)
    return out


def build_timepoint_cells(
    cell_df: pd.DataFrame,
    adjacency_by_t: Dict[int, Dict[str, Dict[str, float]]] | None = None,
) -> Dict[int, Dict[str, Dict]]:
    """
    构造成:
    {
      t: {
        cell_name: {
          'self_coords': np.ndarray([x, y, z]),
          'adjacency_matrix': {...},
          'gi_group': 'G1'
        }
      }
    }
    """
    adjacency_by_t = adjacency_by_t or {}
    timepoint_cells: Dict[int, Dict[str, Dict]] = {}

    for t, t_df in cell_df.groupby("T", sort=True):
        t_df = t_df.copy().reset_index(drop=True)
        cells_at_t = set(t_df["CellName"].tolist())
        adj_matrix_t = adjacency_by_t.get(int(t), {})
        current: Dict[str, Dict] = {}
        for _, row in t_df.iterrows():
            cell_name = row["CellName"]
            raw_adj = adj_matrix_t.get(cell_name, {})
            filtered_adj = {
                nb: float(v)
                for nb, v in raw_adj.items()
                if nb in cells_at_t and nb != cell_name and float(v) > 0
            }
            current[cell_name] = {
                "self_coords": np.array([row["X"], row["Y"], row["Z"]], dtype=float),
                "adjacency_matrix": filtered_adj,
                "gi_group": T_TO_GI.get(int(t), f"G{int(t)+1}"),
            }
        timepoint_cells[int(t)] = current
    return timepoint_cells

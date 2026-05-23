
from __future__ import annotations

import pandas as pd
from typing import Dict, List, Tuple


def load_lineage_map(lineage_file: str) -> Dict[str, Tuple[str, str]]:
    """
    从 cell_mother_daughter.txt 读取母细胞 -> 两个子细胞 的映射。

    文件格式示例:
    cell    mother   daughter
    AB      P0       "ABa,ABp"

    返回:
        {
            "AB": ("ABa", "ABp"),
            "ABa": ("ABar", "ABal"),
            ...
        }
    """
    df = pd.read_csv(lineage_file, sep=None, engine="python")
    cols = [str(c).strip().lower() for c in df.columns]
    rename_map = dict(zip(df.columns, cols))
    df = df.rename(columns=rename_map)

    required = {"cell", "daughter"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"lineage 文件缺少列: {missing}")

    lineage: Dict[str, Tuple[str, str]] = {}
    for _, row in df.iterrows():
        cell = str(row["cell"]).strip().replace("'", "").replace('"', "")
        daughters_raw = str(row["daughter"]).strip()
        daughters_raw = daughters_raw.replace('"', "").replace("'", "")
        if daughters_raw.upper() == "NA":
            continue
        daughters = [x.strip() for x in daughters_raw.split(",") if x.strip()]
        if len(daughters) != 2:
            continue
        lineage[cell] = (daughters[0], daughters[1])
    return lineage


def transition_to_times(transition: str) -> Tuple[int, int]:
    """
    'T0-1' -> (0, 1)
    """
    transition = transition.strip().upper().replace(" ", "")
    if not transition.startswith("T") or "-" not in transition:
        raise ValueError(f"非法 transition: {transition}")
    left, right = transition[1:].split("-")
    return int(left), int(right)


def normalize_transitions(transitions: List[str] | str | None) -> List[str]:
    if transitions is None:
        return ["T0-1", "T1-2", "T2-3"]
    if isinstance(transitions, str):
        transitions = [x.strip() for x in transitions.split(",") if x.strip()]
    normed = []
    for t in transitions:
        a, b = transition_to_times(t)
        if b != a + 1:
            raise ValueError(f"当前只支持相邻时间段，收到: {t}")
        normed.append(f"T{a}-{b}")
    return normed

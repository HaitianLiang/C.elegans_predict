
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from .relations import normalize_transitions, transition_to_times


def build_division_events(
    group_idx: int,
    timepoint_cells: Dict[int, Dict[str, Dict]],
    lineage_map: Dict[str, Tuple[str, str]],
    transitions: List[str] | str | None = None,
) -> pd.DataFrame:
    """
    核心修复：
    1) 一条样本 = 一个母细胞分裂事件，不再依赖 notebook 里“偶数行过滤”的隐式逻辑。
    2) 每条样本同时绑定 mother + 两个 daughters，彻底避免 mother_*.csv / sum_*.csv 错位。
    """
    transitions = normalize_transitions(transitions)
    rows = []

    for transition in transitions:
        t_mother, t_daughter = transition_to_times(transition)
        mothers = timepoint_cells.get(t_mother, {})
        daughters = timepoint_cells.get(t_daughter, {})

        if not mothers or not daughters:
            continue

        for mother_name, mother_info in mothers.items():
            if mother_name not in lineage_map:
                continue

            d1, d2 = lineage_map[mother_name]
            if d1 not in daughters or d2 not in daughters:
                continue

            mcoords = mother_info["self_coords"]
            d1coords = daughters[d1]["self_coords"]
            d2coords = daughters[d2]["self_coords"]

            rows.append(
                {
                    "sample_id": f"group{group_idx}|{transition}|{mother_name}",
                    "group_idx": group_idx,
                    "transition": transition,
                    "mother_t": t_mother,
                    "daughter_t": t_daughter,
                    "mother_name": mother_name,
                    "daughter1_name": d1,
                    "daughter2_name": d2,
                    "mother_x": float(mcoords[0]),
                    "mother_y": float(mcoords[1]),
                    "mother_z": float(mcoords[2]),
                    "daughter1_x": float(d1coords[0]),
                    "daughter1_y": float(d1coords[1]),
                    "daughter1_z": float(d1coords[2]),
                    "daughter2_x": float(d2coords[0]),
                    "daughter2_y": float(d2coords[1]),
                    "daughter2_z": float(d2coords[2]),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "sample_id", "group_idx", "transition", "mother_t", "daughter_t",
                "mother_name", "daughter1_name", "daughter2_name",
                "mother_x", "mother_y", "mother_z",
                "daughter1_x", "daughter1_y", "daughter1_z",
                "daughter2_x", "daughter2_y", "daughter2_z",
            ]
        )

    df = pd.DataFrame(rows).sort_values(["group_idx", "mother_t", "mother_name"]).reset_index(drop=True)
    return df

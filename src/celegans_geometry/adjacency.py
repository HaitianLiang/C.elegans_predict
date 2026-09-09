"""Weighted adjacency matrix loading and lookup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from .constants import G_INDEX_TO_CELL_COUNT


def _clean(x: object) -> str:
    return str(x).replace("'", "").replace('"', "").strip()


def load_square_adjacency_csv(path: str | Path) -> pd.DataFrame:
    """Read a labeled square adjacency CSV while preserving continuous weights."""
    path = Path(path)
    raw = pd.read_csv(path, dtype=str)
    raw.columns = [_clean(c) for c in raw.columns]
    first = raw.columns[0]
    first_values = raw[first].map(_clean)
    numeric_fraction = pd.to_numeric(first_values, errors="coerce").notna().mean()
    if first.lower() in {"cell", "cellname", "cell identity", "cell_identity", "name", "unnamed: 0", ""} or numeric_fraction < 0.8:
        raw[first] = first_values
        raw = raw.set_index(first)
    raw.index = [_clean(x) for x in raw.index]
    raw.columns = [_clean(x) for x in raw.columns]
    for c in raw.columns:
        raw[c] = pd.to_numeric(raw[c].map(_clean), errors="coerce").fillna(0.0)
    common = [x for x in raw.index if x in raw.columns]
    if not common:
        raise ValueError(f"Could not infer matching row/column labels in {path}")
    raw = raw.loc[common, common].astype(float)
    if not np.allclose(np.diag(raw.to_numpy()), 0.0, atol=1e-10):
        warnings.warn(f"{path.name}: nonzero diagonal detected; diagonal is set to zero")
        np.fill_diagonal(raw.values, 0.0)
    if not np.allclose(raw.to_numpy(), raw.to_numpy().T, atol=1e-8):
        warnings.warn(f"{path.name}: adjacency is not symmetric")
    return raw


def _infer_cell_count_from_name(path: Path) -> int | None:
    stem = path.stem.lower()
    m = re.search(r"g\s*(\d+)", stem)
    if m:
        return G_INDEX_TO_CELL_COUNT.get(int(m.group(1)))
    nums = [int(x) for x in re.findall(r"\d+", stem)]
    for n in nums:
        if n in {4, 6, 7, 8, 12, 14, 15}:
            return n
    return None


@dataclass
class WeightedAdjacencyStore:
    by_cell_count: dict[int, pd.DataFrame]
    fallback_all_context: bool = False

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        fallback_all_context: bool = False,
    ) -> "WeightedAdjacencyStore":
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(directory)
        found: dict[int, pd.DataFrame] = {}
        for path in sorted(directory.glob("*.csv")):
            count = _infer_cell_count_from_name(path)
            if count is None:
                continue
            found[count] = load_square_adjacency_csv(path)
        if not found and not fallback_all_context:
            raise FileNotFoundError(
                f"No stage-labeled adjacency CSVs found in {directory}. "
                "Expected names such as G1.csv..G7.csv or 4.csv,6.csv,..."
            )
        return cls(found, fallback_all_context=fallback_all_context)

    def neighbor_weights(
        self,
        stage_count: int,
        mother: str,
        available_cells: list[str] | pd.Index,
    ) -> dict[str, float]:
        available = set(map(str, available_cells))
        if stage_count in self.by_cell_count:
            mat = self.by_cell_count[stage_count]
            if mother not in mat.index:
                if not self.fallback_all_context:
                    raise KeyError(f"{mother} not found in adjacency for {stage_count}-cell stage")
            else:
                row = mat.loc[mother]
                return {
                    str(nb): float(w)
                    for nb, w in row.items()
                    if nb != mother and nb in available and float(w) > 0.0
                }
        if self.fallback_all_context:
            return {c: 1.0 for c in available if c != mother}
        raise KeyError(f"No adjacency matrix for {stage_count}-cell stage")

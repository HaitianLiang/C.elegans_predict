"""Input adapters for CellData-style coordinate tables and canonical event tables."""

from __future__ import annotations

from pathlib import Path
import re
import zipfile
import tempfile

import numpy as np
import pandas as pd

from .constants import MOTHER_DAUGHTERS, STAGE_BLOCK_BY_MOTHER_CELL_COUNT
from .targets import attach_targets


def _clean_cell_name(x: object) -> str:
    return str(x).replace("'", "").replace('"', "").strip()


def _embryo_id_from_path(path: Path, base: Path) -> str:
    """Stable embryo id from the file path relative to the supplied data root."""
    try:
        rel = path.relative_to(base)
    except ValueError:
        rel = Path(path.name)
    return "__".join(rel.with_suffix("").parts)


def discover_celldata_files(root: str | Path) -> list[Path]:
    """Find CellData_*.csv recursively under a directory.

    A ZIP file is also accepted and extracted to a temporary directory for the
    duration of this call. For reproducible workflows, unpack ZIPs explicitly
    and point to the resulting directory.
    """
    root = Path(root)
    if root.is_file() and root.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="celegans_celldata_"))
        with zipfile.ZipFile(root, "r") as zf:
            zf.extractall(tmp)
        root = tmp
    if not root.exists():
        raise FileNotFoundError(root)
    files = sorted(
        p for p in root.rglob("CellData_*.csv")
        if "__MACOSX" not in str(p)
    )
    if not files:
        raise FileNotFoundError(f"No CellData_*.csv found under {root}")
    return files


def load_celldata(root: str | Path) -> pd.DataFrame:
    """Load CellData snapshots into a canonical long table.

    Required columns in each CSV: CellName, T, X, Y, Z.
    Returns: embryo_id, time, stage_count, cell_name, x, y, z, source_file.

    `embryo_id` is derived from the file path relative to the supplied root, so
    identically named `CellData_0.csv` files in different stage archives remain
    distinct after the archives are unpacked into one parent directory.
    """
    supplied_root = Path(root)
    if supplied_root.is_file() and supplied_root.suffix.lower() == ".zip":
        search_root = Path(tempfile.mkdtemp(prefix="celegans_celldata_"))
        with zipfile.ZipFile(supplied_root, "r") as zf:
            zf.extractall(search_root)
    else:
        search_root = supplied_root

    # If a directory contains archives rather than already-extracted CSVs,
    # unpack each archive into its own subdirectory so repeated CellData file
    # names remain distinguishable.
    direct_files = sorted(
        p for p in search_root.rglob("CellData_*.csv")
        if "__MACOSX" not in str(p)
    ) if search_root.exists() else []
    if not direct_files and search_root.is_dir():
        archives = sorted(
            p for p in search_root.rglob("*.zip")
            if "__MACOSX" not in str(p)
        )
        if archives:
            extracted_root = Path(tempfile.mkdtemp(prefix="celegans_multiarchive_"))
            for archive in archives:
                target = extracted_root / archive.stem
                target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(archive, "r") as zf:
                    zf.extractall(target)
            search_root = extracted_root
    files = discover_celldata_files(search_root)

    rows: list[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path)
        required = {"CellName", "T", "X", "Y", "Z"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns {sorted(missing)}")
        df = df[["CellName", "T", "X", "Y", "Z"]].copy()
        df["CellName"] = df["CellName"].map(_clean_cell_name)
        df["T"] = pd.to_numeric(df["T"], errors="raise").astype(int)
        for col in ["X", "Y", "Z"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna()
        counts = df.groupby("T")["CellName"].nunique().to_dict()
        df["stage_count"] = df["T"].map(counts).astype(int)
        df["embryo_id"] = _embryo_id_from_path(path, search_root)
        try:
            source_rel = path.relative_to(search_root)
        except ValueError:
            source_rel = Path(path.name)
        df["source_file"] = source_rel.as_posix()
        df = df.rename(
            columns={
                "CellName": "cell_name",
                "T": "time",
                "X": "x",
                "Y": "y",
                "Z": "z",
            }
        )
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    return out[
        ["embryo_id", "time", "stage_count", "cell_name", "x", "y", "z", "source_file"]
    ]


def reconstruct_division_events(
    positions: pd.DataFrame,
    mother_daughters: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Reconstruct one mother->two-daughter event per embryo and mother.

    For a given mother, the earliest snapshot containing both daughters is
    found. The latest earlier snapshot containing the mother is used as the
    mother/context snapshot. This avoids duplicate events when a mother is
    visible in multiple pre-division snapshots.
    """
    mother_daughters = mother_daughters or MOTHER_DAUGHTERS
    rows: list[dict[str, object]] = []
    for embryo_id, e in positions.groupby("embryo_id", sort=True):
        e = e.sort_values(["time", "cell_name"])
        by_time = {int(t): g.set_index("cell_name") for t, g in e.groupby("time")}
        times = sorted(by_time)
        for mother, (d1, d2) in mother_daughters.items():
            daughter_times = [
                t for t in times if d1 in by_time[t].index and d2 in by_time[t].index
            ]
            if not daughter_times:
                continue
            td = min(daughter_times)
            mother_times = [t for t in times if t < td and mother in by_time[t].index]
            if not mother_times:
                continue
            tm = max(mother_times)
            mrow = by_time[tm].loc[mother]
            d1row = by_time[td].loc[d1]
            d2row = by_time[td].loc[d2]
            stage_count = int(mrow["stage_count"])
            if stage_count not in STAGE_BLOCK_BY_MOTHER_CELL_COUNT:
                continue
            rows.append(
                {
                    "event_id": f"{embryo_id}|{mother}",
                    "embryo_id": embryo_id,
                    "source_file": str(mrow["source_file"]),
                    "mother_name": mother,
                    "daughter1_name": d1,
                    "daughter2_name": d2,
                    "mother_time": int(tm),
                    "daughter_time": int(td),
                    "mother_stage_count": stage_count,
                    "stage_block": STAGE_BLOCK_BY_MOTHER_CELL_COUNT[stage_count],
                    "mother_x": float(mrow["x"]),
                    "mother_y": float(mrow["y"]),
                    "mother_z": float(mrow["z"]),
                    "daughter1_x": float(d1row["x"]),
                    "daughter1_y": float(d1row["y"]),
                    "daughter1_z": float(d1row["z"]),
                    "daughter2_x": float(d2row["x"]),
                    "daughter2_y": float(d2row["y"]),
                    "daughter2_z": float(d2row["z"]),
                }
            )
    if not rows:
        raise ValueError("No mother-division events could be reconstructed")
    events = pd.DataFrame(rows).sort_values(["embryo_id", "mother_time", "mother_name"])
    events = events.reset_index(drop=True)
    if events["event_id"].duplicated().any():
        raise RuntimeError("Duplicate event_id after reconstruction")
    return attach_targets(events)


def save_canonical_tables(
    positions: pd.DataFrame,
    events: pd.DataFrame,
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    positions.to_csv(out_dir / "cell_positions.csv", index=False)
    events.to_csv(out_dir / "division_events.csv", index=False)

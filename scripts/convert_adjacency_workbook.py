#!/usr/bin/env python3
"""Convert a stage-labeled adjacency workbook into labeled CSV matrices."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

STAGES = {4, 6, 7, 8, 12, 14, 15}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("workbook", help="Input .xlsx workbook")
    p.add_argument("--out", default="data/adj")
    return p.parse_args()


def infer_stage(sheet_name: str) -> int | None:
    nums = [int(x) for x in re.findall(r"\d+", str(sheet_name))]
    for n in nums:
        if n in STAGES:
            return n
    return None


def main() -> None:
    args = parse_args()
    workbook = Path(args.workbook)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sheets = pd.read_excel(workbook, sheet_name=None, engine="openpyxl")
    converted = []
    for name, df in sheets.items():
        stage = infer_stage(name)
        if stage is None:
            continue
        path = out / f"{stage}.csv"
        df.to_csv(path, index=False)
        converted.append((name, stage, path))
    if not converted:
        raise ValueError("No sheet names contained a supported stage label: 4,6,7,8,12,14,15")
    for name, stage, path in converted:
        print(f"{name!r} -> {stage}-cell -> {path}")


if __name__ == "__main__":
    main()

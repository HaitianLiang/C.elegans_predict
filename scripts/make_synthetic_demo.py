#!/usr/bin/env python3
"""Create a small CellData-style demo for smoke-testing the full pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/demo")
    p.add_argument("--n-embryos", type=int, default=80)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def noisy(x, rng, s=0.35):
    return np.asarray(x, float) + rng.normal(scale=s, size=3)


def make_embryo(i: int, rng: np.random.Generator) -> pd.DataFrame:
    # Four-cell baseline.
    base = {
        "ABa": noisy([-15, 0, 4], rng, 0.7),
        "ABp": noisy([2, 0, 7], rng, 0.7),
        "EMS": noisy([0, 2, -8], rng, 0.7),
        "P2": noisy([13, -1, -4], rng, 0.7),
    }
    rows = []
    def add(t, name, xyz):
        rows.append({"CellName": name, "T": t, "X": xyz[0], "Y": xyz[1], "Z": xyz[2]})

    for n, p in base.items():
        add(0, n, p)

    # ABa/ABp divide: 4 -> 6.
    aba_c = base["ABa"] + rng.normal(scale=0.25, size=3)
    abp_c = base["ABp"] + rng.normal(scale=0.25, size=3)
    aba_v = np.array([2.3, 2.2, 3.1]) + rng.normal(scale=0.35, size=3)
    abp_v = np.array([-3.0, 3.3, 0.5]) + rng.normal(scale=0.35, size=3)
    t1 = {
        "ABal": aba_c - aba_v / 2,
        "ABar": aba_c + aba_v / 2,
        "ABpl": abp_c - abp_v / 2,
        "ABpr": abp_c + abp_v / 2,
        "EMS": base["EMS"] + rng.normal(scale=0.25, size=3),
        "P2": base["P2"] + rng.normal(scale=0.25, size=3),
    }
    for n, p in t1.items():
        add(1, n, p)

    # EMS divides: 6 -> 7.
    ems_c = t1["EMS"] + rng.normal(scale=0.20, size=3)
    ems_v = np.array([5.8, -2.2, -1.0]) + rng.normal(scale=0.35, size=3)
    t2 = {k: v + rng.normal(scale=0.12, size=3) for k, v in t1.items() if k != "EMS"}
    t2["MS"] = ems_c - ems_v / 2
    t2["E"] = ems_c + ems_v / 2
    for n, p in t2.items():
        add(2, n, p)

    # P2 divides: 7 -> 8.
    p2_c = t2["P2"] + rng.normal(scale=0.20, size=3)
    p2_v = np.array([2.8, -7.2, 5.7]) + rng.normal(scale=0.40, size=3)
    t3 = {k: v + rng.normal(scale=0.12, size=3) for k, v in t2.items() if k != "P2"}
    t3["C"] = p2_c - p2_v / 2
    t3["P3"] = p2_c + p2_v / 2
    for n, p in t3.items():
        add(3, n, p)
    return pd.DataFrame(rows)


def adjacency_from_positions(df: pd.DataFrame, t: int) -> pd.DataFrame:
    g = df[df["T"] == t].set_index("CellName")[["X", "Y", "Z"]]
    names = list(g.index)
    P = g.to_numpy(float)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    W = np.exp(-D / max(np.median(D[D > 0]), 1e-6))
    np.fill_diagonal(W, 0.0)
    W = W / max(W.max(), 1e-12)
    return pd.DataFrame(W, index=names, columns=names)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    cell_dir = out / "CellData"
    adj_dir = out / "adj"
    cell_dir.mkdir(parents=True, exist_ok=True)
    adj_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    first = None
    for i in range(args.n_embryos):
        df = make_embryo(i, rng)
        if first is None:
            first = df.copy()
        df.to_csv(cell_dir / f"CellData_{i:03d}.csv", index=False)
    # Mother snapshots used in this demo are 4, 6 and 7 cells -> G1, G2, G3.
    for g_index, t in [(1, 0), (2, 1), (3, 2)]:
        mat = adjacency_from_positions(first, t)
        mat.insert(0, "CellName", mat.index)
        mat.to_csv(adj_dir / f"G{g_index}.csv", index=False)
    print(f"Synthetic demo written to {out.resolve()}")
    print(f"CellData root: {cell_dir.resolve()}")
    print(f"Adjacency dir: {adj_dir.resolve()}")


if __name__ == "__main__":
    main()

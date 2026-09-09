"""Lineage-level structural dictionary statistics."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .constants import HALF_TARGETS, ROOT_FAMILY


def topk_half_union(ranked_terms: pd.DataFrame, k: int = 15) -> dict[str, set[str]]:
    use = ranked_terms[
        ranked_terms["target"].isin(HALF_TARGETS) & (ranked_terms["rank"] <= int(k))
    ]
    return {
        str(m): set(g["term"].astype(str))
        for m, g in use.groupby("mother_name")
    }


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return float(len(a & b) / len(union)) if union else 1.0


def lineage_jaccard_summary(
    ranked_terms: pd.DataFrame,
    k: int = 15,
    family_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    family_map = family_map or ROOT_FAMILY
    dictionaries = topk_half_union(ranked_terms, k=k)
    rows = []
    for a, b in combinations(sorted(dictionaries), 2):
        if a not in family_map or b not in family_map:
            continue
        same = family_map[a] == family_map[b]
        rows.append({"mother_a": a, "mother_b": b, "same_root": same, "jaccard": jaccard(dictionaries[a], dictionaries[b])})
    pairs = pd.DataFrame(rows)
    if pairs.empty or pairs["same_root"].nunique() < 2:
        return pairs, {"mean_same_root": np.nan, "mean_different_root": np.nan, "difference": np.nan}
    same = float(pairs.loc[pairs.same_root, "jaccard"].mean())
    diff = float(pairs.loc[~pairs.same_root, "jaccard"].mean())
    return pairs, {"mean_same_root": same, "mean_different_root": diff, "difference": same - diff}


def permutation_test_lineage(
    ranked_terms: pd.DataFrame,
    k: int = 15,
    n_perm: int = 20_000,
    random_state: int = 20260907,
    family_map: dict[str, str] | None = None,
) -> dict[str, float]:
    family_map = family_map or ROOT_FAMILY
    dictionaries = topk_half_union(ranked_terms, k=k)
    mothers = [m for m in sorted(dictionaries) if m in family_map]
    if len(mothers) < 4:
        return {"observed_difference": np.nan, "p_one_sided": np.nan}
    labels = np.array([family_map[m] for m in mothers], dtype=object)

    def stat(lbl: np.ndarray) -> float:
        same, diff = [], []
        for i, j in combinations(range(len(mothers)), 2):
            val = jaccard(dictionaries[mothers[i]], dictionaries[mothers[j]])
            (same if lbl[i] == lbl[j] else diff).append(val)
        if not same or not diff:
            return np.nan
        return float(np.mean(same) - np.mean(diff))

    observed = stat(labels)
    rng = np.random.default_rng(random_state)
    null = []
    for _ in range(int(n_perm)):
        s = stat(rng.permutation(labels))
        if np.isfinite(s):
            null.append(s)
    null = np.asarray(null, dtype=float)
    if null.size == 0:
        return {"observed_difference": observed, "p_one_sided": np.nan}
    p = (1.0 + np.sum(null >= observed)) / (1.0 + len(null))
    return {"observed_difference": float(observed), "p_one_sided": float(p)}


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    m = len(p)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out.tolist()

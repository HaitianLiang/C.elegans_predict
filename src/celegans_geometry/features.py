"""Feature construction for FixedTermOLS and WAEF.

The two model families intentionally use different feature namespaces:

- FixedTermOLS candidates are all columns beginning with ``mother_``.  This
  includes mother-coordinate polynomial terms and weighted-neighborhood terms.
- WAEF primitives are all columns beginning with ``wa_``.

Keeping the namespaces separate prevents WAEF diagnostic primitives from
silently entering the FixedTermOLS selection pool.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .adjacency import WeightedAdjacencyStore

ADJ_SUFFIXES = [
    "xi_xj",
    "xj_minus_xi",
    "xj_minus_xi_pow2",
    "xj_minus_xi_pow3",
    "xi_xj_pow2",
    "xi2_xj",
    "xi_xj2",
    "xi_xj_pow3",
    "xj",
    "xj_pow2",
    "xj_pow3",
]


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    denom = float(weights.sum())
    if values.size == 0 or denom <= 0.0:
        return 0.0
    return float(np.dot(weights, values) / denom)


def fixedterm_feature_columns(features: pd.DataFrame) -> list[str]:
    """Return the source-supported FixedTermOLS candidate columns.

    The project reports describe the FixedTerm candidate library as mother
    polynomials plus weighted neighbor moments, relative displacements, and
    mother-neighbor couplings.  WAEF primitives and adjacency diagnostics are
    therefore excluded from this list even though they are present in the
    feature table for other analyses.
    """
    return [
        c
        for c in features.columns
        if c.startswith("mother_") and pd.api.types.is_numeric_dtype(features[c])
    ]


def feature_block(name: str) -> str:
    """Coarse structural block used for interpretation and RF auditing."""
    if "_adj_xj_minus_xi" in name:
        return "relative_displacement"
    if "_adj_" in name and any(k in name for k in ["xi_xj", "xi2_xj", "xi_xj2"]):
        return "mother_neighbor_coupling"
    if "_adj_" in name:
        return "neighbor_moment"
    if name.startswith("mother_"):
        # Mother-only terms include first-order coordinates and their polynomial
        # / interaction expansions.  The first-order coordinates are kept as a
        # separate interpretive block because the reports analyze them directly.
        if name in {"mother_x", "mother_y", "mother_z"}:
            return "mother_coordinate"
        return "mother_polynomial"
    if name.startswith("wa_"):
        return "waef_primitive"
    if name in {"adjacency_degree", "adjacency_neighbor_count"}:
        return "adjacency_diagnostic"
    return "other"


@dataclass
class FeatureBuilder:
    positions: pd.DataFrame
    adjacency: WeightedAdjacencyStore

    def __post_init__(self) -> None:
        self._contexts: dict[tuple[str, int], pd.DataFrame] = {}
        for (embryo, t), g in self.positions.groupby(["embryo_id", "time"]):
            self._contexts[(str(embryo), int(t))] = g.set_index("cell_name")[["x", "y", "z"]].astype(float)

    @staticmethod
    def _mother_polynomials(x: float, y: float, z: float) -> dict[str, float]:
        """Mother-coordinate terms represented in the project equation family.

        The reports repeatedly use single-axis degree 1--3 terms, pairwise
        interactions/differences, a three-axis product, and a small set of
        higher-order three-axis couplings.  These terms are made explicit here
        rather than generated through an unconstrained polynomial expansion.
        """
        out: dict[str, float] = {}
        for dim, value in [("x", x), ("y", y), ("z", z)]:
            out[f"mother_{dim}"] = value
            out[f"mother_{dim}_{dim}"] = value**2
            out[f"mother_{dim}_{dim}_{dim}"] = value**3

        for d1, v1, d2, v2 in [
            ("x", x, "y", y),
            ("x", x, "z", z),
            ("y", y, "z", z),
        ]:
            prefix = f"mother_2d_couple_{d1}{d2}_"
            out[prefix + "xi_xj"] = v1 * v2
            out[prefix + "xj_minus_xi_pow2"] = (v2 - v1) ** 2
            out[prefix + "xi_xj_pow2"] = (v1 * v2) ** 2
            out[prefix + "xi2_xj"] = v1**2 * v2
            out[prefix + "xi_xj2"] = v1 * v2**2
            out[prefix + "xi_xj_pow3"] = (v1 * v2) ** 3

        xyz = x * y * z
        out["mother_3d_xyz"] = xyz
        out["mother_3d_xyz_pow2"] = xyz**2
        # Higher-order three-axis couplings that appear in the archived
        # equation family.  They are named by the monomial before squaring.
        out["mother_3d_x2yz_pow2"] = (x**2 * y * z) ** 2
        out["mother_3d_xy2z_pow2"] = (x * y**2 * z) ** 2
        out["mother_3d_xyz2_pow2"] = (x * y * z**2) ** 2
        return out

    def _event_row(self, event: pd.Series) -> dict[str, float]:
        x = float(event["mother_x"])
        y = float(event["mother_y"])
        z = float(event["mother_z"])
        out = self._mother_polynomials(x, y, z)

        key = (str(event["embryo_id"]), int(event["mother_time"]))
        if key not in self._contexts:
            raise KeyError(f"Missing mother context for event {event['event_id']}: {key}")
        current = self._contexts[key]
        mother = str(event["mother_name"])
        weights_map = self.adjacency.neighbor_weights(
            int(event["mother_stage_count"]), mother, current.index
        )
        neighbors = [n for n in weights_map if n in current.index]
        weights = np.asarray([weights_map[n] for n in neighbors], dtype=float)

        # Diagnostics are exported but intentionally excluded from FixedTermOLS.
        out["adjacency_degree"] = float(weights.sum())
        out["adjacency_neighbor_count"] = float(len(neighbors))

        primitive_displacements: dict[str, float] = {}
        primitive_spreads: dict[str, float] = {}

        for dim, xi in [("x", x), ("y", y), ("z", z)]:
            vals = (
                current.loc[neighbors, dim].to_numpy(float)
                if neighbors
                else np.zeros(0, dtype=float)
            )
            if vals.size == 0:
                averages = np.zeros(len(ADJ_SUFFIXES), dtype=float)
                primitive_displacements[dim] = 0.0
                primitive_spreads[dim] = 0.0
            else:
                per_neighbor = np.column_stack(
                    [
                        xi * vals,
                        vals - xi,
                        (vals - xi) ** 2,
                        (vals - xi) ** 3,
                        (xi * vals) ** 2,
                        xi**2 * vals,
                        xi * vals**2,
                        (xi * vals) ** 3,
                        vals,
                        vals**2,
                        vals**3,
                    ]
                )
                averages = np.asarray(
                    [
                        weighted_mean(per_neighbor[:, j], weights)
                        for j in range(per_neighbor.shape[1])
                    ]
                )
                primitive_displacements[dim] = weighted_mean(vals - xi, weights)
                primitive_spreads[dim] = weighted_mean((vals - xi) ** 2, weights)
            for suffix, value in zip(ADJ_SUFFIXES, averages):
                out[f"mother_{dim}_adj_{suffix}"] = float(value)

        # WAEF primitives are stored in a separate namespace and never enter the
        # FixedTermOLS candidate pool unless a caller explicitly overrides the
        # feature-selection helper.
        out["wa_mother_x"] = x
        out["wa_mother_y"] = y
        out["wa_mother_z"] = z
        out["wa_degree"] = float(weights.sum())
        for dim in "xyz":
            out[f"wa_delta_{dim}"] = float(primitive_displacements[dim])
            out[f"wa_q_{dim}"] = float(primitive_spreads[dim])
        return out

    def transform(self, events: pd.DataFrame) -> pd.DataFrame:
        rows = [self._event_row(row) for _, row in events.iterrows()]
        X = pd.DataFrame(rows, index=events.index)
        X.insert(0, "event_id", events["event_id"].to_numpy())
        return X

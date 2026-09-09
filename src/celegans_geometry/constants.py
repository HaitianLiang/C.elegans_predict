"""Project-wide biological constants and naming conventions."""

from __future__ import annotations

MOTHER_DAUGHTERS: dict[str, tuple[str, str]] = {
    "ABa": ("ABal", "ABar"),
    "ABp": ("ABpl", "ABpr"),
    "EMS": ("MS", "E"),
    "P2": ("C", "P3"),
    "ABal": ("ABala", "ABalp"),
    "ABar": ("ABara", "ABarp"),
    "ABpl": ("ABpla", "ABplp"),
    "ABpr": ("ABpra", "ABprp"),
    "MS": ("MSa", "MSp"),
    "E": ("Ea", "Ep"),
    "C": ("Ca", "Cp"),
    "ABala": ("ABalaa", "ABalap"),
    "ABalp": ("ABalpa", "ABalpp"),
    "ABara": ("ABaraa", "ABarap"),
    "ABarp": ("ABarpa", "ABarpp"),
    "ABpla": ("ABplaa", "ABplap"),
    "ABplp": ("ABplpa", "ABplpp"),
    "ABpra": ("ABpraa", "ABprap"),
    "ABprp": ("ABprpa", "ABprpp"),
    "P3": ("D", "P4"),
}

ROOT_FAMILY: dict[str, str] = {
    "ABa": "ABa",
    "ABal": "ABa",
    "ABar": "ABa",
    "ABala": "ABa",
    "ABalp": "ABa",
    "ABara": "ABa",
    "ABarp": "ABa",
    "ABp": "ABp",
    "ABpl": "ABp",
    "ABpr": "ABp",
    "ABpla": "ABp",
    "ABplp": "ABp",
    "ABpra": "ABp",
    "ABprp": "ABp",
    "EMS": "EMS",
    "MS": "EMS",
    "E": "EMS",
    "P2": "P2",
    "C": "P2",
    "P3": "P2",
}

TARGETS = [
    "x_mean",
    "x_half",
    "y_mean",
    "y_half",
    "z_mean",
    "z_half",
]
MEAN_TARGETS = ["x_mean", "y_mean", "z_mean"]
HALF_TARGETS = ["x_half", "y_half", "z_half"]

# A mother at these cell counts belongs to the corresponding stage-level block.
STAGE_BLOCK_BY_MOTHER_CELL_COUNT = {
    4: "4-8",
    6: "4-8",
    7: "4-8",
    8: "8-12",
    12: "12-14",
    14: "14-15",
    15: "15-24",
}

G_INDEX_TO_CELL_COUNT = {
    1: 4,
    2: 6,
    3: 7,
    4: 8,
    5: 12,
    6: 14,
    7: 15,
}

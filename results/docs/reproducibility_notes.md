# Reproducibility Notes

The public-facing structure assumed by this README is:

```text
.
├── README.md
├── assets/
│   └── figures/
├── docs/
│   ├── all_fixedterm_equations.md
│   ├── key_results_tables.md
│   └── non_main_explorations.md
└── tables/
```

The modeling pipeline uses precomputed per-stage feature tables and target tables. The key stages are:

1. Build per-event mother/neighborhood features.
2. Construct six targets: three mean coordinates and three half coordinates.
3. Run two-stage selection: fixed-alpha Lasso screening followed by FixedTermOLS reorder.
4. Evaluate in-stage performance, cross-stage transfer, RandomForest feature audit, and few-shot coefficient calibration.
5. Export equations, plots, and tables.

The main paper-style PDF and LaTeX source used to generate the figures are available separately in the companion artifact.

# Local data directory

Biological source data are not included in this repository.

## CellData files

Place extracted CellData files under a local ignored directory, for example:

```text
data/raw/
  source_block_1/
    CellData_000.csv
    CellData_001.csv
  source_block_2/
    CellData_000.csv
    ...
```

Each CellData file must contain:

```text
CellName,T,X,Y,Z
```

A directory containing the original ZIP archives is also accepted. Identically named `CellData_*.csv` files from different archives are kept in separate namespaces during loading.

## Weighted adjacency matrices

A typical local layout is:

```text
data/adj/
  G1.csv   # 4-cell
  G2.csv   # 6-cell
  G3.csv   # 7-cell
  G4.csv   # 8-cell
  G5.csv   # 12-cell
  G6.csv   # 14-cell
  G7.csv   # 15-cell
```

Weights remain continuous; the main analysis does not binarize them.

If the matrices are stored in a stage-labeled Excel workbook:

```bash
python scripts/convert_adjacency_workbook.py adjacency.xlsx --out data/adj
```

## Synthetic software test

```bash
python scripts/make_synthetic_demo.py --out data/demo
python scripts/run_mother_pipeline.py \
  --cell-data-root data/demo/CellData \
  --adj-dir data/demo/adj \
  --out results/demo_mother \
  --lasso-alpha 0.001
```

The synthetic output is only a software smoke test and should not be reported as a biological result.

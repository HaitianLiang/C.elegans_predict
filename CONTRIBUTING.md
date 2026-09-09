# Contributing

Contributions that improve reproducibility, numerical validation, documentation, or tests are welcome.

Before opening a pull request:

1. install the development dependencies with `pip install -e '.[dev]'`;
2. run `pytest`;
3. keep biological source data out of commits unless redistribution is explicitly permitted;
4. document any change that alters feature definitions, model selection, train/test splitting, or reported metrics;
5. add a regression test for numerical changes whenever possible.

For scientific changes, please distinguish clearly between:

- a faithful implementation of the documented method;
- a computational optimization that should be numerically equivalent;
- a new modeling variant or exploratory extension.

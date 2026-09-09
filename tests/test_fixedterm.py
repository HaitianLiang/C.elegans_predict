import numpy as np
import pandas as pd
from celegans_geometry.fixedterm import FixedTermOLS, approx_waic


def test_waic_is_finite():
    y = np.array([1.0, 2.0, 3.0])
    assert np.isfinite(approx_waic(y, y + 0.1, 2))


def test_fixedterm_recovers_sparse_signal():
    rng = np.random.default_rng(1)
    n = 180
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": rng.normal(size=n),
        "d": rng.normal(size=n),
    })
    y = 2.5 * X["a"].to_numpy() - 1.2 * X["c"].to_numpy() + rng.normal(scale=0.05, size=n)
    m = FixedTermOLS(lasso_alpha=1e-4, top_k=4, max_steps=4).fit(X, y)
    assert "a" in m.selected_terms_
    assert m.train_metrics_["r2"] > 0.98

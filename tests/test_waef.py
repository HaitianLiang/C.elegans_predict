import numpy as np
import pandas as pd

from celegans_geometry.waef import WAEFRegressor, WAEF_COLUMNS


def test_waef_mean_fit_and_predict_are_finite():
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(60, len(WAEF_COLUMNS))), columns=WAEF_COLUMNS)
    y = 1.5 * X["wa_mother_x"].to_numpy() - 0.7 * X["wa_delta_y"].to_numpy()
    model = WAEFRegressor(target="x_mean", n_starts=2, max_iter=120, random_state=3).fit(X, y)
    pred = model.predict(X)
    assert np.isfinite(pred).all()
    assert model.train_r2_ > 0.9

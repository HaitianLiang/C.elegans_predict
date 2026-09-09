import numpy as np
from celegans_geometry.features import weighted_mean


def test_weighted_mean():
    x = np.array([1.0, 3.0])
    w = np.array([1.0, 3.0])
    assert abs(weighted_mean(x, w) - 2.5) < 1e-12

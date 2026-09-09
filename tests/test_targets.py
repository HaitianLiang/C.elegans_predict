import numpy as np
from celegans_geometry.targets import daughter_pair_targets


def test_daughter_pair_targets():
    t = daughter_pair_targets(np.array([1.0, 2.0, 3.0]), np.array([5.0, 0.0, -1.0]))
    assert t["x_mean"] == 3.0
    assert t["x_half"] == 2.0
    assert t["y_mean"] == 1.0
    assert t["y_half"] == 1.0
    assert t["z_mean"] == 1.0
    assert t["z_half"] == 2.0

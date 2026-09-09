import numpy as np
from celegans_geometry.axis import dominant_axis, undirected_angle_deg


def test_undirected_axis_sign_invariance():
    a = np.array([1.0, 2.0, 3.0])
    assert undirected_angle_deg(a, -a) < 1e-8


def test_dominant_axis_strength_for_identical_axes():
    axes = np.tile(np.array([[1.0, 0.0, 0.0]]), (20, 1))
    axis, lam1, strength = dominant_axis(axes)
    assert abs(lam1 - 1.0) < 1e-10
    assert abs(strength - 1.0) < 1e-10
    assert undirected_angle_deg(axis, np.array([1.0, 0.0, 0.0])) < 1e-8

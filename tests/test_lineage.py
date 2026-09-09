from celegans_geometry.lineage import benjamini_hochberg


def test_bh_matches_reported_four_depth_example():
    q = benjamini_hochberg([0.3098, 0.3118, 0.2746, 0.0304])
    assert abs(q[-1] - 0.1216) < 1e-12

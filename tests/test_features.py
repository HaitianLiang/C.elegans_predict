import pandas as pd

from celegans_geometry.features import fixedterm_feature_columns


def test_fixedterm_namespace_excludes_waef_and_diagnostics():
    frame = pd.DataFrame(
        {
            "event_id": ["e1"],
            "mother_x": [1.0],
            "mother_x_adj_xj_pow2": [2.0],
            "wa_mother_x": [1.0],
            "wa_q_x": [0.2],
            "adjacency_degree": [3.0],
        }
    )
    cols = fixedterm_feature_columns(frame)
    assert "mother_x" in cols
    assert "mother_x_adj_xj_pow2" in cols
    assert "wa_mother_x" not in cols
    assert "wa_q_x" not in cols
    assert "adjacency_degree" not in cols

import numpy as np
import pytest

from creditrisk.scaling import ScoreScaling, assign_band


def test_base_odds_map_to_base_score_and_pdo_doubles_odds():
    s = ScoreScaling(base_score=600, base_odds=50, pdo=20)
    pd_at_base = 1 / (1 + 50)  # good:bad 50:1
    assert s.score_from_pd(pd_at_base) == pytest.approx(600)
    pd_double = 1 / (1 + 100)  # odds doubled
    assert s.score_from_pd(pd_double) == pytest.approx(620)
    pd_half = 1 / (1 + 25)
    assert s.score_from_pd(pd_half) == pytest.approx(580)


def test_documented_factor_and_offset():
    s = ScoreScaling(600, 50, 20)
    assert s.factor == pytest.approx(28.8539, abs=1e-4)
    assert s.offset == pytest.approx(487.1229, abs=1e-4)


def test_score_pd_roundtrip():
    s = ScoreScaling()
    p = np.array([0.001, 0.01, 0.05, 0.2, 0.6])
    assert np.allclose(s.pd_from_score(s.score_from_pd(p)), p)
    assert np.all(np.diff(s.score_from_pd(p)) < 0)  # higher PD -> lower score


def test_scorecard_points_sum_to_the_scaled_model_score(fitted_models, cfg):
    sc = fitted_models["scorecard"]
    X = fitted_models["valid"]
    scaling = ScoreScaling.from_config(cfg["scorecard"])
    assert np.allclose(sc.score(X), scaling.score_from_pd(sc.predict_pd(X)), atol=1e-6)


def test_scorecard_coefficients_negative_and_points_table_integer(fitted_models):
    sc = fitted_models["scorecard"]
    assert (sc.coefficients < 0).all()
    table = sc.points_table()
    assert set(table["feature"]) == set(sc.selected)
    assert table["points"].dtype.kind == "i"


def test_bands_follow_pd_edges():
    edges, labels = [0.01, 0.05], ["A", "B", "C"]
    assert list(assign_band(np.array([0.001, 0.01, 0.03, 0.2]), edges, labels)) == [
        "A",
        "B",
        "B",
        "C",
    ]

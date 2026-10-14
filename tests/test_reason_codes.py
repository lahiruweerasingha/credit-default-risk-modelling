import numpy as np
import pandas as pd

from creditrisk.features.catalog import FEATURE_BY_NAME


def test_scorecard_reason_codes_are_points_lost_sorted(fitted_models):
    sc = fitted_models["scorecard"]
    X = fitted_models["valid"].head(200)
    reasons = sc.reason_codes(X, k=4)
    lost = sc.max_points() - sc.points(X)
    for i, rc in enumerate(reasons):
        assert len(rc) <= 4
        impacts = [r["impact"] for r in rc]
        assert impacts == sorted(impacts, reverse=True)
        assert all(v > 0 for v in impacts)
        for r in rc:
            assert r["feature"] in sc.selected
            assert r["reason"].startswith(FEATURE_BY_NAME[r["feature"]].reason)
            assert r["reason"].endswith("(not provided)") == (r["value"] is None)
            assert r["impact"] == round(float(lost.iloc[i][r["feature"]]), 2)
        # the first reason is the characteristic with the largest points shortfall
        if rc:
            assert rc[0]["feature"] == lost.iloc[i].idxmax()


def test_best_possible_applicant_has_no_scorecard_reasons(fitted_models):
    sc = fitted_models["scorecard"]
    X = fitted_models["valid"].head(1).copy()
    pts = sc.points(fitted_models["valid"])
    for f in sc.selected:  # pick, for each characteristic, a row value that earns max points
        best_row = pts[f].idxmax()
        X[f] = fitted_models["valid"].loc[best_row, f]
    assert sc.reason_codes(X)[0] == []


def test_xgboost_shap_is_additive_and_reasons_increase_risk(fitted_models):
    xg = fitted_models["xgboost"]
    X = fitted_models["valid"].head(200)
    sv = xg.shap_values(X)
    logit = np.log(xg.predict_pd(X) / (1 - xg.predict_pd(X)))
    assert np.allclose(sv.sum(axis=1), logit, atol=1e-4)
    for i, rc in enumerate(xg.reason_codes(X, k=4)):
        impacts = [r["impact"] for r in rc]
        assert impacts == sorted(impacts, reverse=True)
        assert all(v > 0 for v in impacts)
        assert len(rc) <= 4
        row = sv.drop(columns="bias").iloc[i]
        if rc:
            assert rc[0]["feature"] == row.idxmax()


def test_explain_output_contract(fitted_models):
    for m in fitted_models.values():
        if isinstance(m, pd.DataFrame):
            continue
        out = m.explain(fitted_models["valid"].head(5))
        assert list(out.columns) == ["pd", "score", "band", "reason_codes"]
        assert out["pd"].between(0, 1).all()

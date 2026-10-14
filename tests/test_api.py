"""API contract tests with models fitted on synthetic data (no MLflow server needed)."""

import copy

import pytest
from fastapi.testclient import TestClient

from creditrisk.api.main import app
from creditrisk.api.schemas import EXAMPLE_REQUEST
from creditrisk.api.service import LoadedModel, ModelService
from creditrisk.scaling import ScoreScaling


@pytest.fixture(scope="module")
def client(fitted_models):
    app.state.service = ModelService(
        {
            "scorecard": LoadedModel(fitted_models["scorecard"], "test-scorecard/1"),
            "xgboost": LoadedModel(fitted_models["xgboost"], "test-xgboost/1"),
        }
    )
    with TestClient(app) as c:
        yield c
    app.state.service = None


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["models"] == {"scorecard": "test-scorecard/1", "xgboost": "test-xgboost/1"}


@pytest.mark.parametrize("model", ["scorecard", "xgboost"])
def test_score_contract(client, cfg, model):
    r = client.post(f"/score?model={model}", json=EXAMPLE_REQUEST)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"loan_id", "model", "model_version", "pd", "score", "band", "reason_codes"}
    assert body["loan_id"] == "example-001" and body["model"] == model
    assert 0 < body["pd"] < 1
    assert body["band"] in cfg["bands"]["labels"]
    assert body["score"] == pytest.approx(
        float(ScoreScaling.from_config(cfg["scorecard"]).score_from_pd(body["pd"])), abs=0.06
    )
    assert len(body["reason_codes"]) <= cfg["scorecard"]["n_reason_codes"]
    for rc in body["reason_codes"]:
        assert set(rc) == {"feature", "reason", "value", "impact", "impact_unit"}
        assert rc["impact"] > 0


def test_default_model_is_scorecard(client):
    assert client.post("/score", json=EXAMPLE_REQUEST).json()["model"] == "scorecard"


def test_all_missing_features_still_scores(client):
    r = client.post("/score", json={"features": {}})
    assert r.status_code == 200
    assert 0 < r.json()["pd"] < 1


@pytest.mark.parametrize(
    "patch",
    [
        {"credit_score": 900},
        {"orig_dti": 80},
        {"occupancy_status": "X"},
        {"property_state": "Florida"},
        {"dlq_at_obs": 3},  # 90+ DPD loans are outside the model population
        {"unknown_field": 1},
    ],
)
def test_invalid_inputs_rejected(client, patch):
    req = copy.deepcopy(EXAMPLE_REQUEST)
    req["features"].update(patch)
    assert client.post("/score", json=req).status_code == 422


def test_unknown_model_rejected(client):
    assert client.post("/score?model=random_forest", json=EXAMPLE_REQUEST).status_code == 422

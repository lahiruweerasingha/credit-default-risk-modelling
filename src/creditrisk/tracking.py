"""MLflow tracking and registry helpers.

Both models are logged as pyfunc models whose `predict` returns PD, score, band and reason
codes, registered under the names in the config and tagged with aliases:
`champion` (scorecard) and `challenger` (XGBoost). The API loads them by alias.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient

PIP_REQUIREMENTS = [
    "numpy==1.26.4",
    "pandas==2.2.3",
    "scipy==1.14.1",
    "scikit-learn==1.5.2",
    "xgboost==2.1.3",
]
PACKAGE_DIR = Path(__file__).resolve().parent


class ExplainerModel(mlflow.pyfunc.PythonModel):
    """pyfunc wrapper around a model exposing `explain(df) -> DataFrame` and `input_features`."""

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        out = self.model.explain(model_input)
        out["reason_codes"] = out["reason_codes"].map(json.dumps)
        return out


def setup(cfg: dict[str, Any]) -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    return uri


def log_and_register(model, artifact_path: str, registered_name: str, alias: str) -> int:
    info = mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=ExplainerModel(model),
        code_paths=[str(PACKAGE_DIR)],
        pip_requirements=PIP_REQUIREMENTS,
        registered_model_name=registered_name,
    )
    version = int(info.registered_model_version)
    MlflowClient().set_registered_model_alias(registered_name, alias, str(version))
    return version


def load_by_alias(name: str, alias: str):
    """Load a registered model and return the underlying explainer object."""
    pyfunc = mlflow.pyfunc.load_model(f"models:/{name}@{alias}")
    return pyfunc.unwrap_python_model().model

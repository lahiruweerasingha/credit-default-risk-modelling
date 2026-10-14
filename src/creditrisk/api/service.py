"""Model loading and scoring for the API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from creditrisk.features.catalog import ALL_FEATURES, CATEGORICAL_FEATURES


@dataclass
class LoadedModel:
    model: Any  # exposes explain(df) and input_features
    version: str


class ModelService:
    def __init__(self, models: dict[str, LoadedModel]):
        self.models = models

    @classmethod
    def from_registry(cls, cfg: dict[str, Any]) -> ModelService:
        from mlflow.tracking import MlflowClient

        from creditrisk import tracking

        tracking.setup(cfg)
        client = MlflowClient()
        out = {}
        for key, name_key, alias in [
            ("scorecard", "scorecard_model_name", "champion"),
            ("xgboost", "xgboost_model_name", "challenger"),
        ]:
            name = cfg["mlflow"][name_key]
            mv = client.get_model_version_by_alias(name, alias)
            out[key] = LoadedModel(tracking.load_by_alias(name, alias), f"{name}/{mv.version}")
        return cls(out)

    def score(self, model_key: str, features: dict[str, Any]) -> dict[str, Any]:
        loaded = self.models[model_key]
        row = {f: features.get(f) for f in ALL_FEATURES}
        df = pd.DataFrame([row])
        for f in ALL_FEATURES:
            if f in CATEGORICAL_FEATURES:
                df[f] = df[f].astype("object")
            else:
                df[f] = pd.to_numeric(df[f], errors="coerce").astype(float)
        out = loaded.model.explain(df).iloc[0]
        return {
            "model": model_key,
            "model_version": loaded.version,
            "pd": float(out["pd"]),
            "score": float(np.round(out["score"], 1)),
            "band": str(out["band"]),
            "reason_codes": out["reason_codes"],
        }

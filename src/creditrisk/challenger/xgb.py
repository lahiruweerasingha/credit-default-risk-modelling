"""Challenger: single-node XGBoost with exact TreeSHAP explanations.

Why single-node (see docs/decisions/0002): the loan-level table fits comfortably in memory,
distributed XGBoost adds coordination overhead for no accuracy gain here, and TreeSHAP needs
the native booster. SHAP values come from XGBoost's built-in `pred_contribs` (exact TreeSHAP,
supports native categorical splits); the `shap` package is used only for plotting.

No class re-weighting is applied, so predicted probabilities stay calibrated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from creditrisk.features.catalog import CATEGORICAL_FEATURES, reason_text
from creditrisk.scaling import ScoreScaling, assign_band


class XGBChallenger:
    def __init__(self, xgb_cfg: dict, sc_cfg: dict, band_cfg: dict, features: list[str]):
        self.cfg = xgb_cfg
        self.band_cfg = band_cfg
        self.n_reasons = sc_cfg["n_reason_codes"]
        self.scaling = ScoreScaling.from_config(sc_cfg)
        self.features = list(features)
        self.categories: dict[str, list[str]] = {}
        self.booster: xgb.Booster | None = None
        self.params: dict[str, Any] = {}
        self.trials: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ data prep
    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for f in self.features:
            if f in CATEGORICAL_FEATURES:
                vals = X[f].astype("object").where(X[f].notna(), None)
                out[f] = pd.Categorical(vals, categories=self.categories[f])
            else:
                out[f] = pd.to_numeric(X[f], errors="coerce").astype(float)
        return out

    def _dmatrix(self, X: pd.DataFrame, y=None) -> xgb.DMatrix:
        return xgb.DMatrix(self._frame(X), label=y, enable_categorical=True)

    # ------------------------------------------------------------------ fitting
    def fit(self, X: pd.DataFrame, y, X_val: pd.DataFrame, y_val) -> XGBChallenger:
        for f in self.features:
            if f in CATEGORICAL_FEATURES:
                self.categories[f] = sorted(X[f].dropna().astype(str).unique().tolist())
        dtrain, dval = self._dmatrix(X, y), self._dmatrix(X_val, y_val)
        rng = np.random.default_rng(self.cfg["seed"])
        best = (-1.0, None, None)
        for t in range(self.cfg["n_trials"]):
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "hist",
                "max_cat_to_onehot": 1,
                "seed": int(self.cfg["seed"]),
                "nthread": -1,
                "max_depth": int(rng.integers(3, 7)),
                "eta": float(rng.choice([0.02, 0.03, 0.05, 0.08])),
                "min_child_weight": float(rng.choice([1, 5, 10, 20, 50])),
                "subsample": float(rng.uniform(0.7, 1.0)),
                "colsample_bytree": float(rng.uniform(0.6, 1.0)),
                "lambda": float(rng.choice([1.0, 3.0, 10.0])),
                "alpha": float(rng.choice([0.0, 0.5, 2.0])),
            }
            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=self.cfg["max_rounds"],
                evals=[(dval, "validation")],
                early_stopping_rounds=self.cfg["early_stopping_rounds"],
                verbose_eval=False,
            )
            auc = roc_auc_score(
                y_val, booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
            )
            self.trials.append(
                {"trial": t, **params, "best_iteration": booster.best_iteration, "val_auc": auc}
            )
            if auc > best[0]:
                best = (auc, params, booster)
        _, self.params, self.booster = best
        return self

    # ------------------------------------------------------------------ scoring
    @property
    def _iter_range(self) -> tuple[int, int]:
        return (0, self.booster.best_iteration + 1)

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self._dmatrix(X), iteration_range=self._iter_range)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaling.score_from_pd(self.predict_pd(X))

    def shap_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contributions to the log-odds of default (last column is the bias)."""
        contribs = self.booster.predict(
            self._dmatrix(X), pred_contribs=True, iteration_range=self._iter_range
        )
        return pd.DataFrame(contribs, index=X.index, columns=[*self.features, "bias"])

    def reason_codes(self, X: pd.DataFrame, k: int | None = None) -> list[list[dict]]:
        """Top-k features pushing the PD up (positive SHAP on the log-odds scale)."""
        k = k or self.n_reasons
        sv = self.shap_values(X).drop(columns="bias")
        out = []
        for i in range(len(sv)):
            row = sv.iloc[i]
            row = row[row > 0].sort_values(ascending=False).head(k)
            out.append(
                [
                    {
                        "feature": f,
                        "reason": reason_text(f, _jsonable(X.iloc[i][f])),
                        "value": _jsonable(X.iloc[i][f]),
                        "impact": round(float(v), 4),
                        "impact_unit": "log-odds of default (SHAP)",
                    }
                    for f, v in row.items()
                ]
            )
        return out

    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        pd_ = self.predict_pd(X)
        return pd.DataFrame(
            {
                "pd": pd_,
                "score": np.round(self.scaling.score_from_pd(pd_), 1),
                "band": assign_band(pd_, self.band_cfg["pd_edges"], self.band_cfg["labels"]),
                "reason_codes": self.reason_codes(X),
            },
            index=X.index,
        )

    def feature_importance(self) -> pd.Series:
        gain = self.booster.get_score(importance_type="total_gain")
        return pd.Series({f: gain.get(f, 0.0) for f in self.features}).sort_values(ascending=False)

    @property
    def input_features(self) -> list[str]:
        return list(self.features)


def _jsonable(v):
    if v is None or (np.ndim(v) == 0 and pd.isna(v)):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v

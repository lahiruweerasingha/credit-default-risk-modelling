"""Champion model: WoE logistic-regression scorecard scaled to points.

Steps (all fitted on the training split; only the regularisation strength C is chosen on
validation):
    1. Monotonic WoE binning of every candidate feature.
    2. Keep features with IV >= iv_min; flag IV > iv_suspicious for review.
    3. Greedy correlation filter on WoE values (|rho| <= corr_max), highest IV first,
       capped at max_features.
    4. Logistic regression on WoE; C chosen by validation AUC.
    5. Sign check: with WoE = ln(%good/%bad) every coefficient must be negative. Features with
       a non-negative coefficient are dropped one at a time and the model refitted.
    6. Points: points_ij = -(beta_i * woe_ij + alpha / n) * factor + offset / n.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from creditrisk.features.catalog import FEATURE_BY_NAME, reason_text
from creditrisk.scaling import ScoreScaling, assign_band
from creditrisk.scorecard.binning import WoEBinner


class Scorecard:
    def __init__(self, sc_cfg: dict[str, Any], band_cfg: dict[str, Any], features: list[str]):
        self.cfg = sc_cfg
        self.band_cfg = band_cfg
        self.candidates = list(features)
        self.scaling = ScoreScaling.from_config(sc_cfg)
        self.binners: dict[str, WoEBinner] = {}
        self.selected: list[str] = []
        self.model: LogisticRegression | None = None
        self.iv_table: pd.DataFrame | None = None
        self.selection_log: list[str] = []
        self.best_c: float | None = None

    # ------------------------------------------------------------------ fitting
    def fit(
        self, X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray
    ) -> Scorecard:
        c = self.cfg
        for name in self.candidates:
            kind = FEATURE_BY_NAME[name].kind
            self.binners[name] = WoEBinner(
                name, kind, c["max_prebins"], c["min_bin_share"], c["min_missing_count"]
            ).fit(X[name], y)

        iv = pd.DataFrame(
            {"feature": list(self.binners), "iv": [b.iv for b in self.binners.values()]}
        ).sort_values("iv", ascending=False, ignore_index=True)
        iv["passes_iv_min"] = iv["iv"] >= c["iv_min"]
        iv["suspicious"] = iv["iv"] > c["iv_suspicious"]
        self.iv_table = iv
        for f, v in iv.loc[iv["suspicious"], ["feature", "iv"]].itertuples(index=False):
            self.selection_log.append(
                f"{f}: IV {v:.3f} > {c['iv_suspicious']} - flagged for leakage review "
                "(uses data up to month 12 only, enforced by test_leakage)"
            )

        woe_train = self._woe(X, list(iv.loc[iv["passes_iv_min"], "feature"]))
        chosen: list[str] = []
        for f in iv.loc[iv["passes_iv_min"], "feature"]:
            if len(chosen) >= c["max_features"]:
                self.selection_log.append(f"{f}: dropped, max_features reached")
                continue
            if chosen:
                corr = woe_train[chosen].corrwith(woe_train[f]).abs()
                if (corr > c["corr_max"]).any():
                    self.selection_log.append(
                        f"{f}: dropped, |corr| {corr.max():.2f} with {corr.idxmax()}"
                    )
                    continue
            chosen.append(f)

        woe_val = self._woe(X_val, chosen)
        best_auc, best_c = -1.0, None
        for cval in c["c_grid"]:
            m = _logit(cval).fit(woe_train[chosen], y)
            auc = roc_auc_score(y_val, m.predict_proba(woe_val[chosen])[:, 1])
            if auc > best_auc:
                best_auc, best_c = auc, cval
        self.best_c = best_c

        while True:
            m = _logit(best_c).fit(woe_train[chosen], y)
            coefs = pd.Series(m.coef_[0], index=chosen)
            if (coefs < 0).all() or len(chosen) == 1:
                break
            worst = coefs.idxmax()
            self.selection_log.append(f"{worst}: dropped, coefficient sign {coefs[worst]:+.3f}")
            chosen.remove(worst)
        self.selected, self.model = chosen, m
        return self

    def _woe(self, X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        return pd.DataFrame({f: self.binners[f].transform(X[f]) for f in features}, index=X.index)

    # ------------------------------------------------------------------ scoring
    @property
    def coefficients(self) -> pd.Series:
        return pd.Series(self.model.coef_[0], index=self.selected)

    @property
    def intercept(self) -> float:
        return float(self.model.intercept_[0])

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._woe(X, self.selected))[:, 1]

    def points(self, X: pd.DataFrame) -> pd.DataFrame:
        """Points per characteristic per applicant (unrounded)."""
        woe = self._woe(X, self.selected)
        n = len(self.selected)
        s = self.scaling
        beta = self.coefficients
        return -(woe * beta + self.intercept / n) * s.factor + s.offset / n

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.points(X).sum(axis=1).to_numpy()

    def points_table(self) -> pd.DataFrame:
        """The published scorecard: points for every attribute (bin) of every characteristic."""
        n = len(self.selected)
        s = self.scaling
        rows = []
        for f in self.selected:
            t = self.binners[f].table
            pts = -(t["woe"] * self.coefficients[f] + self.intercept / n) * s.factor + s.offset / n
            for (_, r), p in zip(t.iterrows(), pts, strict=True):
                if r["bin"] == "Unseen" or (r["bin"] == "Missing" and r["count"] == 0):
                    continue
                rows.append(
                    {
                        "feature": f,
                        "bin": r["bin"],
                        "count": r["count"],
                        "bad_rate": r["bad_rate"],
                        "woe": r["woe"],
                        "points": int(round(p)),
                    }
                )
        return pd.DataFrame(rows)

    def max_points(self) -> pd.Series:
        n = len(self.selected)
        s = self.scaling
        out = {}
        for f in self.selected:
            t = self.binners[f].table
            observed = t[(t["count"] > 0) & (t["bin"] != "Unseen")]
            pts = (
                -(observed["woe"] * self.coefficients[f] + self.intercept / n) * s.factor
                + s.offset / n
            )
            out[f] = float(pts.max())
        return pd.Series(out)

    def reason_codes(self, X: pd.DataFrame, k: int | None = None) -> list[list[dict]]:
        """Top-k characteristics where the applicant lost the most points vs the best attribute."""
        k = k or self.cfg["n_reason_codes"]
        lost = self.max_points() - self.points(X)
        out = []
        for i, row in enumerate(lost.itertuples(index=False)):
            ser = pd.Series(row, index=lost.columns)
            ser = ser[ser > 0.5].sort_values(ascending=False).head(k)
            out.append(
                [
                    {
                        "feature": f,
                        "reason": reason_text(f, _jsonable(X.iloc[i][f])),
                        "value": _jsonable(X.iloc[i][f]),
                        "impact": round(float(v), 2),
                        "impact_unit": "points below best attribute",
                    }
                    for f, v in ser.items()
                ]
            )
        return out

    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        pd_ = self.predict_pd(X)
        return pd.DataFrame(
            {
                "pd": pd_,
                "score": np.round(self.score(X), 1),
                "band": assign_band(pd_, self.band_cfg["pd_edges"], self.band_cfg["labels"]),
                "reason_codes": self.reason_codes(X),
            },
            index=X.index,
        )

    @property
    def input_features(self) -> list[str]:
        return list(self.selected)


def _logit(c: float) -> LogisticRegression:
    return LogisticRegression(C=c, solver="lbfgs", max_iter=2000)


def _jsonable(v):
    if v is None or (np.ndim(v) == 0 and pd.isna(v)):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v

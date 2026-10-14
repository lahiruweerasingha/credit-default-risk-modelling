"""Score scaling shared by every model.

    score = offset + factor * ln(odds_good),   odds_good = (1 - PD) / PD
    factor = PDO / ln(2)
    offset = base_score - factor * ln(base_odds)

With the defaults (600 at 50:1, PDO 20): factor = 28.854, offset = 487.123. A loan with
good:bad odds of 50:1 scores 600, 100:1 scores 620, 25:1 scores 580.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-9


@dataclass(frozen=True)
class ScoreScaling:
    base_score: float = 600.0
    base_odds: float = 50.0
    pdo: float = 20.0

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2.0)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)

    def score_from_pd(self, pd_: np.ndarray | float) -> np.ndarray:
        p = np.clip(np.asarray(pd_, dtype=float), EPS, 1 - EPS)
        return self.offset + self.factor * np.log((1 - p) / p)

    def pd_from_score(self, score: np.ndarray | float) -> np.ndarray:
        s = np.asarray(score, dtype=float)
        return 1.0 / (1.0 + np.exp((s - self.offset) / self.factor))

    @classmethod
    def from_config(cls, sc: dict) -> ScoreScaling:
        return cls(float(sc["base_score"]), float(sc["base_odds"]), float(sc["pdo"]))


def assign_band(pd_: np.ndarray, edges: list[float], labels: list[str]) -> np.ndarray:
    """Risk band from PD. `edges` are PD upper bounds of all but the last band."""
    if len(labels) != len(edges) + 1:
        raise ValueError("need exactly one more label than edges")
    idx = np.searchsorted(np.asarray(edges), np.asarray(pd_, dtype=float), side="right")
    return np.asarray(labels, dtype=object)[idx]

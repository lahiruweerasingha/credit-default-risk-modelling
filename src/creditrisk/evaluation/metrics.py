"""Validation metrics used by a bank model-validation team."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Kolmogorov-Smirnov: max distance between the score CDFs of bads and goods."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def discrimination(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    auc = roc_auc_score(y, p)
    return {
        "n": int(len(y)),
        "n_default": int(np.sum(y)),
        "default_rate": float(np.mean(y)),
        "mean_pd": float(np.mean(p)),
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y, p),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Observed vs predicted default rate in PD deciles."""
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    t = df.groupby("bin").agg(n=("y", "size"), mean_pd=("p", "mean"), observed=("y", "mean"))
    return t.reset_index()


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index with bins on the expected (reference) quantiles.

    Rule of thumb: < 0.10 stable, 0.10-0.25 some shift, > 0.25 significant shift.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)[1:-1]))
    e = np.bincount(np.searchsorted(edges, expected, side="right"), minlength=len(edges) + 1)
    a = np.bincount(np.searchsorted(edges, actual, side="right"), minlength=len(edges) + 1)
    e = np.clip(e / e.sum(), 1e-6, None)
    a = np.clip(a / a.sum(), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def band_table(y: np.ndarray, p: np.ndarray, bands: np.ndarray, labels: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": p, "band": bands})
    t = (
        df.groupby("band")
        .agg(
            n=("y", "size"),
            defaults=("y", "sum"),
            default_rate=("y", "mean"),
            mean_pd=("p", "mean"),
        )
        .reindex(labels)
        .fillna({"n": 0, "defaults": 0})
    )
    t["share"] = t["n"] / t["n"].sum()
    return t.reset_index().rename(columns={"index": "band"})


def cutoff_table(y: np.ndarray, score: np.ndarray, cutoffs: np.ndarray) -> pd.DataFrame:
    """Approve when score >= cut-off. Approval rate vs bad rate among approved / declined."""
    rows = []
    total_bad = max(int(np.sum(y)), 1)
    for c in cutoffs:
        appr = score >= c
        rows.append(
            {
                "cutoff_score": float(c),
                "approval_rate": float(appr.mean()),
                "bad_rate_approved": float(y[appr].mean()) if appr.any() else np.nan,
                "bad_rate_declined": float(y[~appr].mean()) if (~appr).any() else np.nan,
                "share_of_bads_declined": float(y[~appr].sum() / total_bad),
            }
        )
    return pd.DataFrame(rows)

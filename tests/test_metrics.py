import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from creditrisk.evaluation.metrics import (
    band_table,
    cutoff_table,
    discrimination,
    ks_statistic,
    paired_bootstrap,
    psi,
    stratified_resample,
)
from creditrisk.split import assign_splits


def test_ks_perfect_and_random():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20000)
    assert ks_statistic(y, rng.random(20000)) < 0.05


def test_gini_is_two_auc_minus_one():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.3 + rng.random(2000) * 0.7, 0, 1)
    m = discrimination(y, p)
    assert m["gini"] == pytest.approx(2 * m["auc"] - 1)


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 50000)
    assert psi(a, rng.normal(0, 1, 50000)) < 0.01
    assert psi(a, rng.normal(1, 1, 50000)) > 0.25


def test_cutoff_table_monotone_approval():
    rng = np.random.default_rng(3)
    s = rng.normal(600, 50, 5000)
    y = (rng.random(5000) < 1 / (1 + np.exp((s - 520) / 30))).astype(int)
    t = cutoff_table(y, s, np.array([550, 600, 650]))
    assert t["approval_rate"].is_monotonic_decreasing
    assert t["bad_rate_approved"].is_monotonic_decreasing


def test_band_table_shares_sum_to_one():
    y = np.array([0, 1, 0, 0])
    t = band_table(
        y, np.array([0.1, 0.2, 0.3, 0.4]), np.array(["A", "B", "B", "C"]), ["A", "B", "C"]
    )
    assert t["share"].sum() == pytest.approx(1.0)
    assert list(t["band"]) == ["A", "B", "C"]


def test_splits_are_disjoint_stratified_and_oot_held_out(cfg):
    rng = np.random.default_rng(4)
    n = 10000
    df = pd.DataFrame(
        {"vintage": rng.choice([2010, 2015], n), "target": (rng.random(n) < 0.05).astype(int)}
    )
    split = assign_splits(df, cfg)
    assert set(split[df["vintage"] == 2015]) == {"oot"}
    dev = split[df["vintage"] == 2010]
    assert set(dev) == {"train", "validation", "test"}
    shares = dev.value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.6, abs=0.01)
    rates = df[df["vintage"] == 2010].groupby(dev)["target"].mean()
    assert rates.max() - rates.min() < 0.005


def _two_models(n=6000, rate=0.05, strength_a=1.5, strength_b=1.5, seed=7):
    """Latent risk drives defaults; each model sees it plus its OWN independent noise."""
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(np.log(rate / (1 - rate)) + 1.2 * risk)))).astype(int)
    p_a = strength_a * risk + rng.normal(size=n)
    p_b = strength_b * risk + rng.normal(size=n)
    return y, p_a, p_b


def test_stratified_resample_keeps_default_count():
    y = np.array([1] * 37 + [0] * 963)
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert y[stratified_resample(y, rng)].sum() == 37


def test_bootstrap_is_deterministic_with_seed():
    y, a, b = _two_models()
    kw = dict(preds={"a": a, "b": b}, gaps=[("a", "b")], n_resamples=50, seed=3)
    assert paired_bootstrap(y, **kw) == paired_bootstrap(y, **kw)


def test_gap_interval_contains_zero_for_equally_skilled_but_different_models():
    y, a, b = _two_models()
    assert not np.allclose(a, b)  # genuinely different predictions
    out = paired_bootstrap(y, {"a": a, "b": b}, gaps=[("a", "b")], n_resamples=300, seed=1)
    lo, hi = out["gaps"]["gini_a_minus_b"]["interval"]
    assert lo < 0 < hi
    assert out["gaps"]["gini_a_minus_b"]["excludes_zero"] is False


def test_gap_interval_detects_a_clearly_better_model():
    y, strong, weak = _two_models(strength_a=3.0, strength_b=0.5)
    out = paired_bootstrap(y, {"s": strong, "w": weak}, gaps=[("s", "w")], n_resamples=300)
    lo, _ = out["gaps"]["gini_s_minus_w"]["interval"]
    assert lo > 0 and out["gaps"]["gini_s_minus_w"]["excludes_zero"] is True


def test_bootstrap_interval_brackets_point_estimate():
    y, a, b = _two_models()
    out = paired_bootstrap(y, {"a": a}, gaps=[], n_resamples=300)
    lo, hi = out["models"]["a"]["gini"]
    assert lo < 2 * roc_auc_score(y, a) - 1 < hi

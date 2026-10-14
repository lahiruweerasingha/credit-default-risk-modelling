import numpy as np
import pandas as pd
import pytest

from creditrisk.scorecard.binning import OTHER, WoEBinner


def _noisy_u_shape(n=20000, seed=0):
    """Bad rate is mostly increasing in x but with a dip, so raw quantile bins are NOT monotonic."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 100, n)
    p = 0.02 + 0.10 * (x / 100) + 0.05 * np.exp(-((x - 20) ** 2) / 20)
    y = (rng.random(n) < p).astype(int)
    return x, y


def test_numeric_bins_are_strictly_monotonic():
    x, y = _noisy_u_shape()
    b = WoEBinner("x", "numeric", max_prebins=20, min_bin_share=0.05).fit(pd.Series(x), y)
    assert b.trend == "increasing"
    assert b.is_monotonic()
    regular = b.table.iloc[: len(b.edges) + 1]
    assert np.all(np.diff(regular["bad_rate"]) > 0)
    # WoE = ln(%good/%bad) moves opposite to the bad rate
    assert np.all(np.diff(regular["woe"]) < 0)


def test_decreasing_trend_detected():
    x, y = _noisy_u_shape()
    b = WoEBinner("x", "numeric").fit(pd.Series(-x), y)
    assert b.trend == "decreasing"
    assert b.is_monotonic()


def test_min_bin_share_respected():
    x, y = _noisy_u_shape()
    b = WoEBinner("x", "numeric", min_bin_share=0.08).fit(pd.Series(x), y)
    shares = b.table["count"].iloc[: len(b.edges) + 1] / len(x)
    assert (shares >= 0.08).all()


def test_missing_values_get_own_bin_or_neutral_woe():
    x, y = _noisy_u_shape()
    x = x.copy()
    x[:1000] = np.nan
    b = WoEBinner("x", "numeric", min_missing_count=50).fit(pd.Series(x), y)
    missing_row = b.table[b.table["bin"] == "Missing"].iloc[0]
    assert missing_row["count"] == 1000
    assert b.transform(pd.Series([np.nan]))[0] == pytest.approx(missing_row["woe"])

    few = x.copy()
    few[:] = np.where(np.isnan(few), 1.0, few)
    few[:10] = np.nan
    b2 = WoEBinner("x", "numeric", min_missing_count=50).fit(pd.Series(few), y)
    assert b2.transform(pd.Series([np.nan]))[0] == 0.0


def test_iv_positive_for_predictive_and_zero_for_constant():
    x, y = _noisy_u_shape()
    assert WoEBinner("x", "numeric").fit(pd.Series(x), y).iv > 0.02
    const = WoEBinner("c", "numeric").fit(pd.Series(np.ones_like(x)), y)
    assert const.iv == pytest.approx(0.0, abs=1e-9)


def test_categorical_rare_levels_pooled_and_unseen_mapped_to_other():
    rng = np.random.default_rng(1)
    cats = rng.choice(["A", "B", "C", "D"], 10000, p=[0.6, 0.35, 0.03, 0.02])
    y = (rng.random(10000) < np.where(cats == "A", 0.03, 0.08)).astype(int)
    b = WoEBinner("cat", "categorical", min_bin_share=0.05).fit(pd.Series(cats), y)
    assert b.category_map["C"] == b.category_map["D"] == b.category_map[OTHER]
    other_woe = b.table.loc[b.category_map[OTHER], "woe"]
    assert b.transform(pd.Series(["never-seen"]))[0] == pytest.approx(other_woe)


def test_rare_but_predictive_state_keeps_its_own_bin():
    """Regression test: ~3% of loans delinquent with 10x the bad rate must not be merged away
    by the 5% minimum bin share (this collapsed real delinquency features to IV = 0)."""
    rng = np.random.default_rng(5)
    n = 50000
    months_dlq = np.where(rng.random(n) < 0.03, rng.integers(1, 4, n), 0)
    y = (rng.random(n) < np.where(months_dlq > 0, 0.08, 0.008)).astype(int)
    b = WoEBinner("n_months_dlq", "numeric", min_bin_share=0.05).fit(pd.Series(months_dlq), y)
    assert len(b.edges) >= 1
    assert b.iv > 0.3
    assert b.is_monotonic()

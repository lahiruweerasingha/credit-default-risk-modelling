import numpy as np
import pandas as pd
import pytest

from creditrisk.evaluation.metrics import (
    band_table,
    cutoff_table,
    discrimination,
    ks_statistic,
    psi,
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

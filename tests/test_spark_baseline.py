"""Spark MLlib baseline: reads the loan-level parquet in Spark, respects the saved splits."""

import copy

import pytest

pytest.importorskip("pyspark")

from creditrisk.baseline.spark_lr import fit_and_score  # noqa: E402
from creditrisk.split import assign_splits  # noqa: E402
from tests.fixtures.synthetic import make_loan_level  # noqa: E402

pytestmark = pytest.mark.spark


def test_baseline_scores_every_split_from_parquet(spark, cfg, tmp_path):
    ll = make_loan_level(n=3000, seed=3)
    ll.loc[ll.index % 3 == 0, "vintage"] = 2015  # some out-of-time loans
    processed = tmp_path / "processed"
    (processed / "loan_level").mkdir(parents=True)
    ll.to_parquet(processed / "loan_level" / "part-0.parquet", index=False)
    ll["split"] = assign_splits(ll, cfg)
    splits_path = tmp_path / "splits.parquet"
    ll[["loan_seq_no", "split"]].to_parquet(splits_path, index=False)

    c = copy.deepcopy(cfg)
    c["data"]["processed_dir"] = str(processed)
    c["baseline"]["reg_grid"] = [0.01]
    _, preds, grid = fit_and_score(spark, c, splits_path)

    assert set(preds) == {"train", "validation", "test", "oot"}
    for split, p in preds.items():
        expected = set(ll.loc[ll["split"] == split, "loan_seq_no"])
        assert set(p["loan_seq_no"]) == expected
        assert p["pd"].between(0, 1).all()
    assert grid[0]["val_auc"] > 0.6

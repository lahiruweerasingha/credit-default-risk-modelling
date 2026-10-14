"""Leakage guard.

1. Static: the feature SQL never references a label column.
2. Dynamic: overwriting every performance record AFTER the observation month (forcing all loans
   into serious delinquency) must leave every feature value unchanged, while the labels change.
   If any feature query read from the outcome window this test fails.
"""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyspark")

from pyspark.sql import functions as F  # noqa: E402

from creditrisk.features import build  # noqa: E402
from creditrisk.features.catalog import ALL_FEATURES  # noqa: E402

pytestmark = pytest.mark.spark


def test_feature_sql_never_mentions_the_target():
    assert build._check_sql_has_no_label_reference(build.FEATURE_SQL_FILES) == []


def test_label_detector_catches_a_target_reference(monkeypatch):
    original = build.load_sql
    monkeypatch.setattr(
        build,
        "load_sql",
        lambda name, **kw: "SELECT AVG(target) FROM x"
        if name == "behaviour"
        else original(name, **kw),
    )
    assert build._check_sql_has_no_label_reference(build.FEATURE_SQL_FILES) == ["behaviour"]


def _features_and_labels(spark, cfg, raw_orig, raw_svcg):
    build.register_raw(raw_orig, raw_svcg)
    feats = build.build_features(spark, cfg).toPandas()
    labels = build.build_labels(spark, cfg).toPandas()
    key = "loan_seq_no"
    return feats.sort_values(key).reset_index(drop=True), labels.sort_values(key).reset_index(
        drop=True
    )


def test_features_invariant_to_outcome_window(spark, cfg, raw_dir: Path):
    raw_orig, raw_svcg = build.read_raw_text(spark, raw_dir, [2010, 2015])
    feats_a, labels_a = _features_and_labels(spark, cfg, raw_orig, raw_svcg)
    assert len(feats_a) > 100

    # Month index of the observation point, computed from the raw first-payment date.
    obs_age = cfg["behavioural"]["observation_age_months"]
    fp = raw_orig.select(
        F.trim("loan_seq_no").alias("k"),
        (
            F.substring("first_payment_date", 1, 4).cast("int") * 12
            + F.substring("first_payment_date", 5, 2).cast("int")
            - 1
            + obs_age
            - 1
        ).alias("obs_idx"),
    )
    per = (
        F.substring("monthly_reporting_period", 1, 4).cast("int") * 12
        + F.substring("monthly_reporting_period", 5, 2).cast("int")
        - 1
    )
    joined = raw_svcg.join(fp, F.trim(raw_svcg.loan_seq_no) == fp.k, "left")
    future = per > F.col("obs_idx")
    tampered = (
        joined.withColumn(
            "current_delinquency_status",
            F.when(future, F.lit("6")).otherwise(F.col("current_delinquency_status")),
        )
        .withColumn(
            "current_actual_upb",
            F.when(future, F.lit("1.00")).otherwise(F.col("current_actual_upb")),
        )
        .withColumn(
            "modification_flag", F.when(future, F.lit("Y")).otherwise(F.col("modification_flag"))
        )
        .drop("k", "obs_idx")
    )
    feats_b, labels_b = _features_and_labels(spark, cfg, raw_orig, tampered)

    cols = ["loan_seq_no", *ALL_FEATURES]
    pd.testing.assert_frame_equal(feats_a[cols], feats_b[cols], check_exact=False, rtol=1e-12)
    # sanity: the tampering really did reach the outcome window
    assert labels_b["target"].sum() > labels_a["target"].sum()


def test_population_excludes_loans_already_in_default(spark, cfg, raw_dir: Path):
    raw_orig, raw_svcg = build.read_raw_text(spark, raw_dir, [2010, 2015])
    build.register_raw(raw_orig, raw_svcg)
    feats = build.build_features(spark, cfg).toPandas()
    assert (feats["dlq_at_obs"] < cfg["behavioural"]["default_dpd_status"]).all()
    # ARM loans are out of scope
    arm = raw_orig.filter(F.trim("amortization_type") == "ARM").select("loan_seq_no").toPandas()
    assert not set(arm["loan_seq_no"].str.strip()) & set(feats["loan_seq_no"])

"""Hand-checked feature and label values for a few constructed loans."""

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.types import IntegerType, StringType, StructField, StructType  # noqa: E402

from creditrisk.data.schema import ORIGINATION_COLUMNS, PERFORMANCE_COLUMNS  # noqa: E402
from creditrisk.features import build  # noqa: E402

pytestmark = pytest.mark.spark


def _orig(seq, fp="201003", rate="6.000", upb="120000"):
    row = dict.fromkeys(ORIGINATION_COLUMNS)
    row.update(
        loan_seq_no=seq,
        first_payment_date=fp,
        credit_score="9999",
        orig_ltv="80",
        orig_cltv="80",
        orig_dti="999",
        orig_upb=upb,
        orig_interest_rate=rate,
        amortization_type="FRM",
        orig_loan_term="360",
        num_borrowers="02",
        occupancy_status="P",
        mi_pct="0",
        num_units="1",
    )
    return row


def _perf(seq, statuses, start=2010 * 12 + 2, zbc_at=None, upb="100000"):
    rows = []
    for m, s in enumerate(statuses):
        idx = start + m
        row = dict.fromkeys(PERFORMANCE_COLUMNS)
        row.update(
            loan_seq_no=seq,
            monthly_reporting_period=f"{idx // 12}{idx % 12 + 1:02d}",
            current_actual_upb=upb,
            current_delinquency_status=s,
            current_interest_rate="6.000",
        )
        if zbc_at is not None and m == zbc_at[0]:
            row["zero_balance_code"] = zbc_at[1]
        rows.append(row)
    return rows


def _schema(columns):
    return StructType(
        [StructField(c, StringType()) for c in columns] + [StructField("vintage", IntegerType())]
    )


def _frames(spark, orig_rows, perf_rows):
    o = spark.createDataFrame(
        [tuple(r[c] for c in ORIGINATION_COLUMNS) + (2010,) for r in orig_rows],
        _schema(ORIGINATION_COLUMNS),
    )
    p = spark.createDataFrame(
        [tuple(r[c] for c in PERFORMANCE_COLUMNS) + (2010,) for r in perf_rows],
        _schema(PERFORMANCE_COLUMNS),
    )
    return o, p


def test_behaviour_features_and_label(spark, cfg):
    # Months 1..12: current, 1, 2, 0 (cure), 0, ..., 1 at month 12 -> observation month dlq 1.
    hist_a = ["0", "1", "2", "0", "0", "0", "0", "0", "0", "0", "0", "1"]
    # outcome window: reaches status 3 at month 20 -> default
    fut_a = ["0"] * 7 + ["3"] + ["0"] * 16
    # loan B: clean history and clean outcome
    hist_b = ["0"] * 12
    fut_b = ["0"] * 24
    # loan C: already 90+ at observation -> excluded from the population
    hist_c = ["0"] * 9 + ["1", "2", "3"]
    # loan D: clean, prepaid in month 18 -> non-default, outcome complete
    fut_d = ["0"] * 6

    orig = [_orig("A"), _orig("B"), _orig("C"), _orig("D")]
    perf = (
        _perf("A", hist_a + fut_a)
        + _perf("B", hist_b + fut_b)
        + _perf("C", hist_c + ["3"] * 24)
        + _perf("D", hist_b + fut_d, zbc_at=(17, "01"))
    )
    o, p = _frames(spark, orig, perf)
    build.register_raw(o, p)
    feats = build.build_features(spark, cfg).toPandas().set_index("loan_seq_no")
    labels = build.build_labels(spark, cfg).toPandas().set_index("loan_seq_no")

    assert set(feats.index) == {"A", "B", "D"}
    a = feats.loc["A"]
    assert a["dlq_at_obs"] == 1
    assert a["max_dlq_12m"] == 2
    assert a["n_months_dlq30_12m"] == 3
    assert a["n_months_dlq30_6m"] == 1
    assert a["months_since_last_dlq"] == 0
    assert a["n_cures_12m"] == 1
    assert feats.loc["B", "months_since_last_dlq"] != feats.loc["B", "months_since_last_dlq"]  # NaN
    # sentinels -> NULL; borrowers '02' -> '2+'
    assert a["credit_score"] != a["credit_score"] and a["orig_dti"] != a["orig_dti"]
    assert a["num_borrowers_group"] == "2+"
    # balance 100k of 120k original -> 1/6 repaid
    assert a["upb_paydown_ratio"] == pytest.approx(1 / 6)

    assert labels.loc["A", "target"] == 1 and labels.loc["A", "outcome_complete"] == 1
    assert labels.loc["B", "target"] == 0 and labels.loc["B", "outcome_complete"] == 1
    assert labels.loc["D", "target"] == 0 and labels.loc["D", "outcome_complete"] == 1

"""Spark SQL feature pipeline: raw Freddie Mac files -> one row per loan at the observation point.

Order of operations:
    raw text -> raw_orig / raw_svcg views -> orig_clean / svcg_clean -> population
    -> behaviour (rows <= observation month only) -> features
    -> labels (rows after the observation month only) -> loan_level = features JOIN labels

Features are built before labels and never see them; `tests/test_leakage.py` checks that
features are unchanged when everything after the observation month is altered.
"""

from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from creditrisk.config import reports_dir
from creditrisk.data.schema import (
    ORIGINATION_COLUMNS,
    ORIGINATION_FILE,
    PERFORMANCE_COLUMNS,
    PERFORMANCE_FILE,
)

SQL_PACKAGE = "creditrisk.features.sql"


def load_sql(name: str, **params: Any) -> str:
    text = resources.files(SQL_PACKAGE).joinpath(f"{name}.sql").read_text(encoding="utf-8")
    return text.format(**params) if params else text


def sql_params(cfg: dict[str, Any]) -> dict[str, Any]:
    b = cfg["behavioural"]
    return {
        "obs_age": int(b["observation_age_months"]),
        "window": int(b["outcome_window_months"]),
        "default_dpd": int(b["default_dpd_status"]),
        "zbc_list": ", ".join(f"'{c}'" for c in b["default_zero_balance_codes"]),
    }


def _string_schema(columns: list[str]) -> StructType:
    return StructType([StructField(c, StringType(), True) for c in columns])


def read_raw_text(
    spark: SparkSession, raw_dir: Path, vintages: list[int]
) -> tuple[DataFrame, DataFrame]:
    """Read pipe-delimited sample files for the given vintages; add an integer `vintage` column."""

    def _read(pattern: str, columns: list[str]) -> DataFrame:
        paths = [str((raw_dir / pattern.format(year=y)).resolve()) for y in vintages]
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"Missing raw files: {missing}. Run `make fetch` first.")
        df = spark.read.csv(paths, sep="|", header=False, schema=_string_schema(columns))
        year = F.regexp_extract(F.input_file_name(), r"_(\d{4})\.txt$", 1).cast("int")
        return df.withColumn("vintage", year)

    return _read(ORIGINATION_FILE, ORIGINATION_COLUMNS), _read(
        PERFORMANCE_FILE, PERFORMANCE_COLUMNS
    )


def register_raw(raw_orig: DataFrame, raw_svcg: DataFrame) -> None:
    raw_orig.createOrReplaceTempView("raw_orig")
    raw_svcg.createOrReplaceTempView("raw_svcg")


def build_features(spark: SparkSession, cfg: dict[str, Any]) -> DataFrame:
    """Create the cleaned views and return the feature table. Requires raw_orig/raw_svcg views."""
    p = sql_params(cfg)
    spark.sql(load_sql("orig_clean")).createOrReplaceTempView("orig_clean")
    spark.sql(load_sql("svcg_clean")).createOrReplaceTempView("svcg_clean")
    spark.sql(load_sql("population", **p)).createOrReplaceTempView("population")
    spark.sql(load_sql("behaviour")).createOrReplaceTempView("behaviour")
    features = spark.sql(load_sql("features", **p))
    features.createOrReplaceTempView("features")
    return features


def build_labels(spark: SparkSession, cfg: dict[str, Any]) -> DataFrame:
    """Outcome flags. Requires the views created by `build_features`."""
    return spark.sql(load_sql("labels", **sql_params(cfg)))


def build_loan_level(spark: SparkSession, cfg: dict[str, Any]) -> DataFrame:
    features = build_features(spark, cfg)
    labels = build_labels(spark, cfg)
    return features.join(labels, on="loan_seq_no", how="inner")


def lgd_sample(spark: SparkSession, cfg: dict[str, Any]) -> DataFrame:
    return spark.sql(load_sql("lgd_sample", **sql_params(cfg)))


def data_quality(spark: SparkSession) -> dict[str, Any]:
    orig = [r.asDict() for r in spark.sql(load_sql("data_quality")).collect()]
    perf = spark.sql(load_sql("data_quality_perf")).collect()[0].asDict()
    return {"origination_by_vintage": orig, "performance": perf}


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Full feature step: ingest raw files, write loan-level and LGD tables, return a summary."""
    from creditrisk.features.spark import get_spark

    data = cfg["data"]
    vintages = list(data["vintages_dev"]) + list(data["vintages_oot"])
    raw_dir = Path(data["raw_dir"]).resolve()
    interim = Path(data["interim_dir"]).resolve()  # absolute: executors run elsewhere
    processed = Path(data["processed_dir"]).resolve()
    processed.mkdir(parents=True, exist_ok=True)

    spark = get_spark("creditrisk-features")
    raw_orig, raw_svcg = read_raw_text(spark, raw_dir, vintages)

    # Persist the raw text once as parquet; all later scans read the columnar copy.
    raw_orig.write.mode("overwrite").parquet(str(interim / "raw_orig"))
    raw_svcg.write.mode("overwrite").partitionBy("vintage").parquet(str(interim / "raw_svcg"))
    register_raw(
        spark.read.parquet(str(interim / "raw_orig")),
        spark.read.parquet(str(interim / "raw_svcg")),
    )

    dq = data_quality(spark)
    loan_level = build_loan_level(spark, cfg)
    loan_level.coalesce(1).write.mode("overwrite").parquet(str(processed / "loan_level"))
    lgd_sample(spark, cfg).coalesce(1).write.mode("overwrite").parquet(
        str(processed / "lgd_sample")
    )

    ll = spark.read.parquet(str(processed / "loan_level"))
    by_vintage = [
        r.asDict()
        for r in ll.groupBy("vintage")
        .agg(
            F.count("*").alias("n_population"),
            F.sum("outcome_complete").alias("n_outcome_complete"),
            F.sum(F.when(F.col("outcome_complete") == 1, F.col("target"))).alias("n_default"),
        )
        .orderBy("vintage")
        .collect()
    ]
    summary = {
        "data_quality": dq,
        "population_by_vintage": by_vintage,
        "n_features": len(ll.columns),
    }
    reports = reports_dir(cfg)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "data_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    spark.stop()
    return summary


def _check_sql_has_no_label_reference(sql_names: list[str]) -> list[str]:
    """Return the names of feature SQL files that mention a label column (used by tests)."""
    offenders = []
    for name in sql_names:
        text = load_sql(name)
        code = "\n".join(line.split("--", 1)[0] for line in text.splitlines())
        if re.search(r"\b(target|outcome_complete|is_default)\b", code, flags=re.IGNORECASE):
            offenders.append(name)
    return offenders


FEATURE_SQL_FILES = ["orig_clean", "svcg_clean", "population", "behaviour", "features"]

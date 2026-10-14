"""Baseline: Spark MLlib logistic regression on raw features.

Pipeline: median imputation + missing-indicator for numerics, StringIndexer + one-hot for
categoricals, standardisation, logistic regression. regParam is chosen on validation AUC.
Its job is to show what a plain, untuned linear model on the same features achieves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from creditrisk.features.catalog import CATEGORICAL_FEATURES, ID_COLUMN, NUMERIC_FEATURES, TARGET


def _prepare(df: DataFrame) -> DataFrame:
    out = df
    for c in NUMERIC_FEATURES:
        out = out.withColumn(c, F.col(c).cast("double"))
        out = out.withColumn(f"{c}__missing", F.col(c).isNull().cast("double"))
    for c in CATEGORICAL_FEATURES:
        out = out.withColumn(c, F.coalesce(F.col(c).cast("string"), F.lit("__MISSING__")))
    return out.withColumn(TARGET, F.col(TARGET).cast("double"))


def _pipeline(reg: float) -> Pipeline:
    imputed = [f"{c}__imp" for c in NUMERIC_FEATURES]
    stages = [
        Imputer(inputCols=NUMERIC_FEATURES, outputCols=imputed, strategy="median"),
        StringIndexer(
            inputCols=CATEGORICAL_FEATURES,
            outputCols=[f"{c}__idx" for c in CATEGORICAL_FEATURES],
            handleInvalid="keep",
        ),
        OneHotEncoder(
            inputCols=[f"{c}__idx" for c in CATEGORICAL_FEATURES],
            outputCols=[f"{c}__ohe" for c in CATEGORICAL_FEATURES],
            handleInvalid="keep",
        ),
        VectorAssembler(
            inputCols=imputed
            + [f"{c}__missing" for c in NUMERIC_FEATURES]
            + [f"{c}__ohe" for c in CATEGORICAL_FEATURES],
            outputCol="raw_features",
        ),
        StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True),
        LogisticRegression(featuresCol="features", labelCol=TARGET, regParam=reg, maxIter=200),
    ]
    return Pipeline(stages=stages)


def fit_and_score(
    spark: SparkSession, cfg: dict[str, Any], splits_path: Path
) -> tuple[PipelineModel, dict[str, pd.DataFrame], list[dict[str, float]]]:
    """Fit on train, choose regParam on validation, return PD predictions for every split.

    Reads the loan-level parquet natively in Spark and joins the split assignment saved by the
    train step, so the baseline sees exactly the same loans in each split as the other models.
    """
    cols = [ID_COLUMN, TARGET, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    loan_level = spark.read.parquet(
        str((Path(cfg["data"]["processed_dir"]) / "loan_level").resolve())
    )
    splits = spark.read.parquet(str(Path(splits_path).resolve()))
    data = _prepare(loan_level.join(splits, on=ID_COLUMN, how="inner").select(*cols, "split"))
    sdf = {k: data.filter(F.col("split") == k) for k in ["train", "validation", "test", "oot"]}
    sdf["train"].cache()
    evaluator = BinaryClassificationEvaluator(labelCol=TARGET, metricName="areaUnderROC")

    grid = []
    best = (-1.0, None)
    for reg in cfg["baseline"]["reg_grid"]:
        model = _pipeline(float(reg)).fit(sdf["train"])
        auc = evaluator.evaluate(model.transform(sdf["validation"]))
        grid.append({"reg_param": float(reg), "val_auc": auc})
        if auc > best[0]:
            best = (auc, model)
    model = best[1]

    preds = {}
    for k, df in sdf.items():
        p = model.transform(df).select(
            ID_COLUMN, vector_to_array("probability").getItem(1).alias("pd")
        )
        preds[k] = p.toPandas()
    return model, preds, grid

"""Train step: split, fit scorecard / XGBoost / Spark baseline, tune on validation only,
log to MLflow and register the two explainable models."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from creditrisk import tracking
from creditrisk.challenger.xgb import XGBChallenger
from creditrisk.config import models_dir, reports_dir
from creditrisk.evaluation.metrics import discrimination
from creditrisk.features.catalog import ALL_FEATURES, ID_COLUMN, TARGET
from creditrisk.scorecard.model import Scorecard
from creditrisk.split import SPLITS, assign_splits


def load_loan_level(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    ll = pd.read_parquet(Path(cfg["data"]["processed_dir"]) / "loan_level")
    counts = {"population": len(ll), "censored_excluded": int((ll["outcome_complete"] == 0).sum())}
    ll = ll[ll["outcome_complete"] == 1].reset_index(drop=True)
    ll["split"] = assign_splits(ll, cfg)
    ll = ll[ll["split"].notna()].reset_index(drop=True)
    return ll, counts


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    mdir = models_dir(cfg)
    mdir.mkdir(parents=True, exist_ok=True)
    ll, counts = load_loan_level(cfg)
    frames = {k: ll[ll["split"] == k] for k in SPLITS}
    ll[[ID_COLUMN, "split"]].to_parquet(mdir / "splits.parquet", index=False)
    tr, va = frames["train"], frames["validation"]
    y_tr, y_va = tr[TARGET].to_numpy(), va[TARGET].to_numpy()

    split_summary = {
        k: {"n": len(v), "n_default": int(v[TARGET].sum()), "default_rate": float(v[TARGET].mean())}
        for k, v in frames.items()
    }
    print(json.dumps({"counts": counts, "splits": split_summary}, indent=2))

    print("Fitting scorecard ...")
    scorecard = Scorecard(cfg["scorecard"], cfg["bands"], ALL_FEATURES).fit(tr, y_tr, va, y_va)
    print("Fitting XGBoost challenger ...")
    challenger = XGBChallenger(cfg["xgboost"], cfg["scorecard"], cfg["bands"], ALL_FEATURES).fit(
        tr, y_tr, va, y_va
    )
    print("Fitting Spark MLlib baseline ...")
    from creditrisk.baseline.spark_lr import fit_and_score
    from creditrisk.features.spark import get_spark

    spark = get_spark("creditrisk-baseline")
    spark_model, baseline_preds, baseline_grid = fit_and_score(spark, cfg, mdir / "splits.parquet")
    spark_model.write().overwrite().save(str(mdir / "spark_lr"))
    pd.concat([v.assign(split=k) for k, v in baseline_preds.items()]).to_parquet(
        mdir / "baseline_predictions.parquet", index=False
    )
    spark.stop()

    val_metrics = {
        "scorecard": discrimination(y_va, scorecard.predict_pd(va)),
        "xgboost": discrimination(y_va, challenger.predict_pd(va)),
        "spark_lr": discrimination(
            y_va,
            va[[ID_COLUMN]].merge(baseline_preds["validation"], on=ID_COLUMN)["pd"].to_numpy(),
        ),
    }

    for name, obj in {"scorecard": scorecard, "xgboost": challenger}.items():
        with (mdir / f"{name}.pkl").open("wb") as fh:
            pickle.dump(obj, fh)

    reports = reports_dir(cfg)
    reports.mkdir(parents=True, exist_ok=True)
    scorecard.iv_table.to_csv(reports / "scorecard_iv.csv", index=False)
    scorecard.points_table().to_csv(reports / "scorecard_points.csv", index=False)
    pd.DataFrame(challenger.trials).to_csv(reports / "xgboost_tuning.csv", index=False)
    pd.DataFrame(baseline_grid).to_csv(reports / "spark_lr_tuning.csv", index=False)

    tracking.setup(cfg)
    with mlflow.start_run(run_name="train") as run:
        mlflow.log_params(
            {
                "vintages_dev": cfg["data"]["vintages_dev"],
                "vintages_oot": cfg["data"]["vintages_oot"],
                "observation_age_months": cfg["behavioural"]["observation_age_months"],
                "outcome_window_months": cfg["behavioural"]["outcome_window_months"],
                "scorecard_C": scorecard.best_c,
                "scorecard_features": ",".join(scorecard.selected),
                "base_score": cfg["scorecard"]["base_score"],
                "base_odds": cfg["scorecard"]["base_odds"],
                "pdo": cfg["scorecard"]["pdo"],
                "xgb_best_iteration": challenger.booster.best_iteration,
                **{f"xgb_{k}": v for k, v in challenger.params.items() if k not in ("nthread",)},
            }
        )
        for model_name, m in val_metrics.items():
            mlflow.log_metrics({f"val_{model_name}_{k}": v for k, v in m.items()})
        mlflow.log_artifacts(str(reports), artifact_path="reports")
        mlflow.log_text("\n".join(scorecard.selection_log), "scorecard_selection_log.txt")
        sc_version = tracking.log_and_register(
            scorecard, "scorecard", cfg["mlflow"]["scorecard_model_name"], "champion"
        )
        xgb_version = tracking.log_and_register(
            challenger, "xgboost", cfg["mlflow"]["xgboost_model_name"], "challenger"
        )
        run_id = run.info.run_id

    summary = {
        "counts": counts,
        "splits": split_summary,
        "scorecard_selected": scorecard.selected,
        "scorecard_C": scorecard.best_c,
        "scorecard_selection_log": scorecard.selection_log,
        "xgb_params": challenger.params,
        "xgb_best_iteration": challenger.booster.best_iteration,
        "baseline_grid": baseline_grid,
        "validation_metrics": val_metrics,
        "mlflow_run_id": run_id,
        "registered_versions": {"scorecard": sc_version, "xgboost": xgb_version},
    }
    (reports / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary

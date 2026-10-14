"""Evaluate step: the only place where test and out-of-time (OOT) data are scored.

Outputs (aggregate only - no loan-level rows are written to reports/):
    reports/results.json                      every number quoted in the README
    reports/*.csv                             band, cut-off, reliability, PSI, segment, ECL tables
    reports/figures/*.png                     ROC, PR, reliability, score distribution, SHAP
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from creditrisk import tracking
from creditrisk.config import models_dir, reports_dir
from creditrisk.evaluation.metrics import (
    band_table,
    cutoff_table,
    discrimination,
    psi,
    reliability_table,
)
from creditrisk.features.catalog import EAD_COLUMN, ID_COLUMN, NUMERIC_FEATURES, TARGET
from creditrisk.scaling import ScoreScaling, assign_band
from creditrisk.train import load_loan_level

MODEL_LABELS = {"scorecard": "WoE scorecard", "xgboost": "XGBoost", "spark_lr": "Spark MLlib LR"}
COLORS = {"scorecard": "#2a6fdb", "xgboost": "#d9480f", "spark_lr": "#868e96"}


def _load_pickle(mdir: Path, name: str):
    with (mdir / f"{name}.pkl").open("rb") as fh:
        return pickle.load(fh)


def _predictions(ll: pd.DataFrame, models: dict, baseline: pd.DataFrame) -> pd.DataFrame:
    out = ll[[ID_COLUMN, "split", TARGET]].copy()
    for name, m in models.items():
        out[f"pd_{name}"] = m.predict_pd(ll)
    b = baseline.drop(columns="split").rename(columns={"pd": "pd_spark_lr"})
    return out.merge(b, on=ID_COLUMN, how="left")


def _figures(
    preds: pd.DataFrame, scaling: ScoreScaling, challenger, ll: pd.DataFrame, figures: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    for split in ["test", "oot"]:
        d = preds[preds["split"] == split]
        y = d[TARGET].to_numpy()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        for name in MODEL_LABELS:
            p = d[f"pd_{name}"].to_numpy()
            fpr, tpr, _ = roc_curve(y, p)
            axes[0].plot(
                fpr,
                tpr,
                color=COLORS[name],
                label=f"{MODEL_LABELS[name]} (AUC {roc_auc_score(y, p):.3f})",
            )
            prec, rec, _ = precision_recall_curve(y, p)
            axes[1].plot(rec, prec, color=COLORS[name], label=MODEL_LABELS[name])
            rt = reliability_table(y, p)
            axes[2].plot(
                rt["mean_pd"], rt["observed"], "o-", color=COLORS[name], label=MODEL_LABELS[name]
            )
        axes[0].plot([0, 1], [0, 1], ":", color="#adb5bd")
        axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
        axes[1].axhline(y.mean(), ls=":", color="#adb5bd")
        axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
        lim = max(rt["observed"].max(), d[[f"pd_{n}" for n in MODEL_LABELS]].mean().max()) * 1.1
        axes[2].plot([0, lim], [0, lim], ":", color="#adb5bd")
        axes[2].set(
            xlabel="Mean predicted PD (decile)",
            ylabel="Observed default rate",
            title="Reliability (deciles)",
        )
        for ax in axes:
            ax.legend(fontsize=8, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{'Out-of-time' if split == 'oot' else 'In-time test'} sample", y=1.02)
        fig.tight_layout()
        fig.savefig(figures / f"performance_{split}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    d = preds[preds["split"] == "test"]
    s = scaling.score_from_pd(d["pd_scorecard"])
    bins = np.linspace(np.percentile(s, 0.5), np.percentile(s, 99.5), 50)
    ax.hist(
        s[d[TARGET] == 0], bins=bins, alpha=0.6, density=True, label="Non-default", color="#2a6fdb"
    )
    ax.hist(s[d[TARGET] == 1], bins=bins, alpha=0.6, density=True, label="Default", color="#d9480f")
    ax.set(xlabel="Scorecard score", ylabel="Density", title="Score distribution (test)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / "score_distribution_test.png", dpi=130)
    plt.close(fig)

    sample = ll[ll["split"] == "test"].sample(
        n=min(5000, (ll["split"] == "test").sum()), random_state=0
    )
    mean_abs = challenger.shap_values(sample).drop(columns="bias").abs().mean()
    mean_abs = mean_abs.sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(mean_abs.index, mean_abs.values, color="#d9480f")
    ax.set(
        xlabel="Mean |SHAP| (log-odds of default)", title="XGBoost: global feature impact (test)"
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / "xgboost_shap_importance.png", dpi=130)
    plt.close(fig)


def _lgd(cfg: dict[str, Any]) -> dict[str, Any]:
    lgd = pd.read_parquet(Path(cfg["data"]["processed_dir"]) / "lgd_sample")
    lgd = lgd[lgd["vintage"].isin(cfg["data"]["vintages_dev"])]
    ratio = lgd["actual_loss"] / lgd["zero_balance_removal_upb"]
    # Sign convention is checked empirically rather than assumed.
    sign = -1.0 if ratio.median() < 0 else 1.0
    lgd_i = (sign * ratio).clip(0, 1)
    return {
        "n_credit_events": int(len(lgd)),
        "loss_sign_convention": "negative = loss" if sign < 0 else "positive = loss",
        "lgd_mean": float(lgd_i.mean()) if len(lgd) else None,
        "lgd_median": float(lgd_i.median()) if len(lgd) else None,
        "lgd_ead_weighted": float(
            (lgd_i * lgd["zero_balance_removal_upb"]).sum() / lgd["zero_balance_removal_upb"].sum()
        )
        if len(lgd)
        else None,
    }


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    reports = reports_dir(cfg)
    reports.mkdir(parents=True, exist_ok=True)
    mdir = models_dir(cfg)
    scaling = ScoreScaling.from_config(cfg["scorecard"])
    bands_cfg = cfg["bands"]
    ll, counts = load_loan_level(cfg)
    stored = pd.read_parquet(mdir / "splits.parquet")
    current = ll[[ID_COLUMN, "split"]].reset_index(drop=True)
    if len(current) != len(stored) or not (current.to_numpy() == stored.to_numpy()).all():
        raise RuntimeError("Split assignment differs from training run - refusing to evaluate.")

    models = {name: _load_pickle(mdir, name) for name in ["scorecard", "xgboost"]}
    baseline = pd.read_parquet(mdir / "baseline_predictions.parquet")
    preds = _predictions(ll, models, baseline)
    names = list(MODEL_LABELS)

    results: dict[str, Any] = {"counts": counts, "splits": {}, "metrics": {}}
    for split in ["train", "validation", "test", "oot"]:
        d = preds[preds["split"] == split]
        results["splits"][split] = {
            "n": len(d),
            "n_default": int(d[TARGET].sum()),
            "default_rate": float(d[TARGET].mean()),
        }
        if split == "train":
            continue
        results["metrics"][split] = {
            n: discrimination(d[TARGET].to_numpy(), d[f"pd_{n}"].to_numpy()) for n in names
        }

    # Interpretability cost: Gini given up by the scorecard relative to XGBoost.
    results["interpretability_cost"] = {
        s: {
            "gini_xgboost": results["metrics"][s]["xgboost"]["gini"],
            "gini_scorecard": results["metrics"][s]["scorecard"]["gini"],
            "gini_points_lost": results["metrics"][s]["xgboost"]["gini"]
            - results["metrics"][s]["scorecard"]["gini"],
        }
        for s in ["test", "oot"]
    }

    # Score bands, reliability and stability.
    tables: dict[str, pd.DataFrame] = {}
    for n in ["scorecard", "xgboost"]:
        for split in ["test", "oot"]:
            d = preds[preds["split"] == split]
            p = d[f"pd_{n}"].to_numpy()
            tables[f"bands_{n}_{split}"] = band_table(
                d[TARGET].to_numpy(),
                p,
                assign_band(p, bands_cfg["pd_edges"], bands_cfg["labels"]),
                bands_cfg["labels"],
            )
            tables[f"reliability_{n}_{split}"] = reliability_table(d[TARGET].to_numpy(), p)

    train_mask = preds["split"] == "train"
    psi_rows = []
    for n in names:
        ref = scaling.score_from_pd(preds.loc[train_mask, f"pd_{n}"])
        for split in ["test", "oot"]:
            cur = scaling.score_from_pd(preds.loc[preds["split"] == split, f"pd_{n}"])
            psi_rows.append({"model": n, "comparison": f"train_vs_{split}", "psi": psi(ref, cur)})
    tables["psi_scores"] = pd.DataFrame(psi_rows)
    csi_rows = []
    for f in NUMERIC_FEATURES:
        ref = ll.loc[ll["split"] == "train", f].dropna()
        cur = ll.loc[ll["split"] == "oot", f].dropna()
        if ref.nunique() > 1 and len(cur):
            csi_rows.append({"feature": f, "psi_train_vs_oot": psi(ref, cur)})
    tables["psi_features_oot"] = pd.DataFrame(csi_rows).sort_values(
        "psi_train_vs_oot", ascending=False
    )
    results["psi"] = psi_rows
    # Aggregate profile by vintage, used to explain out-of-time drift.
    prof = ll.groupby("vintage").agg(
        mean_orig_rate=("orig_interest_rate", "mean"), default_rate=(TARGET, "mean")
    )
    results["vintage_profile"] = {int(k): v for k, v in prof.round(5).to_dict("index").items()}

    # Cut-off: fixed on validation (approve the best target_approval_rate share), applied to
    # test and OOT unchanged.
    target_rate = cfg["cutoff"]["target_approval_rate"]
    results["cutoff"] = {"target_approval_rate_on_validation": target_rate}
    for n in ["scorecard", "xgboost"]:
        val_scores = scaling.score_from_pd(preds.loc[preds["split"] == "validation", f"pd_{n}"])
        cut = float(np.quantile(val_scores, 1 - target_rate))
        results["cutoff"][n] = {"cutoff_score": cut}
        for split in ["test", "oot"]:
            d = preds[preds["split"] == split]
            sc = scaling.score_from_pd(d[f"pd_{n}"])
            y = d[TARGET].to_numpy()
            grid = np.quantile(sc, np.linspace(0.05, 0.5, 10))
            tables[f"cutoff_grid_{n}_{split}"] = cutoff_table(y, sc, np.sort(np.append(grid, cut)))
            results["cutoff"][n][split] = cutoff_table(y, sc, np.array([cut])).iloc[0].to_dict()

    # Segment comparison at the scorecard cut-off. The data has no gender or age, so this is
    # NOT a protected-attribute fairness test - see the model card.
    seg_rows = []
    cut = results["cutoff"]["scorecard"]["cutoff_score"]
    for split in ["test", "oot"]:
        d = ll[ll["split"] == split].merge(preds[[ID_COLUMN, "pd_scorecard"]], on=ID_COLUMN)
        d["approved"] = scaling.score_from_pd(d["pd_scorecard"]) >= cut
        for seg in cfg["segments"]:
            for val, g in d.groupby(d[seg].fillna("Missing")):
                y = g[TARGET].to_numpy()
                seg_rows.append(
                    {
                        "split": split,
                        "segment": seg,
                        "value": val,
                        "n": len(g),
                        "default_rate": float(y.mean()),
                        "mean_pd": float(g["pd_scorecard"].mean()),
                        "approval_rate": float(g["approved"].mean()),
                        "bad_rate_approved": float(y[g["approved"]].mean())
                        if g["approved"].any()
                        else None,
                        "auc": float(roc_auc_score(y, g["pd_scorecard"]))
                        if 0 < y.sum() < len(y)
                        else None,
                    }
                )
    tables["segments"] = pd.DataFrame(seg_rows)

    # Reason-code frequency (aggregate): how often each feature is the top reason on test.
    sample = ll[ll["split"] == "test"].sample(
        n=min(5000, (ll["split"] == "test").sum()), random_state=1
    )
    freq = {}
    for n, m in models.items():
        top = [r[0]["feature"] for r in m.reason_codes(sample) if r]
        freq[n] = pd.Series(top).value_counts(normalize=True).round(4).to_dict()
    results["top_reason_frequency_test_sample"] = freq

    # Illustrative ECL on OOT: PD (24-month window) x LGD (empirical, dev vintages) x EAD (UPB).
    lgd = _lgd(cfg)
    results["lgd"] = lgd
    if lgd["lgd_ead_weighted"] is not None:
        d = ll[ll["split"] == "oot"].merge(preds[[ID_COLUMN, "pd_scorecard"]], on=ID_COLUMN)
        d["band"] = assign_band(d["pd_scorecard"], bands_cfg["pd_edges"], bands_cfg["labels"])
        d["ecl"] = d["pd_scorecard"] * lgd["lgd_ead_weighted"] * d[EAD_COLUMN]
        ecl = (
            d.groupby("band")
            .agg(
                n=(ID_COLUMN, "size"),
                ead=(EAD_COLUMN, "sum"),
                ecl=("ecl", "sum"),
                mean_pd=("pd_scorecard", "mean"),
                observed_default_rate=(TARGET, "mean"),
            )
            .reindex(bands_cfg["labels"])
            .reset_index()
        )
        ecl["ecl_to_ead"] = ecl["ecl"] / ecl["ead"]
        tables["ecl_illustrative_oot"] = ecl
        results["ecl_illustrative_oot"] = {
            "total_ead": float(d[EAD_COLUMN].sum()),
            "total_ecl": float(d["ecl"].sum()),
            "ecl_to_ead": float(d["ecl"].sum() / d[EAD_COLUMN].sum()),
        }

    for name, t in tables.items():
        t.to_csv(reports / f"{name}.csv", index=False)
    _figures(preds, scaling, models["xgboost"], ll, reports / "figures")
    (reports / "results.json").write_text(json.dumps(results, indent=2, default=_json_default))

    tracking.setup(cfg)
    with mlflow.start_run(run_name="evaluate"):
        for split in ["validation", "test", "oot"]:
            for n in names:
                mlflow.log_metrics(
                    {f"{split}_{n}_{k}": v for k, v in results["metrics"][split][n].items()}
                )
        mlflow.log_artifacts(str(reports), artifact_path="reports")
    return results


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float | np.floating) and np.isnan(o):
        return None
    return str(o)

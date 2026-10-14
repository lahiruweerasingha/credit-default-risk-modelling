"""Render the results section of README.md and the counts in docs/data_issues.md from the
JSON/CSV files written by the pipeline. Every published number therefore comes from a run.

    python scripts/render_results.py [--reports reports]

Text between <!-- RESULTS:START --> and <!-- RESULTS:END --> (README) and between
<!-- DQ:START --> and <!-- DQ:END --> (data_issues.md) is replaced.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

LABELS = {"scorecard": "WoE scorecard", "xgboost": "XGBoost", "spark_lr": "Spark MLlib LR"}


def _pct(x, d=2):
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:.{d}f}%"


def _f(x, d=3):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{d}f}"


def _table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def results_markdown(rep: Path) -> str:
    res = json.loads((rep / "results.json").read_text())
    train = json.loads((rep / "train_summary.json").read_text())
    data = json.loads((rep / "data_summary.json").read_text())
    out = []

    pop = pd.DataFrame(data["population_by_vintage"])
    pop["default_rate"] = pop["n_default"] / pop["n_outcome_complete"]
    pop["default_rate"] = pop["default_rate"].map(_pct)
    out.append("### Data (observation population at month 12)\n")
    out.append(
        _table(
            pop.rename(
                columns={
                    "n_population": "loans on book, not in default",
                    "n_outcome_complete": "outcome observed",
                    "n_default": "defaults (months 13-36)",
                }
            )
        )
    )
    s = res["splits"]
    out.append(
        f"\nSplits: train {s['train']['n']:,} / validation {s['validation']['n']:,} / "
        f"in-time test {s['test']['n']:,} (vintages 2010-2014) and out-of-time "
        f"{s['oot']['n']:,} (2015-2016). Loans with an incomplete outcome window excluded: "
        f"{res['counts']['censored_excluded']:,}.\n"
    )

    rows = []
    for split, name in [("test", "In-time test"), ("oot", "Out-of-time")]:
        for m in LABELS:
            r = res["metrics"][split][m]
            rows.append(
                [
                    name,
                    LABELS[m],
                    _f(r["auc"]),
                    _f(r["gini"]),
                    _f(r["ks"]),
                    _f(r["average_precision"]),
                    _f(r["brier"], 5),
                    _pct(r["default_rate"]),
                    _pct(r["mean_pd"]),
                    _f(r["default_rate"] / r["mean_pd"], 2),
                ]
            )
    out.append("### Discrimination and calibration\n")
    out.append(
        _table(
            pd.DataFrame(
                rows,
                columns=[
                    "sample",
                    "model",
                    "ROC-AUC",
                    "Gini",
                    "KS",
                    "avg precision",
                    "Brier",
                    "observed default rate",
                    "mean PD",
                    "observed / predicted",
                ],
            )
        )
    )

    ic = res["interpretability_cost"]
    out.append(
        "\n**Cost of interpretability** (XGBoost Gini minus scorecard Gini): "
        f"{_f(ic['test']['gini_points_lost'])} on the in-time test, "
        f"{_f(ic['oot']['gini_points_lost'])} out of time.\n"
    )

    out.append(_findings(rep, res))

    psi = pd.DataFrame(res["psi"])
    psi["model"] = psi["model"].map(LABELS)
    psi["psi"] = psi["psi"].map(lambda v: _f(v, 4))
    out.append("### Score stability (PSI)\n")
    out.append(_table(psi))
    out.append(
        "\nTrain-vs-test PSI is near zero by construction (random split); train-vs-OOT is "
        "the meaningful check (< 0.10 stable, 0.10-0.25 monitor, > 0.25 shift).\n"
    )
    feat_psi = pd.read_csv(rep / "psi_features_oot.csv").head(5)
    feat_psi["psi_train_vs_oot"] = feat_psi["psi_train_vs_oot"].map(lambda v: _f(v, 4))
    out.append("Largest feature shifts, train vs OOT:\n")
    out.append(_table(feat_psi))

    c = res["cutoff"]
    rows = []
    for m in ["scorecard", "xgboost"]:
        for split in ["test", "oot"]:
            r = c[m][split]
            rows.append(
                [
                    LABELS[m],
                    split,
                    f"{c[m]['cutoff_score']:.0f}",
                    _pct(r["approval_rate"], 1),
                    _pct(r["bad_rate_approved"]),
                    _pct(r["bad_rate_declined"]),
                    _pct(r["share_of_bads_declined"], 1),
                ]
            )
    out.append(
        f"\n### Decision cut-off (fixed on validation at "
        f"{_pct(c['target_approval_rate_on_validation'], 0)} approval)\n"
    )
    out.append(
        _table(
            pd.DataFrame(
                rows,
                columns=[
                    "model",
                    "sample",
                    "cut-off score",
                    "approval rate",
                    "bad rate approved",
                    "bad rate declined",
                    "share of bads declined",
                ],
            )
        )
    )

    bands = pd.read_csv(rep / "bands_scorecard_oot.csv")
    bands = bands.assign(
        share=bands["share"].map(lambda v: _pct(v, 1)),
        default_rate=bands["default_rate"].map(_pct),
        mean_pd=bands["mean_pd"].map(_pct),
        n=bands["n"].astype(int),
        defaults=bands["defaults"].astype(int),
    )[["band", "n", "share", "defaults", "default_rate", "mean_pd"]]
    out.append("\n### Scorecard risk bands, out-of-time\n")
    out.append(_table(bands))

    seg = pd.read_csv(rep / "segments.csv")
    seg = seg[seg["split"] == "oot"].assign(
        default_rate=lambda d: d["default_rate"].map(_pct),
        approval_rate=lambda d: d["approval_rate"].map(lambda v: _pct(v, 1)),
        bad_rate_approved=lambda d: d["bad_rate_approved"].map(_pct),
        auc=lambda d: d["auc"].map(_f),
    )[["segment", "value", "n", "default_rate", "approval_rate", "bad_rate_approved", "auc"]]
    out.append("\n### Segment check at the scorecard cut-off, out-of-time\n")
    out.append("Non-protected segments only - the data has no sex, age or race (see model card).\n")
    out.append(_table(seg))

    out.append("\n### Scorecard characteristics\n")
    iv = pd.read_csv(rep / "scorecard_iv.csv").set_index("feature")["iv"]
    sel = pd.DataFrame({"characteristic": train["scorecard_selected"]})
    sel["IV (train)"] = sel["characteristic"].map(lambda f: _f(iv[f]))
    out.append(_table(sel))
    out.append(
        f"\nRegularisation C = {train['scorecard_C']} (chosen on validation). "
        "Full points table: `reports/scorecard_points.csv`.\n"
    )

    if "ecl_illustrative_oot" in res:
        lgd, ecl = res["lgd"], res["ecl_illustrative_oot"]
        out.append("### Illustrative expected credit loss (not an SLFRS 9 / IFRS 9 estimate)\n")
        out.append(
            f"ECL = PD x LGD x EAD on the OOT book. LGD = {_pct(lgd['lgd_ead_weighted'], 1)} "
            f"(exposure-weighted realised loss on {lgd['n_credit_events']:,} credit-event "
            f"terminations in development vintages; median {_pct(lgd['lgd_median'], 1)}). "
            f"EAD = balance at month 12. PD is the 24-month model PD. Result: ECL "
            f"{_pct(ecl['ecl_to_ead'], 3)} of EAD. Amounts are USD for US mortgages; this "
            "illustrates the mechanics only - no staging, lifetime PD or forward-looking "
            "scenarios.\n"
        )
    return "\n".join(out)


def _rate_story(res: dict) -> str:
    """Explain the rate shift from the per-vintage profile written by evaluate."""
    prof = res.get("vintage_profile")
    if not prof:
        return ""
    df = pd.DataFrame(prof).T
    df.index = df.index.astype(int)
    dev = df.loc[df.index <= 2014]
    oot = df.loc[df.index >= 2015]
    low_rate_dev = dev.nsmallest(2, "mean_orig_rate")
    safest = set(dev.nsmallest(2, "default_rate").index)
    if (
        set(low_rate_dev.index) != safest
        or oot["mean_orig_rate"].mean() >= dev["mean_orig_rate"].mean()
    ):
        return ""  # the simple rate-vintage story does not hold for this run; say nothing
    return (
        "Average note rates were lower out of time ("
        + ", ".join(f"{y}: {r:.2f}%" for y, r in oot["mean_orig_rate"].items())
        + f") than across development ({dev['mean_orig_rate'].mean():.2f}% on average), and the "
        "lowest-rate development vintages ("
        + ", ".join(
            f"{y}: {r:.2f}% rate, {_pct(d)} defaults"
            for y, (r, d) in low_rate_dev[["mean_orig_rate", "default_rate"]].iterrows()
        )
        + ") were also the two safest, so the absolute rate partly encodes the vintage. "
    )


def _findings(rep: Path, res: dict) -> str:
    """Key findings, with every number read from the run outputs."""
    m = res["metrics"]
    sc_t, xg_t = m["test"]["scorecard"], m["test"]["xgboost"]
    sc_o, xg_o = m["oot"]["scorecard"], m["oot"]["xgboost"]
    top = pd.read_csv(rep / "psi_features_oot.csv").iloc[0]
    cut = res["cutoff"]["scorecard"]
    bands = pd.read_csv(rep / "bands_scorecard_oot.csv").dropna(subset=["default_rate"])
    monotonic = bool(bands["default_rate"].is_monotonic_increasing)
    lines = [
        "### Key findings\n",
        "- **Interpretability is almost free in time, not out of time.** On the in-time test "
        f"the scorecard and XGBoost are nearly tied (Gini {_f(sc_t['gini'])} vs "
        f"{_f(xg_t['gini'])}); on the 2015-2016 vintages XGBoost degrades less (Gini "
        f"{_f(sc_o['gini'])} vs {_f(xg_o['gini'])}).",
        "- **Both models under-predict out of time.** In time, mean PD matches the observed "
        f"default rate ({_pct(sc_t['mean_pd'])} vs {_pct(sc_t['default_rate'])} for the "
        f"scorecard). Out of time the observed rate is {_pct(sc_o['default_rate'])} against a "
        "mean PD of "
        f"{_pct(sc_o['mean_pd'])} (scorecard) and {_pct(xg_o['mean_pd'])} (XGBoost). The largest "
        f"characteristic shift is `{top['feature']}` (PSI {_f(top['psi_train_vs_oot'], 2)}). "
        f"{_rate_story(res)}The production response would be recalibration on recent vintages and "
        "replacing the absolute rate with the rate spread. That change is deliberately **not** "
        "made here: modifying the model after seeing out-of-time results would contaminate the "
        "out-of-time test.",
        f"- **The cut-off drifts with it.** The scorecard cut-off fixed on validation at "
        f"{_pct(res['cutoff']['target_approval_rate_on_validation'], 0)} approval approves "
        f"{_pct(cut['oot']['approval_rate'], 1)} of the out-of-time book, while still declining "
        f"{_pct(cut['oot']['share_of_bads_declined'], 1)} of its defaults.",
        (
            "- **Ranking holds out of time.** Default rates still rise monotonically across "
            "bands A-G (table below), so recalibration rather than redevelopment is indicated."
            if monotonic
            else "- **Ranking weakens out of time:** default rates are not monotonic across "
            "bands A-G (table below)."
        ),
        "",
    ]
    return "\n".join(lines)


def dq_markdown(rep: Path) -> str:
    data = json.loads((rep / "data_summary.json").read_text())
    orig = pd.DataFrame(data["data_quality"]["origination_by_vintage"])
    perf = data["data_quality"]["performance"]
    lines = ["Counts from the run (raw origination files, by vintage):\n", _table(orig)]
    lines.append(
        f"\nPerformance files: {perf['n_rows']:,} rows for {perf['n_loans']:,} loans, reporting "
        f"periods {perf['first_period']} to {perf['last_period']}; 'RA' (REO) rows: "
        f"{perf['dlq_ra']:,}; other non-numeric delinquency codes: "
        f"{perf['dlq_other_non_numeric']:,}; duplicate loan-month keys: "
        f"{perf['duplicate_loan_months']:,}.\n"
    )
    return "\n".join(lines)


def _replace(path: Path, tag: str, body: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(rf"(<!-- {tag}:START -->)(.*?)(<!-- {tag}:END -->)", re.S)
    if not pattern.search(text):
        raise SystemExit(f"markers for {tag} not found in {path}")
    path.write_text(pattern.sub(lambda m: f"{m[1]}\n{body}\n{m[3]}", text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    rep = Path(args.reports)
    _replace(Path("README.md"), "RESULTS", results_markdown(rep))
    _replace(Path("docs/data_issues.md"), "DQ", dq_markdown(rep))
    print("README.md and docs/data_issues.md updated from", rep)


if __name__ == "__main__":
    main()

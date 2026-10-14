"""Seeded synthetic data for tests. Nothing here is derived from the real dataset; only the
public file layout (column order and codes) is reproduced.

make_loan_level   -> a model-ready table (one row per loan) for pandas-only tests
write_raw_files   -> pipe-delimited sample_orig_YYYY.txt / sample_perf_YYYY.txt for Spark tests
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from creditrisk.data.schema import ORIGINATION_COLUMNS, PERFORMANCE_COLUMNS

STATES = ["CA", "TX", "FL", "NY", "IL", "OH", "GA", "WA"]


def make_loan_level(n: int = 6000, seed: int = 7, vintage: int = 2010) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    credit = rng.normal(735, 45, n).clip(300, 850).round()
    ltv = rng.uniform(40, 97, n).round()
    dti = rng.uniform(10, 50, n).round()
    rate = rng.normal(4.5, 0.6, n).round(3)
    dlq_months = rng.poisson(0.25, n).clip(0, 12)
    dlq_now = np.where(dlq_months > 0, rng.integers(0, 3, n), 0)
    occ = rng.choice(["P", "I", "S"], n, p=[0.85, 0.1, 0.05])
    df = pd.DataFrame(
        {
            "loan_seq_no": [f"F{vintage % 100:02d}Q1{i:07d}" for i in range(n)],
            "vintage": vintage,
            "credit_score": credit,
            "orig_ltv": ltv,
            "orig_cltv": ltv + rng.integers(0, 5, n),
            "orig_dti": dti,
            "orig_upb": rng.integers(80, 450, n) * 1000.0,
            "orig_interest_rate": rate,
            "rate_spread_at_orig": rng.normal(0, 0.3, n).round(3),
            "mi_pct": np.where(ltv > 80, 25.0, 0.0),
            "orig_loan_term": rng.choice([180, 360], n, p=[0.2, 0.8]).astype(float),
            "num_units": 1.0,
            "num_borrowers_group": rng.choice(["1", "2+"], n),
            "first_time_homebuyer_flag": rng.choice(["Y", "N", None], n, p=[0.3, 0.65, 0.05]),
            "occupancy_status": occ,
            "channel": rng.choice(["R", "B", "C", "T"], n),
            "loan_purpose": rng.choice(["P", "C", "N"], n),
            "property_type": rng.choice(["SF", "CO", "PU"], n, p=[0.7, 0.1, 0.2]),
            "property_state": rng.choice(STATES, n),
            "relief_refi_indicator": rng.choice(["Y", "N"], n, p=[0.05, 0.95]),
            "dlq_at_obs": dlq_now.astype(float),
            "max_dlq_12m": np.maximum(dlq_now, (dlq_months > 0).astype(int)).astype(float),
            "n_months_dlq30_12m": dlq_months.astype(float),
            "n_months_dlq30_6m": np.minimum(dlq_months, 6).astype(float),
            "months_since_last_dlq": np.where(dlq_months > 0, rng.integers(0, 12, n), np.nan),
            "n_cures_12m": np.minimum(dlq_months, rng.integers(0, 3, n)).astype(float),
            "upb_paydown_ratio": rng.uniform(0.0, 0.08, n),
            "excess_paydown_ratio": rng.normal(0.0, 0.02, n),
            "rate_change": 0.0,
            "ever_modified": rng.choice([0.0, 1.0], n, p=[0.98, 0.02]),
            "ever_payment_deferral": 0.0,
            "ever_assistance": rng.choice([0.0, 1.0], n, p=[0.97, 0.03]),
        }
    )
    df.loc[rng.random(n) < 0.03, "orig_dti"] = np.nan  # like the 999 sentinel
    logit = (
        -3.2
        - 0.025 * (credit - 735)
        + 0.03 * (ltv - 75)
        + 0.02 * (dti - 30)
        + 1.1 * dlq_now
        + 0.45 * dlq_months
        + 0.6 * (occ == "I")
        - 8.0 * df["excess_paydown_ratio"].to_numpy()
    )
    df["target"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df["upb_at_obs"] = df["orig_upb"] * (1 - df["upb_paydown_ratio"])
    df["outcome_complete"] = 1
    return df


# ------------------------------------------------------------------------- raw files
def _yyyymm(idx: int) -> str:
    return f"{idx // 12:04d}{idx % 12 + 1:02d}"


def write_raw_files(
    out_dir: Path, vintages: list[int], n_per_vintage: int = 150, seed: int = 11
) -> Path:
    """Write synthetic origination + performance files in the official layout."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for year in vintages:
        orig_rows, perf_rows = [], []
        for i in range(n_per_vintage):
            seq = f"F{year % 100:02d}Q{rng.integers(1, 5)}{i:07d}"
            fp_idx = year * 12 + int(rng.integers(1, 12))
            credit = int(np.clip(rng.normal(730, 50), 300, 850))
            ltv = int(rng.integers(50, 97))
            rate = round(float(rng.normal(4.5, 0.5)), 3)
            upb = float(rng.integers(80, 400) * 1000)
            arm = rng.random() < 0.03
            orig = dict.fromkeys(ORIGINATION_COLUMNS, "")
            orig.update(
                credit_score="9999" if rng.random() < 0.02 else str(credit),
                first_payment_date=_yyyymm(fp_idx),
                first_time_homebuyer_flag=str(rng.choice(["Y", "N", "9"], p=[0.3, 0.65, 0.05])),
                maturity_date=_yyyymm(fp_idx + 359),
                mi_pct="25" if ltv > 80 else "0",
                num_units="1",
                occupancy_status=str(rng.choice(["P", "I", "S"], p=[0.85, 0.1, 0.05])),
                orig_cltv=str(ltv),
                orig_dti="999" if rng.random() < 0.03 else str(int(rng.integers(15, 50))),
                orig_upb=str(int(upb)),
                orig_ltv=str(ltv),
                orig_interest_rate=f"{rate:.3f}",
                channel=str(rng.choice(["R", "B", "C", "T"])),
                ppm_flag="N",
                amortization_type="ARM" if arm else "FRM",
                property_state=str(rng.choice(STATES)),
                property_type=str(rng.choice(["SF", "CO", "PU"])),
                postal_code="12300",
                loan_seq_no=seq,
                loan_purpose=str(rng.choice(["P", "C", "N"])),
                orig_loan_term="360",
                num_borrowers=str(rng.choice(["01", "02"])),
                seller_name="Other sellers",
                relief_refi_indicator="N",
                property_valuation_method="9",
                io_indicator="N",
            )
            orig_rows.append("|".join(orig[c] for c in ORIGINATION_COLUMNS))

            risk = 0.01 + 0.04 * (credit < 680) + 0.03 * (ltv > 90)
            dlq, bal = 0, upb
            r = rate / 1200
            pay = upb * r / (1 - (1 + r) ** -360)
            for m in range(40):
                per = fp_idx + m
                zbc, actual_loss, removal = "", "", ""
                status = str(dlq)
                if dlq == 0:
                    bal = bal * (1 + r) - pay
                    if rng.random() < 0.01 and m > 2:
                        zbc, removal = "01", f"{bal:.2f}"
                    elif rng.random() < risk:
                        dlq = 1
                elif dlq >= 4:
                    status, zbc = "RA", ""
                    if rng.random() < 0.5:
                        zbc, removal = "09", f"{bal:.2f}"
                        actual_loss = f"{-0.3 * bal:.2f}"
                else:
                    u = rng.random()
                    dlq = 0 if u < 0.35 else (dlq + 1 if u < 0.8 else dlq)
                perf = dict.fromkeys(PERFORMANCE_COLUMNS, "")
                perf.update(
                    loan_seq_no=seq,
                    monthly_reporting_period=_yyyymm(per),
                    current_actual_upb=f"{0 if zbc else bal:.2f}",
                    current_delinquency_status=status,
                    loan_age=str(m + 1),
                    remaining_months_to_maturity=str(359 - m),
                    zero_balance_code=zbc,
                    zero_balance_effective_date=_yyyymm(per) if zbc else "",
                    current_interest_rate=f"{rate:.3f}",
                    current_non_interest_bearing_upb="0",
                    actual_loss=actual_loss,
                    zero_balance_removal_upb=removal,
                    interest_bearing_upb=f"{bal:.2f}",
                    undocumented_33="7",
                    servicer_name="OTHER",
                )
                perf_rows.append("|".join(perf[c] for c in PERFORMANCE_COLUMNS))
                if zbc:
                    break
        (out_dir / f"sample_orig_{year}.txt").write_text("\n".join(orig_rows) + "\n")
        (out_dir / f"sample_perf_{year}.txt").write_text("\n".join(perf_rows) + "\n")
    return out_dir

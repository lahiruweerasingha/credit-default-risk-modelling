"""Model feature catalogue: the single source of truth for feature names, types,
plain-English descriptions and reason-code wording.

Every feature is known at the observation point (month 12 of the loan). Origination
features come from the origination file; behavioural features are aggregated by Spark
from performance records up to and including the observation month only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str  # "numeric" | "categorical"
    source: str  # "origination" | "behaviour"
    description: str
    reason: str  # adverse-action style wording used in reason codes


FEATURES: list[Feature] = [
    # --- origination -------------------------------------------------------------
    Feature(
        "credit_score",
        "numeric",
        "origination",
        "Borrower credit score at origination",
        "Credit score at origination",
    ),
    Feature(
        "orig_ltv",
        "numeric",
        "origination",
        "Original loan-to-value ratio (%)",
        "Loan-to-value at origination",
    ),
    Feature(
        "orig_cltv",
        "numeric",
        "origination",
        "Original combined loan-to-value ratio (%)",
        "Combined loan-to-value at origination",
    ),
    Feature(
        "orig_dti",
        "numeric",
        "origination",
        "Original debt-to-income ratio (%)",
        "Debt-to-income at origination",
    ),
    Feature(
        "orig_upb",
        "numeric",
        "origination",
        "Original unpaid principal balance (USD)",
        "Original loan amount",
    ),
    Feature(
        "orig_interest_rate",
        "numeric",
        "origination",
        "Note rate at origination (%)",
        "Interest rate at origination",
    ),
    Feature(
        "rate_spread_at_orig",
        "numeric",
        "origination",
        "Note rate minus the average rate of loans with the same origination quarter "
        "and term (percentage points)",
        "Interest rate high relative to market at origination",
    ),
    Feature(
        "mi_pct",
        "numeric",
        "origination",
        "Mortgage insurance coverage (%)",
        "Mortgage insurance coverage",
    ),
    Feature(
        "orig_loan_term",
        "numeric",
        "origination",
        "Original loan term (months)",
        "Original loan term",
    ),
    Feature(
        "num_units",
        "numeric",
        "origination",
        "Number of units in the property",
        "Number of units in the property",
    ),
    Feature(
        "num_borrowers_group",
        "categorical",
        "origination",
        "Single borrower (1) or more than one borrower (2+)",
        "Number of borrowers",
    ),
    Feature(
        "first_time_homebuyer_flag",
        "categorical",
        "origination",
        "First-time homebuyer (Y/N)",
        "First-time homebuyer status",
    ),
    Feature(
        "occupancy_status",
        "categorical",
        "origination",
        "P = primary, I = investment, S = second home",
        "Occupancy type",
    ),
    Feature(
        "channel",
        "categorical",
        "origination",
        "R = retail, B = broker, C = correspondent, T = TPO not specified",
        "Origination channel",
    ),
    Feature(
        "loan_purpose",
        "categorical",
        "origination",
        "P = purchase, C = cash-out refi, N = no cash-out refi",
        "Loan purpose",
    ),
    Feature("property_type", "categorical", "origination", "SF, CO, PU, MH, CP", "Property type"),
    Feature(
        "property_state",
        "categorical",
        "origination",
        "Two-letter US state of the property",
        "Property location (state)",
    ),
    Feature(
        "relief_refi_indicator",
        "categorical",
        "origination",
        "Y = relief refinance (incl. HARP)",
        "Relief refinance loan",
    ),
    # --- behaviour, months 1..12 ---------------------------------------------------
    Feature(
        "dlq_at_obs",
        "numeric",
        "behaviour",
        "Delinquency status at the observation month (0 = current, 1 = 30-59 DPD, 2 = 60-89)",
        "Currently delinquent",
    ),
    Feature(
        "max_dlq_12m",
        "numeric",
        "behaviour",
        "Worst delinquency status in the first 12 months",
        "Worst delinquency in the past year",
    ),
    Feature(
        "n_months_dlq30_12m",
        "numeric",
        "behaviour",
        "Months 30+ days delinquent in the first 12 months",
        "Months delinquent in the past year",
    ),
    Feature(
        "n_months_dlq30_6m",
        "numeric",
        "behaviour",
        "Months 30+ days delinquent in the last 6 months before observation",
        "Months delinquent in the past six months",
    ),
    Feature(
        "months_since_last_dlq",
        "numeric",
        "behaviour",
        "Months since the most recent 30+ DPD month (missing = never delinquent)",
        "Recent delinquency",
    ),
    Feature(
        "n_cures_12m",
        "numeric",
        "behaviour",
        "Number of transitions from delinquent back to current",
        "Repeated delinquency and cure",
    ),
    Feature(
        "upb_paydown_ratio",
        "numeric",
        "behaviour",
        "Share of original balance repaid by the observation month",
        "Slow balance repayment",
    ),
    Feature(
        "excess_paydown_ratio",
        "numeric",
        "behaviour",
        "Repayment beyond the contractual amortisation schedule, as a share of original balance",
        "No prepayment beyond schedule",
    ),
    Feature(
        "rate_change",
        "numeric",
        "behaviour",
        "Current note rate minus original note rate (percentage points)",
        "Change in interest rate",
    ),
    Feature(
        "ever_modified",
        "numeric",
        "behaviour",
        "Loan modified at or before observation (0/1)",
        "Loan has been modified",
    ),
    Feature(
        "ever_payment_deferral",
        "numeric",
        "behaviour",
        "Payment deferral granted at or before observation (0/1)",
        "Payment deferral granted",
    ),
    Feature(
        "ever_assistance",
        "numeric",
        "behaviour",
        "Forbearance / repayment / trial plan at or before observation (0/1)",
        "Borrower assistance plan",
    ),
]

FEATURE_BY_NAME: dict[str, Feature] = {f.name: f for f in FEATURES}


def reason_text(name: str, value) -> str:
    """Reason-code wording; flags inputs that were missing rather than observed."""
    missing = value is None or (isinstance(value, float) and value != value)
    return FEATURE_BY_NAME[name].reason + (" (not provided)" if missing else "")


NUMERIC_FEATURES: list[str] = [f.name for f in FEATURES if f.kind == "numeric"]
CATEGORICAL_FEATURES: list[str] = [f.name for f in FEATURES if f.kind == "categorical"]
ALL_FEATURES: list[str] = [f.name for f in FEATURES]

ID_COLUMN = "loan_seq_no"
TARGET = "target"
VINTAGE = "vintage"
EAD_COLUMN = "upb_at_obs"

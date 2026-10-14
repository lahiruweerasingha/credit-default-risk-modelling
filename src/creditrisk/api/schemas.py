"""Request / response contract for the scoring API.

Inputs are the model features at the observation point (month 12 of the loan). In production
the behavioural fields would come from a feature store fed by the same Spark aggregation; here
the caller supplies them. Every field is optional - missing values are scored through the
models' missing-value handling - but supplied values are range- and code-checked.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Flag = Literal["Y", "N"]

# A synthetic example (not a real loan) used in the docs and README.
EXAMPLE_REQUEST = {
    "loan_id": "example-001",
    "features": {
        "credit_score": 682,
        "orig_ltv": 90,
        "orig_cltv": 90,
        "orig_dti": 44,
        "orig_upb": 210000,
        "orig_interest_rate": 4.6,
        "rate_spread_at_orig": 0.35,
        "mi_pct": 25,
        "orig_loan_term": 360,
        "num_units": 1,
        "num_borrowers_group": "1",
        "first_time_homebuyer_flag": "Y",
        "occupancy_status": "P",
        "channel": "R",
        "loan_purpose": "P",
        "property_type": "SF",
        "property_state": "FL",
        "relief_refi_indicator": "N",
        "dlq_at_obs": 1,
        "max_dlq_12m": 1,
        "n_months_dlq30_12m": 2,
        "n_months_dlq30_6m": 2,
        "months_since_last_dlq": 0,
        "n_cures_12m": 1,
        "upb_paydown_ratio": 0.012,
        "excess_paydown_ratio": -0.002,
        "rate_change": 0.0,
        "ever_modified": 0,
        "ever_payment_deferral": 0,
        "ever_assistance": 0,
    },
}


class LoanFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # origination
    credit_score: int | None = Field(None, ge=300, le=850)
    orig_ltv: float | None = Field(None, ge=1, le=998)
    orig_cltv: float | None = Field(None, ge=1, le=998)
    orig_dti: float | None = Field(None, ge=0, le=65)
    orig_upb: float | None = Field(None, gt=0)
    orig_interest_rate: float | None = Field(None, ge=0, le=25)
    rate_spread_at_orig: float | None = Field(None, ge=-10, le=10)
    mi_pct: float | None = Field(None, ge=0, le=55)
    orig_loan_term: int | None = Field(None, ge=1, le=600)
    num_units: int | None = Field(None, ge=1, le=4)
    num_borrowers_group: Literal["1", "2+"] | None = None
    first_time_homebuyer_flag: Flag | None = None
    occupancy_status: Literal["P", "I", "S"] | None = None
    channel: Literal["R", "B", "C", "T"] | None = None
    loan_purpose: Literal["P", "C", "N", "R"] | None = None
    property_type: Literal["SF", "CO", "PU", "MH", "CP"] | None = None
    property_state: str | None = Field(None, pattern=r"^[A-Z]{2}$")
    relief_refi_indicator: Flag | None = None
    # behaviour, months 1..12
    dlq_at_obs: int | None = Field(None, ge=0, le=2)
    max_dlq_12m: int | None = Field(None, ge=0, le=12)
    n_months_dlq30_12m: int | None = Field(None, ge=0, le=12)
    n_months_dlq30_6m: int | None = Field(None, ge=0, le=6)
    months_since_last_dlq: int | None = Field(None, ge=0, le=12)
    n_cures_12m: int | None = Field(None, ge=0, le=12)
    upb_paydown_ratio: float | None = Field(None, ge=-1, le=1)
    excess_paydown_ratio: float | None = Field(None, ge=-1, le=1)
    rate_change: float | None = Field(None, ge=-15, le=15)
    ever_modified: int | None = Field(None, ge=0, le=1)
    ever_payment_deferral: int | None = Field(None, ge=0, le=1)
    ever_assistance: int | None = Field(None, ge=0, le=1)


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_REQUEST]})
    loan_id: str | None = Field(None, max_length=64)
    features: LoanFeatures


class ReasonCode(BaseModel):
    feature: str
    reason: str
    value: float | int | str | None
    impact: float
    impact_unit: str


class ScoreResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    loan_id: str | None
    model: Literal["scorecard", "xgboost"]
    model_version: str
    pd: float = Field(
        ..., description="Probability of 90+ DPD / credit event in the next 24 months"
    )
    score: float
    band: str
    reason_codes: list[ReasonCode]

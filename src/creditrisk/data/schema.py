"""File layouts of the Freddie Mac Single-Family Loan-Level Dataset (sample files).

Files are pipe-delimited with no header row. Column order follows the official
General User Guide ("File Layout & Data Dictionary"), corrected against the files actually
delivered (checked on the 2010-2016 sample files downloaded in October 2026):
  * origination files have 31 columns - SERVICER NAME (guide position 25) is no longer there,
    so every later field sits one position earlier than in the guide;
  * performance files have 35 columns - the documented 32 plus three trailing columns, of which
    the 34th is the servicer name;
  * the performance file is named sample_perf_YYYY.txt (the guide says sample_svcg_YYYY.txt).
Everything is read as a string
and typed explicitly in SQL so that sentinel codes (9999, 999, 99, 9) are handled
deliberately rather than silently coerced.
"""

ORIGINATION_COLUMNS: list[str] = [
    "credit_score",
    "first_payment_date",
    "first_time_homebuyer_flag",
    "maturity_date",
    "msa",
    "mi_pct",
    "num_units",
    "occupancy_status",
    "orig_cltv",
    "orig_dti",
    "orig_upb",
    "orig_ltv",
    "orig_interest_rate",
    "channel",
    "ppm_flag",
    "amortization_type",
    "property_state",
    "property_type",
    "postal_code",
    "loan_seq_no",
    "loan_purpose",
    "orig_loan_term",
    "num_borrowers",
    "seller_name",
    "super_conforming_flag",
    "pre_relief_refi_loan_seq_no",
    "special_eligibility_program",
    "relief_refi_indicator",
    "property_valuation_method",
    "io_indicator",
    "mi_cancellation_indicator",
]

PERFORMANCE_COLUMNS: list[str] = [
    "loan_seq_no",
    "monthly_reporting_period",
    "current_actual_upb",
    "current_delinquency_status",
    "loan_age",
    "remaining_months_to_maturity",
    "defect_settlement_date",
    "modification_flag",
    "zero_balance_code",
    "zero_balance_effective_date",
    "current_interest_rate",
    "current_non_interest_bearing_upb",
    "ddlpi",
    "mi_recoveries",
    "net_sale_proceeds",
    "non_mi_recoveries",
    "total_expenses",
    "legal_costs",
    "maintenance_preservation_costs",
    "taxes_insurance",
    "misc_expenses",
    "actual_loss",
    "cumulative_modification_cost",
    "interest_rate_step_indicator",
    "payment_deferral_flag",
    "eltv",
    "zero_balance_removal_upb",
    "delinquent_accrued_interest",
    "delinquency_due_to_disaster",
    "borrower_assistance_status_code",
    "current_month_modification_cost",
    "interest_bearing_upb",
    "undocumented_33",  # constant '7' in the 2010-2016 samples; not used
    "servicer_name",
    "undocumented_35",  # blank or 0.00; not used
]

ORIGINATION_FILE = "sample_orig_{year}.txt"
PERFORMANCE_FILE = "sample_perf_{year}.txt"

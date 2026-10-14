-- Typed origination records, one row per loan.
-- Sentinel codes from the Freddie Mac file layout become NULL (see docs/data_issues.md):
--   credit score 9999 (or outside 300-850), MI% / CLTV / DTI / LTV 999, units 99,
--   occupancy / channel / first-time-buyer / purpose 9, property type 99, borrowers 99.
SELECT
  trim(loan_seq_no)                                                     AS loan_seq_no,
  vintage,
  substr(trim(loan_seq_no), 2, 4)                                       AS orig_quarter,
  CASE WHEN CAST(trim(credit_score) AS INT) BETWEEN 300 AND 850
       THEN CAST(trim(credit_score) AS INT) END                         AS credit_score,
  CAST(substr(trim(first_payment_date), 1, 4) AS INT) * 12
    + CAST(substr(trim(first_payment_date), 5, 2) AS INT) - 1           AS fp_idx,
  CASE WHEN trim(first_time_homebuyer_flag) IN ('Y', 'N')
       THEN trim(first_time_homebuyer_flag) END                         AS first_time_homebuyer_flag,
  CASE WHEN CAST(trim(mi_pct) AS INT) BETWEEN 0 AND 55
       THEN CAST(trim(mi_pct) AS INT) END                               AS mi_pct,
  CASE WHEN CAST(trim(num_units) AS INT) BETWEEN 1 AND 4
       THEN CAST(trim(num_units) AS INT) END                            AS num_units,
  CASE WHEN trim(occupancy_status) IN ('P', 'I', 'S')
       THEN trim(occupancy_status) END                                  AS occupancy_status,
  CASE WHEN CAST(trim(orig_cltv) AS INT) BETWEEN 1 AND 998
       THEN CAST(trim(orig_cltv) AS INT) END                            AS orig_cltv,
  CASE WHEN CAST(trim(orig_dti) AS INT) BETWEEN 0 AND 65
       THEN CAST(trim(orig_dti) AS INT) END                             AS orig_dti,
  CAST(trim(orig_upb) AS DOUBLE)                                        AS orig_upb,
  CASE WHEN CAST(trim(orig_ltv) AS INT) BETWEEN 1 AND 998
       THEN CAST(trim(orig_ltv) AS INT) END                             AS orig_ltv,
  CAST(trim(orig_interest_rate) AS DOUBLE)                              AS orig_interest_rate,
  CASE WHEN trim(channel) IN ('R', 'B', 'C', 'T') THEN trim(channel) END AS channel,
  trim(amortization_type)                                               AS amortization_type,
  NULLIF(trim(property_state), '')                                      AS property_state,
  CASE WHEN trim(property_type) IN ('SF', 'CO', 'PU', 'MH', 'CP')
       THEN trim(property_type) END                                     AS property_type,
  CASE WHEN trim(loan_purpose) IN ('P', 'C', 'N', 'R')
       THEN trim(loan_purpose) END                                      AS loan_purpose,
  CAST(trim(orig_loan_term) AS INT)                                     AS orig_loan_term,
  CASE WHEN CAST(trim(num_borrowers) AS INT) = 1 THEN '1'
       WHEN CAST(trim(num_borrowers) AS INT) BETWEEN 2 AND 10 THEN '2+' END AS num_borrowers_group,
  CASE WHEN trim(relief_refi_indicator) = 'Y' THEN 'Y' ELSE 'N' END     AS relief_refi_indicator
FROM raw_orig

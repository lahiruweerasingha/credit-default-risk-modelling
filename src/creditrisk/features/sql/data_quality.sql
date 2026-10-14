-- Counts of sentinel / invalid codes in the raw origination file, per vintage.
SELECT
  vintage,
  COUNT(*)                                                                    AS n_loans,
  SUM(CASE WHEN trim(credit_score) = '9999' THEN 1 ELSE 0 END)                AS credit_score_9999,
  SUM(CASE WHEN trim(orig_dti) = '999' THEN 1 ELSE 0 END)                     AS dti_999,
  SUM(CASE WHEN trim(orig_ltv) = '999' THEN 1 ELSE 0 END)                     AS ltv_999,
  SUM(CASE WHEN trim(orig_cltv) = '999' THEN 1 ELSE 0 END)                    AS cltv_999,
  SUM(CASE WHEN trim(mi_pct) = '999' THEN 1 ELSE 0 END)                       AS mi_pct_999,
  SUM(CASE WHEN trim(num_units) = '99' THEN 1 ELSE 0 END)                     AS units_99,
  SUM(CASE WHEN trim(occupancy_status) = '9' THEN 1 ELSE 0 END)               AS occupancy_9,
  SUM(CASE WHEN trim(channel) = '9' THEN 1 ELSE 0 END)                        AS channel_9,
  SUM(CASE WHEN trim(property_type) = '99' THEN 1 ELSE 0 END)                 AS property_type_99,
  SUM(CASE WHEN trim(first_time_homebuyer_flag) = '9' THEN 1 ELSE 0 END)      AS first_time_buyer_9,
  SUM(CASE WHEN trim(loan_purpose) = '9' THEN 1 ELSE 0 END)                   AS loan_purpose_9,
  SUM(CASE WHEN trim(num_borrowers) = '99' THEN 1 ELSE 0 END)                 AS num_borrowers_99,
  SUM(CASE WHEN trim(relief_refi_indicator) = 'Y' THEN 1 ELSE 0 END)          AS relief_refi,
  SUM(CASE WHEN trim(amortization_type) <> 'FRM' THEN 1 ELSE 0 END)           AS non_frm
FROM raw_orig
GROUP BY vintage
ORDER BY vintage

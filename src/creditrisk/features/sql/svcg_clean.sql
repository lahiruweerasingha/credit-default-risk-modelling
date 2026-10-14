-- Typed monthly performance records, one row per loan per reporting month.
-- per_idx is a month index (year * 12 + month - 1) so month arithmetic is plain subtraction.
-- Delinquency status 'RA' (REO acquired) is flagged separately; non-numeric codes become NULL.
SELECT
  trim(loan_seq_no)                                                     AS loan_seq_no,
  CAST(substr(trim(monthly_reporting_period), 1, 4) AS INT) * 12
    + CAST(substr(trim(monthly_reporting_period), 5, 2) AS INT) - 1     AS per_idx,
  CAST(trim(current_actual_upb) AS DOUBLE)                              AS current_upb,
  CASE WHEN trim(current_delinquency_status) = 'RA' THEN 1 ELSE 0 END   AS is_reo,
  CASE WHEN trim(current_delinquency_status) RLIKE '^[0-9]+$'
       THEN CAST(trim(current_delinquency_status) AS INT) END           AS dlq,
  CASE WHEN trim(modification_flag) IN ('Y', 'P') THEN 1 ELSE 0 END     AS is_modified,
  CASE WHEN trim(zero_balance_code) RLIKE '^[0-9]+$'
       THEN lpad(trim(zero_balance_code), 2, '0') END                   AS zero_balance_code,
  CAST(trim(current_interest_rate) AS DOUBLE)                           AS current_interest_rate,
  -- 'C' appears in the delivered files alongside the documented 'Y' / 'P'
  CASE WHEN trim(payment_deferral_flag) IN ('Y', 'P', 'C') THEN 1 ELSE 0 END AS is_payment_deferral,
  CASE WHEN trim(borrower_assistance_status_code) IN ('F', 'R', 'T')
       THEN 1 ELSE 0 END                                                AS is_assistance,
  CAST(trim(actual_loss) AS DOUBLE)                                     AS actual_loss,
  CAST(trim(zero_balance_removal_upb) AS DOUBLE)                        AS zero_balance_removal_upb,
  NULLIF(trim(defect_settlement_date), '')                              AS defect_settlement_date
FROM raw_svcg

-- Raw performance-file checks: row counts, non-numeric delinquency codes, REO rows,
-- duplicate (loan, month) keys and the reporting-period range.
SELECT
  COUNT(*)                                                                     AS n_rows,
  COUNT(DISTINCT trim(loan_seq_no))                                            AS n_loans,
  SUM(CASE WHEN trim(current_delinquency_status) = 'RA' THEN 1 ELSE 0 END)     AS dlq_ra,
  SUM(CASE WHEN NOT trim(current_delinquency_status) RLIKE '^[0-9]+$'
            AND trim(current_delinquency_status) <> 'RA' THEN 1 ELSE 0 END)    AS dlq_other_non_numeric,
  COUNT(*) - COUNT(DISTINCT trim(loan_seq_no), trim(monthly_reporting_period)) AS duplicate_loan_months,
  MIN(trim(monthly_reporting_period))                                          AS first_period,
  MAX(trim(monthly_reporting_period))                                          AS last_period
FROM raw_svcg

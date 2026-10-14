-- Observation population: fixed-rate loans still on book at the observation month
-- (month {obs_age} after the first payment), not terminated, not REO and not already
-- {default_dpd}+ statuses delinquent (i.e. performing or early-stage delinquent).
SELECT
  o.loan_seq_no,
  o.vintage,
  o.fp_idx,
  o.fp_idx + {obs_age} - 1 AS obs_idx
FROM orig_clean o
JOIN svcg_clean s
  ON s.loan_seq_no = o.loan_seq_no
 AND s.per_idx = o.fp_idx + {obs_age} - 1
WHERE o.amortization_type = 'FRM'
  AND s.zero_balance_code IS NULL
  AND s.is_reo = 0
  AND s.dlq IS NOT NULL
  AND s.dlq < {default_dpd}

-- Behavioural aggregates per loan from performance history up to the observation month.
-- LEAKAGE GUARD: the WHERE clause keeps only rows with per_idx <= obs_idx. Nothing from the
-- outcome window and no label column is visible to this query (enforced by tests/test_leakage.py).
WITH hist AS (
  SELECT s.loan_seq_no, s.per_idx, s.dlq, s.current_upb, s.current_interest_rate,
         s.is_modified, s.is_payment_deferral, s.is_assistance, p.obs_idx
  FROM svcg_clean s
  JOIN population p ON s.loan_seq_no = p.loan_seq_no
  WHERE s.per_idx <= p.obs_idx
),
lagged AS (
  SELECT *, LAG(dlq) OVER (PARTITION BY loan_seq_no ORDER BY per_idx) AS prev_dlq
  FROM hist
)
SELECT
  loan_seq_no,
  COUNT(*)                                                              AS n_months_observed,
  MAX(CASE WHEN per_idx = obs_idx THEN dlq END)                         AS dlq_at_obs,
  MAX(CASE WHEN per_idx = obs_idx THEN current_upb END)                 AS upb_at_obs,
  MAX(CASE WHEN per_idx = obs_idx THEN current_interest_rate END)       AS rate_at_obs,
  MAX(dlq)                                                              AS max_dlq_12m,
  SUM(CASE WHEN dlq >= 1 THEN 1 ELSE 0 END)                             AS n_months_dlq30_12m,
  SUM(CASE WHEN dlq >= 1 AND per_idx > obs_idx - 6 THEN 1 ELSE 0 END)   AS n_months_dlq30_6m,
  MAX(obs_idx) - MAX(CASE WHEN dlq >= 1 THEN per_idx END)               AS months_since_last_dlq,
  SUM(CASE WHEN prev_dlq >= 1 AND dlq = 0 THEN 1 ELSE 0 END)            AS n_cures_12m,
  MAX(is_modified)                                                      AS ever_modified,
  MAX(is_payment_deferral)                                              AS ever_payment_deferral,
  MAX(is_assistance)                                                    AS ever_assistance
FROM lagged
GROUP BY loan_seq_no

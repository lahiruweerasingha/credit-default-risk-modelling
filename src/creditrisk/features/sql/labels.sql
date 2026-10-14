-- Outcome per loan over months obs+1 .. obs+{window}. This is the ONLY query that reads
-- performance rows after the observation month. Default = {default_dpd}+ status (90+ DPD),
-- REO acquisition, or a credit-event zero-balance code ({zbc_list}).
WITH fut AS (
  SELECT s.loan_seq_no, s.per_idx, s.dlq, s.is_reo, s.zero_balance_code
  FROM svcg_clean s
  JOIN population p ON s.loan_seq_no = p.loan_seq_no
  WHERE s.per_idx > p.obs_idx
    AND s.per_idx <= p.obs_idx + {window}
),
agg AS (
  SELECT
    p.loan_seq_no,
    p.obs_idx,
    MAX(CASE WHEN f.dlq >= {default_dpd} OR f.is_reo = 1
                  OR f.zero_balance_code IN ({zbc_list}) THEN 1 ELSE 0 END) AS is_default,
    MAX(CASE WHEN f.zero_balance_code IS NOT NULL THEN 1 ELSE 0 END)        AS terminated,
    MAX(f.per_idx)                                                          AS last_per_idx
  FROM population p
  LEFT JOIN fut f ON f.loan_seq_no = p.loan_seq_no
  GROUP BY p.loan_seq_no, p.obs_idx
)
SELECT
  loan_seq_no,
  COALESCE(is_default, 0) AS target,
  -- The outcome is known if the loan defaulted, terminated, or was observed to the window end.
  CASE WHEN is_default = 1 OR terminated = 1 OR last_per_idx = obs_idx + {window}
       THEN 1 ELSE 0 END  AS outcome_complete
FROM agg

-- Realised losses on credit-event terminations, used only for the illustrative ECL
-- (never as a model feature). Loans with an underwriting/servicing defect and loans whose
-- loss is not yet populated (disposed within 3 months of the data cut-off) are excluded.
SELECT
  s.loan_seq_no,
  o.vintage,
  s.zero_balance_code,
  s.zero_balance_removal_upb,
  s.actual_loss
FROM svcg_clean s
JOIN orig_clean o ON o.loan_seq_no = s.loan_seq_no
WHERE s.zero_balance_code IN ({zbc_list})
  AND s.defect_settlement_date IS NULL
  AND s.actual_loss IS NOT NULL
  AND s.zero_balance_removal_upb > 0

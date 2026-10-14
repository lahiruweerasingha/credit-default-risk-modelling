-- One row per loan in the observation population: origination + behavioural features.
-- No label information is referenced here.
WITH orig_ctx AS (
  -- Rate spread vs loans originated in the same quarter with the same term. This compares
  -- each loan with its peers' note rates only; it does not use any outcome information.
  SELECT o.*,
         o.orig_interest_rate
           - AVG(o.orig_interest_rate) OVER (PARTITION BY o.orig_quarter, o.orig_loan_term)
           AS rate_spread_at_orig
  FROM orig_clean o
)
SELECT
  p.loan_seq_no,
  p.vintage,
  o.credit_score,
  o.orig_ltv,
  o.orig_cltv,
  o.orig_dti,
  o.orig_upb,
  o.orig_interest_rate,
  o.rate_spread_at_orig,
  o.mi_pct,
  o.orig_loan_term,
  o.num_units,
  o.num_borrowers_group,
  o.first_time_homebuyer_flag,
  o.occupancy_status,
  o.channel,
  o.loan_purpose,
  o.property_type,
  o.property_state,
  o.relief_refi_indicator,
  b.dlq_at_obs,
  b.max_dlq_12m,
  b.n_months_dlq30_12m,
  b.n_months_dlq30_6m,
  b.months_since_last_dlq,
  b.n_cures_12m,
  1 - b.upb_at_obs / o.orig_upb                                         AS upb_paydown_ratio,
  -- Contractual balance after {obs_age} payments on a level-payment fixed-rate loan:
  -- B_k = P * ((1+r)^n - (1+r)^k) / ((1+r)^n - 1), r = monthly rate, n = term, k = payments.
  (CASE WHEN o.orig_interest_rate > 0 THEN
        o.orig_upb * (POWER(1 + o.orig_interest_rate / 1200, o.orig_loan_term)
                      - POWER(1 + o.orig_interest_rate / 1200, {obs_age}))
                   / (POWER(1 + o.orig_interest_rate / 1200, o.orig_loan_term) - 1)
        ELSE o.orig_upb * (1 - {obs_age} / o.orig_loan_term) END
   - b.upb_at_obs) / o.orig_upb                                         AS excess_paydown_ratio,
  b.rate_at_obs - o.orig_interest_rate                                  AS rate_change,
  b.ever_modified,
  b.ever_payment_deferral,
  b.ever_assistance,
  b.upb_at_obs,
  b.n_months_observed
FROM population p
JOIN orig_ctx o ON o.loan_seq_no = p.loan_seq_no
JOIN behaviour b ON b.loan_seq_no = p.loan_seq_no

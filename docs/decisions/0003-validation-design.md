# ADR 0003 — Validation design

**Status:** accepted

## Decision
- Development vintages are split **stratified on (vintage, target)** into train 60% / validation
  20% / in-time test 20% with a fixed seed. The assignment is saved and re-checked at evaluation.
- **Tuning uses validation only**: scorecard regularisation C, XGBoost random search with early
  stopping, MLlib regParam.
- `creditrisk.cli evaluate` scores the **in-time test** and the **out-of-time** vintages once.
- Metrics: ROC-AUC, Gini, KS, average precision, Brier score, decile reliability, default rate by
  PD band, PSI of scores (train vs test, train vs OOT) and of each numeric feature (train vs OOT).
- Uncertainty: 95% intervals for AUC, Gini and KS from a **paired, stratified bootstrap**
  (1,000 resamples; defaults and non-defaults resampled separately so every resample has the same
  default count). Paired means every model is scored on the same resampled loans, so the interval
  for the scorecard-vs-XGBoost Gini gap is valid. A model is only called better when that gap
  interval excludes zero; the README wording is generated from this test.
- The cut-off is fixed on validation (approve the best-scoring 90%) and applied unchanged to test and
  OOT.

## Notes
- Train-vs-test PSI is close to zero by construction under a random split; it is reported only as a
  sanity check. Train-vs-OOT PSI is the meaningful stability measure.
- Estimated LTV (ELTV) is excluded: it is only populated from April 2017, so it would be missing for
  every development observation and partly present for OOT — a pure availability shift.

# ADR 0004 — WoE logistic scorecard as champion

**Status:** accepted

## Decision
- **Custom monotonic WoE binner** (`scorecard/binning.py`) rather than a library: quantile pre-bins,
  trend from Spearman correlation, adjacent-bin merging until bad rates are strictly monotonic,
  minimum bin share 5%, separate missing bin (neutral WoE if fewer than 50 missing). It is short,
  fully unit-tested, and has no solver dependency.
- **Selection:** IV ≥ 0.02; IV > 0.5 flagged for leakage review (behavioural delinquency features are
  expected to be strong, so they are reviewed rather than auto-dropped); greedy correlation filter
  |ρ| ≤ 0.7 on WoE values, highest IV first; at most 15 characteristics.
- **Model:** L2 logistic regression on WoE, C chosen on validation AUC. Every coefficient must be
  negative (WoE = ln(%good/%bad)); offenders are removed one at a time.
- **Scaling:** 600 points at 50:1 good:bad odds, 20 points to double the odds
  (factor = 20 / ln 2 = 28.854, offset = 600 − 28.854 × ln 50 = 487.123). Attribute points:
  −(βᵢ·WoEᵢⱼ + α/n)·factor + offset/n. XGBoost PDs use the same scale so both models are comparable.
- **Reason codes:** points lost on each characteristic versus its best attribute, top four.

## Consequences
The scorecard can be printed on a page and audited attribute by attribute; the cost in
discrimination versus XGBoost is measured and reported, not assumed.

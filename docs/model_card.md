# Model card — behavioural mortgage PD (champion scorecard, XGBoost challenger)

## Overview
| | |
|---|---|
| Purpose | Estimate the probability that a performing US fixed-rate mortgage, observed at its 12th scheduled payment, becomes 90+ days past due or has a credit event in the next 24 months |
| Intended use | Portfolio project demonstrating bank-style model development and validation. Illustrative decision: an account-management action at month 12 (e.g. further-advance eligibility, early-collections prioritisation) |
| Not for | Any real lending, pricing, provisioning or regulatory capital decision |
| Models | Champion: WoE logistic scorecard (points, 600 @ 50:1, PDO 20). Challenger: XGBoost with TreeSHAP reason codes. Baseline: Spark MLlib logistic regression |
| Data | Freddie Mac Single-Family Loan-Level Dataset, official sample files, vintages 2010–2016 (used under Freddie Mac's terms; data not redistributed) |
| Owner | Project author |

## Target and population
- **Observation point:** month 12 after the first payment date.
- **Population:** fixed-rate loans on book at month 12, not terminated, not REO, not 90+ DPD.
- **Default (target = 1):** in months 13–36, delinquency status ≥ 3 (90+ DPD), REO acquisition, or a
  zero-balance code of 02 (third-party sale), 03 (short sale / charge-off) or 09 (REO disposition).
- Prepayment in the window without prior default counts as non-default. Loans with an incomplete
  outcome window are excluded and counted.

## Features
30 candidate features: 18 origination characteristics and 12 behavioural aggregates over months 1–12
(`src/creditrisk/features/catalog.py`). Postal code, MSA, seller and servicer are excluded
(ADR 0005). Estimated LTV is excluded because of an availability shift (ADR 0003).

## Validation design
Stratified 60/20/20 split of 2010–2014 vintages; tuning on validation only; in-time test and
2015–2016 out-of-time vintages scored once. Metrics, PSI, cut-off and segment tables: see README
"Results" (rendered from `reports/results.json`).

## Explainability
- Scorecard: published points table (`reports/scorecard_points.csv`); reason codes = characteristics
  with the most points lost versus the best attribute.
- XGBoost: exact TreeSHAP contributions on the log-odds scale; reason codes = largest positive
  contributions. SHAP explains the model, not causation.

## Fairness
The data contains no sex, age, race or ethnicity, so a protected-attribute fairness test is **not
possible**. The project reports approval and default rates at the cut-off for non-protected segments
(first-time homebuyer, number of borrowers, occupancy). Differences between segments are
descriptive, not evidence of unfair treatment or its absence.

## Validation outcome (see README "Results" for the numbers)
- In-time: discrimination and calibration-in-the-large are good for both models.
- Champion vs challenger: the Gini gap is tested with a paired, stratified bootstrap; README
  "Key findings" states, from that test, whether the gap is statistically clear in time and out of
  time.
- Out-of-time (2015-2016): ranking holds (band default rates stay monotonic) but **both models
  under-predict PD**. The largest characteristic shift is the absolute note rate, which partly
  encodes the vintage. **Status: would not pass validation for PD-level use without
  recalibration**; acceptable for rank-ordering. The model was deliberately not changed after
  seeing out-of-time results.

## Limitations
- US conforming mortgages bought by Freddie Mac: a selected, prime-skewed population. Results do not
  transfer to other products or markets (including Sri Lanka) without re-development.
- Out-of-time test covers 2015–2016 vintages only; all outcome windows end before COVID-19
  forbearance, so behaviour under that regime is untested.
- Accepted, purchased loans only — no reject inference.
- 24-month PD from month 12 is neither a 12-month PD nor a lifetime PD; the ECL is illustrative and
  not SLFRS 9 / IFRS 9 compliant (no staging, no SICR, no forward-looking macro scenarios).
- Low default rates make band-level estimates in the best bands noisy.
- Out-of-time PD under-prediction (above): absolute interest rate is time-dependent; a
  recalibration layer and a rate-spread-only specification are needed before PD-level use.
- Behavioural inputs at scoring time must be computed exactly as in training (same Spark SQL).

## Monitoring (if this were deployed)
Monthly score PSI and characteristic PSI against the development sample, band-level default rates
versus predicted PD once outcomes mature, reason-code distribution drift, and override rates.

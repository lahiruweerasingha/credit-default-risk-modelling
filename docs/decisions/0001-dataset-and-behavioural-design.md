# ADR 0001 — Dataset and modelling design: behavioural PD on Freddie Mac loan-level data

**Status:** accepted

## Context
The project was first scoped on the Kaggle *Home Credit Default Risk* data. Its competition-specific
rule restricts use of the data to "the purposes of the Competition", overriding the general rule that
allows academic and non-commercial use, so a public portfolio project on it is not clearly permitted.

The Freddie Mac Single-Family Loan-Level Dataset terms (effective November 2025) allow academic or
research use and publishing results and derived products, provided distribution is non-commercial and
cannot be used to recreate the data or identify individuals. Redistribution of the data itself is
prohibited.

## Decision
- Use the official **sample** files (50,000 loans per vintage) for vintages 2010–2016.
- Build a **behavioural** PD model: observation point at the 12th scheduled payment; features from
  origination data plus months 1–12 of performance; target = 90+ days past due, REO acquisition or a
  credit-event termination (zero-balance codes 02, 03, 09) in months 13–36.
- Exclude loans already 90+ DPD, REO or terminated at the observation point, and ARMs.
- Development vintages 2010–2014; out-of-time (OOT) vintages 2015–2016, so every outcome window
  ends before the COVID-19 forbearance period (from April 2020) distorted delinquency reporting.

## Consequences
- A true out-of-time test is possible (the Kaggle data had no dates).
- Protected attributes (sex, age, race) are not in the data; the fairness review is limited to
  non-protected segments and stated as a limitation.
- The default definition is close to the Basel 90-DPD definition, unlike the Kaggle target.
- No raw or loan-level derived data, and no individual loan rows, are committed or published.
- Post-2009 mortgage default rates are low; class imbalance is handled by metrics choice (AUC, Gini,
  KS, PR) and calibration checks rather than re-sampling.

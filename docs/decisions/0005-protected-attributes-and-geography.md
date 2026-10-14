# ADR 0005 — Protected attributes, geography and the fairness review

**Status:** accepted

## Context
The dataset contains no borrower sex, age, race or ethnicity. It does contain property state,
3-digit postal code and MSA.

## Decision
- No protected attribute can enter a model because none is present.
- Postal code and MSA are **excluded**: fine geography is a well-known proxy for protected
  characteristics and carries redlining risk. Property **state** is kept: state-level foreclosure law
  is a legitimate driver of mortgage outcomes; its effect is visible in the scorecard points table.
- The fairness section compares approval and default rates at the chosen cut-off across the
  available **non-protected** segments (first-time homebuyer, number of borrowers, occupancy). This
  is a disparity check, not a fair-lending test, and it is reported as such.
- Linking to other data (e.g. HMDA) to infer demographics is not done: the Freddie Mac terms prohibit
  linking that could identify individuals, and loan-level matching is not possible anyway.

## Consequences
A real fair-lending review would need applicant demographics and is listed under "what I'd do next".

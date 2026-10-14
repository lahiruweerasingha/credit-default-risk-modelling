# Known data issues and how they are handled

All handling lives in SQL under `src/creditrisk/features/sql/`; the counts below are produced by
`data_quality.sql` / `data_quality_perf.sql` on every run and rendered by `scripts/render_results.py`.

| Issue (from the Freddie Mac file layout) | Handling |
|---|---|
| **Delivered files differ from the user guide:** origination has 31 columns (servicer name removed, later fields shift left by one); performance has 35 (three undocumented trailing columns); performance file is `sample_perf_YYYY.txt` | layout in `data/schema.py` corrected against the delivered files and verified by value distributions (e.g. relief-refinance `Y` count equals the count of populated pre-relief loan numbers) |
| Payment-deferral flag uses an undocumented code `C` besides `Y` / `P` | all three treated as "deferral granted" |
| Delinquency status is two-digit (`00`, `01`, ...) | cast to integer |
| Credit score `9999` = not available (also anything outside 300–850) | → NULL; scorecard missing bin, XGBoost native missing |
| Original DTI `999` = not available or > 65%; **all HARP / relief-refinance loans are 999** | → NULL; relief-refinance flag kept as a feature so the missingness is explained |
| Original LTV / CLTV `999` = not available (and CLTV < LTV is reported as 999) | → NULL |
| MI % `999` = not available; `0` = no MI | 999 → NULL, 0 kept |
| Number of units `99`, occupancy / channel / first-time-buyer / purpose `9`, property type `99`, borrowers `99` | → NULL |
| Number of borrowers coded `01/02` up to 2018Q1 and `01..10` afterwards | grouped to `1` vs `2+` so all vintages share one definition |
| Delinquency status `RA` = REO acquired (non-numeric) | separate `is_reo` flag; counts as default in the outcome window |
| Other non-numeric delinquency codes | → NULL; a loan with NULL status at month 12 is excluded from the population |
| UPB rounded to the nearest $1,000 for the first six months | features use the month-12 balance, which is not rounded |
| Loan age resets on modification | the observation month is computed from the first-payment date, not from loan age |
| Estimated LTV only populated from April 2017 | **excluded** — it would be missing for every development observation (availability shift) |
| Actual loss is null for disposals within 3 months of the data cut-off and for defect loans | excluded from the LGD sample |
| Net sale proceeds can hold codes (`C`, `U`) | not used; LGD uses the `actual_loss` field |
| ARMs, a small share of the sample | out of scope (fixed-rate only) |
| Loans whose outcome window is not fully observed (no default, no termination, data ends early) | flagged `outcome_complete = 0` and excluded from modelling; count reported |
| Duplicate (loan, month) records | counted on every run (expected 0) |

<!-- DQ:START -->
Counts from the run (raw origination files, by vintage):

| vintage | n_loans | credit_score_9999 | dti_999 | ltv_999 | cltv_999 | mi_pct_999 | units_99 | occupancy_9 | channel_9 | property_type_99 | first_time_buyer_9 | loan_purpose_9 | num_borrowers_99 | relief_refi | non_frm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2010 | 50000 | 3 | 14662 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 15 | 0 | 1 | 14641 | 0 |
| 2011 | 50000 | 0 | 15498 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 52 | 0 | 0 | 15493 | 0 |
| 2012 | 50000 | 0 | 17400 | 3 | 3 | 2 | 62 | 0 | 0 | 0 | 0 | 0 | 0 | 17399 | 0 |
| 2013 | 50000 | 1 | 14384 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 14382 | 0 |
| 2014 | 50000 | 13 | 7309 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 7307 | 0 |
| 2015 | 50000 | 0 | 4020 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 4017 | 0 |
| 2016 | 50000 | 0 | 2528 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 2525 | 0 |

Performance files: 26,027,039 rows for 350,000 loans, reporting periods 201001 to 202603; 'RA' (REO) rows: 6,955; other non-numeric delinquency codes: 0; duplicate loan-month keys: 0.

<!-- DQ:END -->

# ADR 0002 — Spark SQL for aggregation; single-node XGBoost for the challenger

**Status:** accepted

## Context
Monthly performance files hold one row per loan per month, tens of millions of rows across seven
vintages. The modelling table is one row per loan, a few hundred thousand rows.

## Decision
- **Spark SQL** (PySpark, standalone master + worker in Docker) for ingestion and aggregation. Each step
  is a separate `.sql` file under `src/creditrisk/features/sql/` so the logic is reviewable as SQL.
  The same code runs with `SPARK_MASTER=local[*]` in tests and CI.
- **Single-node XGBoost** (`hist`) rather than XGBoost-on-Spark. After aggregation the data fits in
  memory, distributed training adds coordination overhead for no accuracy gain, and exact TreeSHAP
  needs the native booster. SHAP values come from `pred_contribs=True`, which supports native
  categorical splits.
- **Spark MLlib logistic regression** is kept as a baseline to show what a plain linear model
  achieves on the same features.

## Consequences
- One Docker image serves master, worker and driver, so Python and Java versions always match.
- Executors and the driver share `./data` through a bind mount; on a real cluster this would be
  object storage.

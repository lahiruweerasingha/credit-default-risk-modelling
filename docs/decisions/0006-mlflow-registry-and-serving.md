# ADR 0006 — MLflow tracking, registry aliases and serving

**Status:** accepted

## Decision
- MLflow server in docker-compose (SQLite backend, artifacts proxied by the server).
- `train` logs parameters, validation metrics and the IV / points / tuning tables, and registers both
  models as pyfunc models whose `predict` returns PD, score, band and reason codes.
- Registry aliases: `creditrisk-scorecard@champion`, `creditrisk-xgboost@challenger`. The API loads
  by alias, so promoting a new version is an alias change, not a redeploy.
- `evaluate` logs test and OOT metrics to a separate run.
- FastAPI validates every input (ranges, code lists, unknown fields rejected); all fields are optional
  and missing values are scored through each model's missing-value handling.

## Consequences
In production the behavioural inputs would come from a feature store populated by the same Spark SQL;
the API contract documents that the caller supplies them here.

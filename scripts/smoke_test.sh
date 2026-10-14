#!/usr/bin/env bash
# Score the synthetic example loan against the running API with both models.
set -euo pipefail
URL="${API_URL:-http://localhost:8000}"
PAYLOAD="$(${PYTHON:-uv run --no-sync python} -c 'import json,sys; sys.path.insert(0,"src"); from creditrisk.api.schemas import EXAMPLE_REQUEST; print(json.dumps(EXAMPLE_REQUEST))')"
curl -fsS "$URL/health"; echo
for m in scorecard xgboost; do
  curl -fsS -X POST "$URL/score?model=$m" -H 'Content-Type: application/json' -d "$PAYLOAD"; echo
done

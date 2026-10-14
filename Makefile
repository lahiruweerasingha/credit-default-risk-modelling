COMPOSE := docker compose
RUN := $(COMPOSE) --profile jobs run --rm pipeline

.PHONY: help up down fetch features train evaluate pipeline test test-local lint serve smoke
.PHONY: system-test clean

help:  ## List targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

up:  ## Build images and start Spark (master + worker) and MLflow
	$(COMPOSE) build spark-master api
	$(COMPOSE) up -d spark-master spark-worker mlflow

down:  ## Stop everything
	$(COMPOSE) --profile jobs --profile serve down

fetch:  ## Verify and unzip the manually downloaded Freddie Mac sample files
	bash scripts/fetch_data.sh

features:  ## Spark SQL: raw files -> one row per loan at the observation point
	$(RUN) python -m creditrisk.cli features

train:  ## Fit scorecard, XGBoost and Spark MLlib LR; tune on validation; register in MLflow
	$(RUN) python -m creditrisk.cli train

evaluate:  ## Score test and out-of-time once; write reports/
	$(RUN) python -m creditrisk.cli evaluate

pipeline: features train evaluate  ## features + train + evaluate

test:  ## Full test suite (incl. Spark tests) inside the pipeline image
	$(COMPOSE) --profile jobs run --rm -e SPARK_MASTER=local[2] pipeline pytest -q

test-local:  ## Tests on the host (Spark tests skip without Java)
	uv run pytest -q

lint:  ## ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

serve:  ## Start the scoring API on http://localhost:8000 (loads models from MLflow)
	$(COMPOSE) --profile serve up -d --build api

smoke:  ## Call the running API with the synthetic example
	bash scripts/smoke_test.sh

system-test:  ## Whole pipeline + API on seeded SYNTHETIC data (outputs in data/synthetic/)
	$(RUN) python scripts/make_synthetic_data.py
	CREDITRISK_CONFIG=configs/synthetic.yaml $(RUN) python -m creditrisk.cli all
	CREDITRISK_CONFIG=configs/synthetic.yaml $(COMPOSE) --profile serve up -d --force-recreate api
	sleep 20
	bash scripts/smoke_test.sh

clean:  ## Remove generated data and models (keeps raw downloads)
	rm -rf data/interim data/processed data/models

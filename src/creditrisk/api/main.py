"""FastAPI scoring service.

GET  /health                    liveness + loaded model versions
POST /score?model=scorecard     PD, score, band and reason codes (model=xgboost for the challenger)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request

from creditrisk.api.schemas import ScoreRequest, ScoreResponse
from creditrisk.api.service import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "service", None) is None:  # tests inject a service directly
        from creditrisk.config import load_config

        app.state.service = ModelService.from_registry(load_config())
    yield


app = FastAPI(
    title="Behavioural mortgage PD API",
    version="0.1.0",
    description="Scores a loan at month 12: probability of 90+ DPD or a credit event in the "
    "next 24 months, a points score, a risk band and reason codes.",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict:
    service: ModelService = request.app.state.service
    return {"status": "ok", "models": {k: v.version for k, v in service.models.items()}}


@app.post("/score", response_model=ScoreResponse)
def score(
    request: Request,
    body: ScoreRequest,
    model: Literal["scorecard", "xgboost"] = Query("scorecard"),
) -> ScoreResponse:
    service: ModelService = request.app.state.service
    if model not in service.models:
        raise HTTPException(status_code=503, detail=f"model '{model}' not loaded")
    result = service.score(model, body.features.model_dump())
    return ScoreResponse(loan_id=body.loan_id, **result)

"""
serve.py
--------
FastAPI serving endpoint for the Fraud Detection Engine.

Loads:
  - FeaturePipeline from data/processed/feature_pipeline.pkl
  - Best XGBoost model from MLflow experiment 'fraud-detection-engine-v2'

Endpoints:
  POST /predict  — Score a transaction, return fraud probability + explanation
  GET  /health   — Health check
  GET  /model    — Current model metadata

Run with:
  uvicorn src.serve:app --reload --port 8000

Example:
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"TransactionAmt": 1500.0, "card4": "visa", "tx_hour": 3}'
"""
import json
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import shap
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
from features import FeaturePipeline

app = FastAPI(
    title="Fraud Detection Engine",
    description=(
        "Real-time transaction risk scoring using XGBoost + CatBoost ensemble. "
        "Returns fraud probability, risk decision, and top contributing features (SHAP). "
        "Built to demonstrate production ML engineering for risk-adaptive security systems."
    ),
    version="2.0.0"
)

# Global state
MODEL = None
PIPELINE = None
ISOLATION_FOREST = None
EXPLAINER = None
MODEL_METADATA = {}

ARTIFACTS_DIR = Path("data/processed")
PIPELINE_PATH = ARTIFACTS_DIR / "feature_pipeline.pkl"
MODEL_PATH = ARTIFACTS_DIR / "xgboost_model.pkl"
ISOLATION_FOREST_PATH = ARTIFACTS_DIR / "isolation_forest.pkl"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"


# ── Request / Response schemas ─────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    TransactionDT: Optional[float] = Field(None, ge=0)
    TransactionAmt: float       = Field(..., gt=0)
    ProductCD:      Optional[str]   = None
    card1:          Optional[float] = None
    card2:          Optional[float] = None
    card4:          Optional[str]   = None
    card6:          Optional[str]   = None
    P_emaildomain:  Optional[str]   = None
    R_emaildomain:  Optional[str]   = None
    addr1:          Optional[float] = None
    addr2:          Optional[float] = None
    dist1:          Optional[float] = None
    tx_hour:        Optional[int]   = Field(None, ge=0, le=23)
    DeviceType:     Optional[str]   = None
    DeviceInfo:     Optional[str]   = None
    id_01:          Optional[float] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "TransactionAmt": 1500.00,
                "card4": "visa",
                "card6": "debit",
                "P_emaildomain": "protonmail.com",
                "tx_hour": 3,
                "DeviceType": "mobile"
            }
        }
    }


class FraudPrediction(BaseModel):
    fraud_probability:  float
    risk_decision:      str
    risk_level:         str
    decision_threshold: float
    top_risk_factors:   list
    processing_time_ms: float
    model_version:      str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    pipeline_loaded: bool
    isolation_forest_loaded: bool
    model_version: str


# ── Risk decision ──────────────────────────────────────────────────────────────

def get_risk_level(prob: float) -> tuple:
    if prob >= 0.8:   return "BLOCK",   "CRITICAL"
    elif prob >= 0.5: return "REVIEW",  "HIGH"
    elif prob >= 0.3: return "MONITOR", "MEDIUM"
    else:             return "APPROVE", "LOW"

@app.on_event("startup")
async def load_model():
    global MODEL
    global PIPELINE
    global ISOLATION_FOREST
    global EXPLAINER
    global MODEL_METADATA

    missing_artifacts = []

    for path in (
        PIPELINE_PATH,
        MODEL_PATH,
        ISOLATION_FOREST_PATH,
        METADATA_PATH,
    ):
        if not path.exists():
            missing_artifacts.append(str(path))

    if missing_artifacts:
        print("WARNING: Missing production artifacts:")
        for artifact in missing_artifacts:
            print(f"  - {artifact}")
        print("Run: python src/pipeline.py --force-rebuild")
        return

    try:
        PIPELINE = FeaturePipeline.load(PIPELINE_PATH)

        with open(MODEL_PATH, "rb") as f:
            MODEL = pickle.load(f)

        with open(ISOLATION_FOREST_PATH, "rb") as f:
            ISOLATION_FOREST = pickle.load(f)

        with open(METADATA_PATH, "r") as f:
            MODEL_METADATA = json.load(f)

        EXPLAINER = shap.TreeExplainer(MODEL)

        print(
            f"Production model loaded successfully "
            f"(version {MODEL_METADATA.get('version', 'unknown')})"
        )

    except Exception as e:
        MODEL = None
        PIPELINE = None
        ISOLATION_FOREST = None
        EXPLAINER = None
        MODEL_METADATA = {}

        print(f"Model loading error: {e}")

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    all_loaded = all(
        artifact is not None
        for artifact in (MODEL, PIPELINE, ISOLATION_FOREST)
    )

    return HealthResponse(
        status="healthy" if all_loaded else "degraded",
        model_loaded=MODEL is not None,
        pipeline_loaded=PIPELINE is not None,
        isolation_forest_loaded=ISOLATION_FOREST is not None,
        model_version=MODEL_METADATA.get("version", "not_loaded"),
    )


@app.get("/model")
async def model_info():
    if not MODEL_METADATA:
        return {"status": "no model loaded", "instructions": "Run src/pipeline.py first"}
    return MODEL_METADATA


@app.post("/predict", response_model=FraudPrediction)
async def predict(request: TransactionRequest):
    """
    Score a transaction for fraud risk.

    Returns fraud probability, risk decision (APPROVE/MONITOR/REVIEW/BLOCK),
    risk level (LOW/MEDIUM/HIGH/CRITICAL), and top 3 SHAP features explaining
    why the decision was made.

    In Uber's production system this runs in <10ms backed by a feature store.
    This FastAPI implementation demonstrates the same architectural pattern.
    """
    if any(
        artifact is None
        for artifact in (
            MODEL,
            PIPELINE,
            ISOLATION_FOREST,
            EXPLAINER,
        )
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Production model artifacts are not fully loaded. "
                "Run src/pipeline.py --force-rebuild first."
            ),
        )

    t0 = time.perf_counter()

    # Convert request to dict and transform using fitted pipeline
    tx_dict = request.model_dump()
    X = PIPELINE.transform_single(tx_dict)

    # Generate the serving-time Isolation Forest anomaly score
    if ISOLATION_FOREST is None:
        raise HTTPException(
            status_code=503,
            detail="Isolation Forest artifact is not loaded.",
        )

    # Isolation Forest was trained before isolation_forest_score was added,
    # so use only the exact features seen during fitting.
    if hasattr(ISOLATION_FOREST, "feature_names_in_"):
        if_columns = list(ISOLATION_FOREST.feature_names_in_)
        X_if = X.reindex(columns=if_columns, fill_value=-999)
    else:
        X_if = X.drop(columns=["isolation_forest_score"], errors="ignore")

    raw_if_score = float(ISOLATION_FOREST.decision_function(X_if)[0])

    score_min = getattr(ISOLATION_FOREST, "score_min_", None)
    score_max = getattr(ISOLATION_FOREST, "score_max_", None)

    if score_min is None or score_max is None:
        raise HTTPException(
            status_code=503,
            detail="Isolation Forest normalization metadata is unavailable.",
        )

    normalized_if_score = 1 - (
        (raw_if_score - score_min)
        / (score_max - score_min + 1e-8)
    )

    normalized_if_score = float(np.clip(normalized_if_score, 0.0, 1.0))
    X["isolation_forest_score"] = normalized_if_score
    
    expected_features = MODEL_METADATA.get("features", [])

    if expected_features:
        missing_features = [
            feature for feature in expected_features
            if feature not in X.columns
        ]

        for feature in missing_features:
            X[feature] = 0.0

        X = X.reindex(columns=expected_features, fill_value=0.0)
    
    # Predict
    fraud_prob = float(MODEL.predict_proba(X)[0, 1])

    # Generate SHAP explanation using the startup-loaded explainer
    try:
        if EXPLAINER is None:
            raise RuntimeError("SHAP explainer is not loaded.")

        shap_values = EXPLAINER.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_row = np.asarray(shap_values).reshape(-1)

        top_indices = np.argsort(np.abs(shap_row))[::-1][:3]

        top_factors = [
            {
                "feature": X.columns[index],
                "shap_value": round(float(shap_row[index]), 4),
                "direction": (
                    "increases_risk"
                    if shap_row[index] > 0
                    else "decreases_risk"
                ),
            }
            for index in top_indices
        ]

    except Exception as exc:
        print(f"SHAP error: {exc}")
        top_factors = []

    threshold           = MODEL_METADATA.get("threshold", 0.3)
    decision, risk_level = get_risk_level(fraud_prob)
    elapsed_ms          = (time.perf_counter() - t0) * 1000

    return FraudPrediction(
        fraud_probability=round(fraud_prob, 4),
        risk_decision=decision,
        risk_level=risk_level,
        decision_threshold=threshold,
        top_risk_factors=top_factors,
        processing_time_ms=round(elapsed_ms, 2),
        model_version=MODEL_METADATA.get("version", "unknown")
    )
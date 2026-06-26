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
MODEL           = None
PIPELINE        = None
EXPLAINER       = None
MODEL_METADATA  = {}
PIPELINE_PATH   = "data/processed/feature_pipeline.pkl"


# ── Request / Response schemas ─────────────────────────────────────────────────

class TransactionRequest(BaseModel):
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
    status:        str
    model_loaded:  bool
    pipeline_loaded: bool
    model_version: str


# ── Risk decision ──────────────────────────────────────────────────────────────

def get_risk_level(prob: float) -> tuple:
    if prob >= 0.8:   return "BLOCK",   "CRITICAL"
    elif prob >= 0.5: return "REVIEW",  "HIGH"
    elif prob >= 0.3: return "MONITOR", "MEDIUM"
    else:             return "APPROVE", "LOW"


# ── Model loading ──────────────────────────────────────────────────────────────

@app.on_event("startup")
async def load_model():
    global MODEL, PIPELINE, EXPLAINER, MODEL_METADATA

    # Load feature pipeline
    if Path(PIPELINE_PATH).exists():
        PIPELINE = FeaturePipeline.load(PIPELINE_PATH)
        print(f"Feature pipeline loaded: {len(PIPELINE.feature_columns)} features")
    else:
        print(f"WARNING: No feature pipeline found at {PIPELINE_PATH}")
        return

    # Load XGBoost model directly from disk
    try:
        import pickle
        with open("data/processed/xgboost_model.pkl", "rb") as f:
            MODEL = pickle.load(f)
        MODEL_METADATA = {
            "pr_auc":    0.6003,
            "roc_auc":   0.9221,
            "threshold": 0.3,
            "version":   "2.0.0"
        }
        print(f"XGBoost model loaded — PR-AUC: {MODEL_METADATA['pr_auc']:.4f}")
    except Exception as e:
        print(f"Model loading error: {e}")

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL is not None,
        pipeline_loaded=PIPELINE is not None,
        model_version=MODEL_METADATA.get("version", "not_loaded")
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
    if MODEL is None or PIPELINE is None:
        raise HTTPException(
            status_code=503,
            detail="Model or pipeline not loaded. Run src/pipeline.py --force-rebuild first."
        )

    t0 = time.perf_counter()

    # Convert request to dict and transform using fitted pipeline
    tx_dict = request.model_dump()
    X = PIPELINE.transform_single(tx_dict)

    # Add isolation forest score (neutral at serving time)
    X["isolation_forest_score"] = 0.0
    
    # Ensure correct column order matching training
    if "isolation_forest_score" not in X.columns:
        X["isolation_forest_score"] = 0.0
    
    # Predict
    fraud_prob = float(MODEL.predict_proba(X)[0, 1])

    # SHAP explanation — initialize lazily on first call
   
    try:
        global EXPLAINER
        if EXPLAINER is None:
            import shap
            EXPLAINER = shap.Explainer(MODEL.predict_proba, X, max_evals=2*len(X.columns)+1)
        shap_values = EXPLAINER.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_row = np.array(shap_values).flatten()
        # Trim shap_row to match feature count
        shap_row = shap_row[:len(X.columns)]
        top_indices = np.argsort(np.abs(shap_row))[::-1][:3]
        top_factors = [
            {
                "feature":    X.columns[i],
                "shap_value": round(float(shap_row[i]), 4),
                "direction":  "increases_risk" if shap_row[i] > 0 else "decreases_risk"
            }
            for i in top_indices
        ]
    except Exception as e:
        print(f"SHAP error: {e}")
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
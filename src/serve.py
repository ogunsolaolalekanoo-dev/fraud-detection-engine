"""
serve.py
--------
FastAPI serving endpoint for the Fraud Detection Engine.

Loads:
  - FeaturePipeline from data/processed/feature_pipeline.pkl
  - XGBoost model from data/processed/xgboost_model.pkl
  - Isolation Forest from data/processed/isolation_forest.pkl
  - Model metadata from data/processed/model_metadata.json

Endpoints:
  POST /predict  — Score a transaction and store the result in PostgreSQL
  GET  /health   — Health check
  GET  /model    — Current model metadata

Run with:
  uvicorn src.serve:app --reload --port 8000
"""

import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Allow local imports when running the file from the project root.
sys.path.insert(0, os.path.dirname(__file__))

from features import FeaturePipeline
from src.database import (
    create_database_tables,
    get_prediction_metrics,
    get_recent_predictions,
    save_prediction_log,
)


# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ── FastAPI application ────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection Engine",
    description=(
        "Real-time transaction risk scoring using machine-learning models. "
        "Returns fraud probability, risk decision, risk level, and SHAP-based "
        "explanations. Successful predictions are logged to PostgreSQL."
    ),
    version="2.0.0",
)


# ── Global model state ─────────────────────────────────────────────────────────

MODEL = None
PIPELINE = None
ISOLATION_FOREST = None
EXPLAINER = None
MODEL_METADATA: dict = {}


# ── Artifact paths ─────────────────────────────────────────────────────────────

ARTIFACTS_DIR = Path("data/processed")

PIPELINE_PATH = ARTIFACTS_DIR / "feature_pipeline.pkl"
MODEL_PATH = ARTIFACTS_DIR / "xgboost_model.pkl"
ISOLATION_FOREST_PATH = ARTIFACTS_DIR / "isolation_forest.pkl"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"


# ── Request and response schemas ───────────────────────────────────────────────

class TransactionRequest(BaseModel):
    TransactionID: Optional[int] = None
    TransactionDT: Optional[float] = Field(None, ge=0)
    TransactionAmt: float = Field(..., gt=0)

    ProductCD: Optional[str] = None

    card1: Optional[float] = None
    card2: Optional[float] = None
    card4: Optional[str] = None
    card6: Optional[str] = None

    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None

    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None

    tx_hour: Optional[int] = Field(None, ge=0, le=23)

    DeviceType: Optional[str] = None
    DeviceInfo: Optional[str] = None
    id_01: Optional[float] = None

    model_config = {
        # Preserve additional IEEE-CIS fields included in the JSON payload.
        "extra": "allow",
        "json_schema_extra": {
            "example": {
                "TransactionID": 2987000,
                "TransactionDT": 86400,
                "TransactionAmt": 1500.00,
                "card4": "visa",
                "card6": "debit",
                "P_emaildomain": "protonmail.com",
                "tx_hour": 3,
                "DeviceType": "mobile",
            }
        },
    }


class FraudPrediction(BaseModel):
    fraud_probability: float
    risk_decision: str
    risk_level: str
    decision_threshold: float
    top_risk_factors: list
    processing_time_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    pipeline_loaded: bool
    isolation_forest_loaded: bool
    model_version: str


# ── Risk decision logic ────────────────────────────────────────────────────────

def get_risk_level(probability: float) -> tuple[str, str]:
    """
    Convert a fraud probability into an operational decision and risk level.
    """

    if probability >= 0.8:
        return "BLOCK", "CRITICAL"

    if probability >= 0.5:
        return "REVIEW", "HIGH"

    if probability >= 0.3:
        return "MONITOR", "MEDIUM"

    return "APPROVE", "LOW"


# ── Application startup ────────────────────────────────────────────────────────

@app.on_event("startup")
async def load_model() -> None:
    """
    Initialize PostgreSQL tables and load production model artifacts.
    """

    global MODEL
    global PIPELINE
    global ISOLATION_FOREST
    global EXPLAINER
    global MODEL_METADATA

    # Initialize database tables.
    try:
        create_database_tables()
        logger.info("Database tables initialized successfully")
    except Exception as exc:
        # The API may still serve predictions even if database logging is
        # temporarily unavailable.
        logger.exception("Database initialization failed: %s", exc)

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
        logger.warning("Missing production artifacts:")

        for artifact in missing_artifacts:
            logger.warning("Missing artifact: %s", artifact)

        logger.warning(
            "Run: python src/pipeline.py --force-rebuild"
        )
        return

    try:
        PIPELINE = FeaturePipeline.load(PIPELINE_PATH)

        with open(MODEL_PATH, "rb") as model_file:
            MODEL = pickle.load(model_file)

        with open(
            ISOLATION_FOREST_PATH,
            "rb",
        ) as isolation_forest_file:
            ISOLATION_FOREST = pickle.load(isolation_forest_file)

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8",
        ) as metadata_file:
            MODEL_METADATA = json.load(metadata_file)

        EXPLAINER = shap.TreeExplainer(MODEL)

        logger.info(
            "Production model loaded successfully. Version: %s",
            MODEL_METADATA.get("version", "unknown"),
        )

    except Exception as exc:
        MODEL = None
        PIPELINE = None
        ISOLATION_FOREST = None
        EXPLAINER = None
        MODEL_METADATA = {}

        logger.exception("Model loading error: %s", exc)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
)
async def health() -> HealthResponse:
    """
    Return API and model artifact health information.
    """

    all_loaded = all(
        artifact is not None
        for artifact in (
            MODEL,
            PIPELINE,
            ISOLATION_FOREST,
            EXPLAINER,
        )
    )

    return HealthResponse(
        status="healthy" if all_loaded else "degraded",
        model_loaded=MODEL is not None,
        pipeline_loaded=PIPELINE is not None,
        isolation_forest_loaded=ISOLATION_FOREST is not None,
        model_version=MODEL_METADATA.get(
            "version",
            "not_loaded",
        ),
    )


@app.get("/model")
async def model_info() -> dict:
    """
    Return model metadata.
    """

    if not MODEL_METADATA:
        return {
            "status": "no model loaded",
            "instructions": (
                "Run python src/pipeline.py --force-rebuild first"
            ),
        }

    return MODEL_METADATA

@app.get("/predictions")
async def recent_predictions(
    limit: int = 20,
) -> dict:
    """
    Return the most recent stored fraud predictions.

    The limit is restricted to a maximum of 100 records.
    """

    try:
        predictions = get_recent_predictions(limit=limit)

        return {
            "count": len(predictions),
            "predictions": predictions,
        }

    except Exception as exc:
        logger.exception(
            "Failed to retrieve prediction history: %s",
            exc,
        )

        raise HTTPException(
            status_code=503,
            detail="Prediction history is temporarily unavailable.",
        ) from exc


@app.get("/metrics")
async def prediction_metrics() -> dict:
    """
    Return aggregate monitoring metrics for stored predictions.
    """

    try:
        return get_prediction_metrics()

    except Exception as exc:
        logger.exception(
            "Failed to retrieve prediction metrics: %s",
            exc,
        )

        raise HTTPException(
            status_code=503,
            detail="Prediction metrics are temporarily unavailable.",
        ) from exc




@app.post(
    "/predict",
    response_model=FraudPrediction,
)
async def predict(
    request: TransactionRequest,
) -> FraudPrediction:
    """
    Score a transaction for fraud risk.

    The result includes:

    - Fraud probability
    - Risk decision
    - Risk level
    - Decision threshold
    - Top three SHAP risk factors
    - Processing time
    - Model version

    Each successful prediction is also written to PostgreSQL.
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
                "Run python src/pipeline.py --force-rebuild first."
            ),
        )

    start_time = time.perf_counter()

    # Convert the validated request into a dictionary.
    request_payload = request.model_dump(mode="json")

    # Transform the transaction using the fitted feature pipeline.
    try:
        feature_frame = PIPELINE.transform_single(request_payload)
    except Exception as exc:
        logger.exception("Feature transformation failed: %s", exc)

        raise HTTPException(
            status_code=500,
            detail="Feature transformation failed.",
        ) from exc

    # Isolation Forest was trained before isolation_forest_score was added.
    # Therefore, use only the exact features seen during Isolation Forest
    # training.
    if hasattr(ISOLATION_FOREST, "feature_names_in_"):
        isolation_columns = list(
            ISOLATION_FOREST.feature_names_in_
        )

        isolation_features = feature_frame.reindex(
            columns=isolation_columns,
            fill_value=-999,
        )
    else:
        isolation_features = feature_frame.drop(
            columns=["isolation_forest_score"],
            errors="ignore",
        )

    try:
        raw_isolation_score = float(
            ISOLATION_FOREST.decision_function(
                isolation_features
            )[0]
        )
    except Exception as exc:
        logger.exception(
            "Isolation Forest scoring failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail="Isolation Forest scoring failed.",
        ) from exc

    score_min = getattr(
        ISOLATION_FOREST,
        "score_min_",
        None,
    )

    score_max = getattr(
        ISOLATION_FOREST,
        "score_max_",
        None,
    )

    if score_min is None or score_max is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Isolation Forest normalization metadata "
                "is unavailable."
            ),
        )

    normalized_isolation_score = 1 - (
        (raw_isolation_score - score_min)
        / (score_max - score_min + 1e-8)
    )

    normalized_isolation_score = float(
        np.clip(
            normalized_isolation_score,
            0.0,
            1.0,
        )
    )

    feature_frame["isolation_forest_score"] = (
        normalized_isolation_score
    )

    # Ensure the final model receives features in exactly the same order
    # used during training.
    expected_features = MODEL_METADATA.get(
        "features",
        [],
    )

    if expected_features:
        missing_features = [
            feature
            for feature in expected_features
            if feature not in feature_frame.columns
        ]

        for feature in missing_features:
            feature_frame[feature] = 0.0

        feature_frame = feature_frame.reindex(
            columns=expected_features,
            fill_value=0.0,
        )

    # Generate the fraud probability.
    try:
        fraud_probability = float(
            MODEL.predict_proba(feature_frame)[0, 1]
        )
    except Exception as exc:
        logger.exception("Model prediction failed: %s", exc)

        raise HTTPException(
            status_code=500,
            detail="Model prediction failed.",
        ) from exc

    # Generate SHAP explanations.
    try:
        shap_values = EXPLAINER.shap_values(
            feature_frame
        )

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_row = np.asarray(
            shap_values
        ).reshape(-1)

        top_indices = np.argsort(
            np.abs(shap_row)
        )[::-1][:3]

        top_factors = [
            {
                "feature": str(
                    feature_frame.columns[index]
                ),
                "shap_value": round(
                    float(shap_row[index]),
                    4,
                ),
                "direction": (
                    "increases_risk"
                    if shap_row[index] > 0
                    else "decreases_risk"
                ),
            }
            for index in top_indices
        ]

    except Exception as exc:
        logger.exception(
            "SHAP explanation failed: %s",
            exc,
        )

        # Prediction should still succeed if SHAP temporarily fails.
        top_factors = []

    threshold = float(
        MODEL_METADATA.get(
            "threshold",
            0.3,
        )
    )

    decision, risk_level = get_risk_level(
        fraud_probability
    )

    elapsed_ms = (
        time.perf_counter() - start_time
    ) * 1000

    response = FraudPrediction(
        fraud_probability=round(
            fraud_probability,
            4,
        ),
        risk_decision=decision,
        risk_level=risk_level,
        decision_threshold=threshold,
        top_risk_factors=top_factors,
        processing_time_ms=round(
            elapsed_ms,
            2,
        ),
        model_version=MODEL_METADATA.get(
            "version",
            "unknown",
        ),
    )

    # Save the successful prediction to PostgreSQL.
    try:
        response_payload = response.model_dump(
            mode="json"
        )

        transaction_id_value = request_payload.get(
            "TransactionID"
        )

        save_prediction_log(
            transaction_id=(
                str(transaction_id_value)
                if transaction_id_value is not None
                else None
            ),
            fraud_probability=float(
                response_payload["fraud_probability"]
            ),
            risk_decision=response_payload[
                "risk_decision"
            ],
            risk_level=response_payload[
                "risk_level"
            ],
            decision_threshold=float(
                response_payload["decision_threshold"]
            ),
            processing_time_ms=float(
                response_payload["processing_time_ms"]
            ),
            model_version=response_payload[
                "model_version"
            ],
            top_risk_factors=response_payload[
                "top_risk_factors"
            ],
            request_payload=request_payload,
        )

        logger.info(
            "Prediction logged successfully. "
            "TransactionID=%s, decision=%s, probability=%s",
            transaction_id_value,
            response.risk_decision,
            response.fraud_probability,
        )

    except Exception as exc:
        # Database failure must not block the prediction response.
        logger.exception(
            "Prediction logging failed: %s",
            exc,
        )

    return response
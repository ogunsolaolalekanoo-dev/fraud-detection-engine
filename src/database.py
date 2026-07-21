"""
database.py
-----------
PostgreSQL persistence and monitoring queries for the Fraud Detection Engine.
"""

import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    JSON,
    String,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://fraud_user:fraud_password@localhost:5432/fraud_db",
)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy database models."""

    pass


class PredictionLog(Base):
    """Stores each successful fraud prediction."""

    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    transaction_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    fraud_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    risk_decision: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        index=True,
    )

    risk_level: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        index=True,
    )

    decision_threshold: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    processing_time_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    model_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    top_risk_factors: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
    )

    request_payload: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def create_database_tables() -> None:
    """Create database tables that do not already exist."""

    Base.metadata.create_all(bind=engine)


def save_prediction_log(
    *,
    transaction_id: str | None,
    fraud_probability: float,
    risk_decision: str,
    risk_level: str,
    decision_threshold: float,
    processing_time_ms: float,
    model_version: str,
    top_risk_factors: list[dict[str, Any]],
    request_payload: dict[str, Any],
) -> PredictionLog:
    """Save a successful fraud prediction."""

    prediction_log = PredictionLog(
        transaction_id=transaction_id,
        fraud_probability=fraud_probability,
        risk_decision=risk_decision,
        risk_level=risk_level,
        decision_threshold=decision_threshold,
        processing_time_ms=processing_time_ms,
        model_version=model_version,
        top_risk_factors=top_risk_factors,
        request_payload=request_payload,
    )

    with SessionLocal() as session:
        session.add(prediction_log)
        session.commit()
        session.refresh(prediction_log)

        # Detach the object before the session closes.
        session.expunge(prediction_log)

    return prediction_log


def get_recent_predictions(
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Retrieve the most recent prediction records.

    The maximum allowed limit is 100 to prevent excessively large responses.
    """

    safe_limit = max(1, min(limit, 100))

    statement = (
        select(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .limit(safe_limit)
    )

    with SessionLocal() as session:
        records = session.scalars(statement).all()

        return [
            {
                "id": record.id,
                "transaction_id": record.transaction_id,
                "fraud_probability": record.fraud_probability,
                "risk_decision": record.risk_decision,
                "risk_level": record.risk_level,
                "decision_threshold": record.decision_threshold,
                "processing_time_ms": record.processing_time_ms,
                "model_version": record.model_version,
                "top_risk_factors": record.top_risk_factors,
                "created_at": record.created_at.isoformat(),
            }
            for record in records
        ]


def get_prediction_metrics() -> dict[str, Any]:
    """Calculate aggregate monitoring metrics from stored predictions."""

    with SessionLocal() as session:
        summary_statement = select(
            func.count(PredictionLog.id),
            func.avg(PredictionLog.fraud_probability),
            func.avg(PredictionLog.processing_time_ms),
            func.max(PredictionLog.created_at),
        )

        (
            total_predictions,
            average_fraud_probability,
            average_processing_time_ms,
            latest_prediction_at,
        ) = session.execute(summary_statement).one()

        decision_statement = (
            select(
                PredictionLog.risk_decision,
                func.count(PredictionLog.id),
            )
            .group_by(PredictionLog.risk_decision)
        )

        decision_counts = {
            decision: count
            for decision, count in session.execute(
                decision_statement
            ).all()
        }

        risk_level_statement = (
            select(
                PredictionLog.risk_level,
                func.count(PredictionLog.id),
            )
            .group_by(PredictionLog.risk_level)
        )

        risk_level_counts = {
            risk_level: count
            for risk_level, count in session.execute(
                risk_level_statement
            ).all()
        }

    return {
        "total_predictions": int(total_predictions or 0),
        "average_fraud_probability": round(
            float(average_fraud_probability or 0.0),
            4,
        ),
        "average_processing_time_ms": round(
            float(average_processing_time_ms or 0.0),
            2,
        ),
        "decision_counts": {
            "APPROVE": int(decision_counts.get("APPROVE", 0)),
            "MONITOR": int(decision_counts.get("MONITOR", 0)),
            "REVIEW": int(decision_counts.get("REVIEW", 0)),
            "BLOCK": int(decision_counts.get("BLOCK", 0)),
        },
        "risk_level_counts": {
            "LOW": int(risk_level_counts.get("LOW", 0)),
            "MEDIUM": int(risk_level_counts.get("MEDIUM", 0)),
            "HIGH": int(risk_level_counts.get("HIGH", 0)),
            "CRITICAL": int(risk_level_counts.get("CRITICAL", 0)),
        },
        "latest_prediction_at": (
            latest_prediction_at.isoformat()
            if latest_prediction_at is not None
            else None
        ),
    }
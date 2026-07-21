import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, Integer, JSON, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://fraud_user:fraud_password@localhost:5432/fraud_db",
)


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    fraud_probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_decision: Mapped[str] = mapped_column(String(30), nullable=False)
    risk_level: Mapped[str] = mapped_column(String(30), nullable=False)
    decision_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    processing_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
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

    return prediction_log
"""
model.py
--------
Training, evaluation, and experiment tracking for the Fraud Detection Engine.

Model strategy:
  1. Isolation Forest  — unsupervised anomaly detection baseline (no labels needed)
  2. LightGBM          — primary supervised classifier (handles imbalance natively)
  3. XGBoost           — comparison model

Evaluation philosophy:
  ROC-AUC flatters imbalanced classifiers. In fraud detection, we optimize
  Precision-Recall AUC (PR-AUC) instead — it is sensitive to performance on
  the minority (fraud) class, which is the class that actually matters.

  We also tune the decision threshold explicitly: the default 0.5 is almost
  never optimal for imbalanced problems. We optimize F-beta (beta=2) to
  weight recall higher than precision — missing fraud is more costly than
  a false alarm.
"""

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import shap
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score,
    classification_report, confusion_matrix
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from features import load_features, get_feature_columns

RANDOM_STATE = 42
MLFLOW_EXPERIMENT = "fraud-detection-engine"


# ── Threshold tuning ───────────────────────────────────────────────────────────

def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray, beta: float = 2.0) -> float:
    """
    Find the decision threshold that maximizes F-beta score.

    beta=2 weights recall twice as heavily as precision — in fraud detection,
    a missed fraud (false negative) is more costly than a false alarm.
    This mirrors how Uber's risk systems would tune thresholds: based on the
    asymmetric cost of different error types.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    f_beta_scores = (
        (1 + beta**2) * precisions * recalls /
        ((beta**2 * precisions) + recalls + 1e-8)
    )

    best_idx = np.argmax(f_beta_scores[:-1])
    best_threshold = thresholds[best_idx]
    best_f_beta = f_beta_scores[best_idx]

    print(f"  Optimal threshold: {best_threshold:.4f} (F-{beta} = {best_f_beta:.4f})")
    return float(best_threshold)


# ── Isolation Forest (unsupervised baseline) ───────────────────────────────────

def train_isolation_forest(X_train: pd.DataFrame, X_val: pd.DataFrame,
                            y_val: pd.Series) -> dict:
    """
    Isolation Forest scores anomalies without using labels.

    This is architecturally important: in production, you often encounter
    new fraud patterns before you have labeled examples. An unsupervised
    layer catches these zero-day patterns while the supervised model handles
    known fraud signatures — exactly the layered defense Uber needs.
    """
    print("\nTraining Isolation Forest (unsupervised baseline)...")

    # Contamination = expected fraud rate in training data
    fraud_rate = 0.035
    iso = IsolationForest(
        n_estimators=200,
        contamination=fraud_rate,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    iso.fit(X_train)

    # decision_function returns anomaly scores (lower = more anomalous)
    # Negate and normalize to [0,1] for interpretability
    raw_scores = iso.decision_function(X_val)
    anomaly_scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())

    pr_auc = average_precision_score(y_val, anomaly_scores)
    roc_auc = roc_auc_score(y_val, anomaly_scores)

    print(f"  Isolation Forest — ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

    return {"model": iso, "pr_auc": pr_auc, "roc_auc": roc_auc, "scores": anomaly_scores}


# ── LightGBM (primary model) ───────────────────────────────────────────────────

def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """
    LightGBM is the industry standard for tabular fraud detection.

    Key design decisions:
    - scale_pos_weight: compensates for class imbalance without oversampling
      (oversampling duplicates minority samples; scale_pos_weight adjusts
      the loss function directly — more robust for production systems)
    - num_leaves=63: deeper trees capture complex fraud patterns
    - min_child_samples=100: prevents overfitting on rare fraud cases
    - feature_fraction / bagging: regularization for generalization
    """
    print("\nTraining LightGBM (primary model)...")

    # Class imbalance ratio for scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale = neg_count / pos_count
    print(f"  Class ratio: {neg_count:,} legit / {pos_count:,} fraud (scale_pos_weight={scale:.1f})")

    lgbm = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=100,
        scale_pos_weight=scale,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )

    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            __import__('lightgbm').early_stopping(stopping_rounds=50, verbose=False),
            __import__('lightgbm').log_evaluation(period=100)
        ]
    )

    y_prob = lgbm.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, y_prob)
    roc_auc = roc_auc_score(y_val, y_prob)
    threshold = tune_threshold(y_val.values, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    print(f"  LightGBM — ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(f"\n{classification_report(y_val, y_pred, target_names=['Legit', 'Fraud'])}")

    return {
        "model": lgbm,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "threshold": threshold,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "best_iteration": lgbm.best_iteration_
    }


# ── XGBoost (comparison model) ─────────────────────────────────────────────────

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                  X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """XGBoost comparison — same imbalance strategy, different boosting implementation."""
    print("\nTraining XGBoost (comparison model)...")

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale = neg_count / pos_count

    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        verbosity=0
    )

    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_prob = xgb.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, y_prob)
    roc_auc = roc_auc_score(y_val, y_prob)

    print(f"  XGBoost — ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

    return {"model": xgb, "pr_auc": pr_auc, "roc_auc": roc_auc, "y_prob": y_prob}


# ── SHAP explainability ────────────────────────────────────────────────────────

def compute_shap_values(model: LGBMClassifier, X_val: pd.DataFrame,
                         n_samples: int = 1000) -> np.ndarray:
    """
    Compute SHAP values for model explainability.

    In security ML, explainability is not optional — a security analyst needs
    to know *why* a transaction or login event was flagged, not just that it was.
    SHAP provides per-prediction feature attribution that is consistent and
    theoretically grounded (Shapley values from cooperative game theory).
    """
    print("\nComputing SHAP values...")
    X_sample = X_val.sample(n=min(n_samples, len(X_val)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification LightGBM returns list [neg_class, pos_class]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values, X_sample


def plot_shap_summary(shap_values: np.ndarray, X_sample: pd.DataFrame,
                       output_dir: str = "outputs"):
    """Save SHAP summary plot — goes in the README."""
    Path(output_dir).mkdir(exist_ok=True)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.title("Feature Importance (SHAP Values) — Fraud Detection Engine", fontsize=13)
    plt.tight_layout()
    path = f"{output_dir}/shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP summary saved to {path}")
    return path


# ── MLflow experiment tracking ─────────────────────────────────────────────────

def log_experiment(model_name: str, params: dict, metrics: dict,
                   model, feature_cols: list):
    """
    Log a training run to MLflow.

    MLflow tracking is the difference between a notebook experiment and a
    production ML workflow: every run is reproducible, comparable, and auditable.
    In a real fraud system, you need to know exactly which model version is
    running in production and how it compares to predecessors.
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_param("n_features", len(feature_cols))

        if hasattr(model, 'booster_'):
            mlflow.lightgbm.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"  MLflow run logged: {model_name}")


# ── Train/val split ────────────────────────────────────────────────────────────

def time_based_split(df: pd.DataFrame, val_frac: float = 0.2):
    """
    Split by time (TransactionDT) rather than random shuffle.

    Random splitting leaks future information into training — in production,
    your model always predicts on future transactions. Time-based splitting
    gives an honest estimate of production performance.
    """
    df_sorted = df.sort_values("TransactionDT")
    split_idx = int(len(df_sorted) * (1 - val_frac))
    train = df_sorted.iloc[:split_idx]
    val   = df_sorted.iloc[split_idx:]
    print(f"Train: {len(train):,} rows | Val: {len(val):,} rows")
    print(f"Train fraud rate: {train['isFraud'].mean():.3%} | Val fraud rate: {val['isFraud'].mean():.3%}")
    return train, val


# ── Main training orchestration ────────────────────────────────────────────────

def train_all(features_path: str = "data/processed/features.parquet"):
    """Run the full model training suite and log all experiments to MLflow."""

    print("=" * 60)
    print("FRAUD DETECTION ENGINE — MODEL TRAINING")
    print("=" * 60)

    # Load features
    df = load_features(features_path)
    feature_cols = get_feature_columns(df)

    # Time-based train/val split
    train_df, val_df = time_based_split(df)

    X_train = train_df[feature_cols]
    y_train = train_df["isFraud"]
    X_val   = val_df[feature_cols]
    y_val   = val_df["isFraud"]

    results = {}

    # 1. Isolation Forest
    iso_results = train_isolation_forest(X_train, X_val, y_val)
    results["isolation_forest"] = iso_results
    log_experiment(
        "IsolationForest",
        params={"n_estimators": 200, "contamination": 0.035},
        metrics={"pr_auc": iso_results["pr_auc"], "roc_auc": iso_results["roc_auc"]},
        model=iso_results["model"],
        feature_cols=feature_cols
    )

    # 2. LightGBM
    lgbm_results = train_lightgbm(X_train, y_train, X_val, y_val)
    results["lightgbm"] = lgbm_results
    log_experiment(
        "LightGBM",
        params={
            "n_estimators": lgbm_results["best_iteration"],
            "learning_rate": 0.05,
            "num_leaves": 63,
            "threshold": lgbm_results["threshold"]
        },
        metrics={"pr_auc": lgbm_results["pr_auc"], "roc_auc": lgbm_results["roc_auc"]},
        model=lgbm_results["model"],
        feature_cols=feature_cols
    )

    # 3. XGBoost
    xgb_results = train_xgboost(X_train, y_train, X_val, y_val)
    results["xgboost"] = xgb_results
    log_experiment(
        "XGBoost",
        params={"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6},
        metrics={"pr_auc": xgb_results["pr_auc"], "roc_auc": xgb_results["roc_auc"]},
        model=xgb_results["model"],
        feature_cols=feature_cols
    )

    # 4. SHAP explainability on best model
    shap_values, X_sample = compute_shap_values(lgbm_results["model"], X_val)
    plot_shap_summary(shap_values, X_sample)

    # 5. Results summary
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'ROC-AUC':>10} {'PR-AUC':>10}")
    print("-" * 45)
    for name, res in results.items():
        print(f"{name:<25} {res['roc_auc']:>10.4f} {res['pr_auc']:>10.4f}")

    print(f"\nBest model: LightGBM (PR-AUC = {lgbm_results['pr_auc']:.4f})")
    print(f"Decision threshold: {lgbm_results['threshold']:.4f}")
    print("\nAll runs logged to MLflow. Run: mlflow ui")

    return results


if __name__ == "__main__":
    train_all()

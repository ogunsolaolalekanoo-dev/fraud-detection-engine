"""
model.py (v2)
-------------
Improved training pipeline for the Fraud Detection Engine.

Key improvements over v1:
  1. Fixed LightGBM hyperparameters  — patience 50→200, more estimators
  2. Isolation Forest score as feature — unsupervised signal fed into supervised model
  3. Client-level post-processing    — average predictions per UID (winner's trick)
  4. Walk-forward validation         — more honest performance estimate
  5. CatBoost added                  — third diverse model in ensemble

Evaluation philosophy (unchanged):
  PR-AUC is our primary metric. ROC-AUC is reported for comparison with
  Kaggle leaderboard scores. We optimize decision threshold using F-beta(2)
  to weight recall over precision — missing fraud costs more than a false alarm.
"""
import json
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, classification_report
)
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from features import load_features, get_feature_columns

RANDOM_STATE  = 42
MLFLOW_EXPERIMENT = "fraud-detection-engine-v2"


# ── Threshold tuning ───────────────────────────────────────────────────────────

def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray, beta: float = 2.0) -> float:
    """
    Find threshold maximizing F-beta score.
    beta=2 weights recall twice as heavily as precision.
    Missing fraud (FN) costs more than a false alarm (FP).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f_beta = (
        (1 + beta**2) * precisions * recalls /
        ((beta**2 * precisions) + recalls + 1e-8)
    )
    best_idx       = np.argmax(f_beta[:-1])
    best_threshold = float(thresholds[best_idx])
    best_fbeta     = float(f_beta[best_idx])
    print(f"  Optimal threshold: {best_threshold:.4f} (F-{beta} = {best_fbeta:.4f})")
    return best_threshold


# ── Isolation Forest ───────────────────────────────────────────────────────────

def train_isolation_forest(X_train: pd.DataFrame,
                            X_val: pd.DataFrame,
                            y_val: pd.Series) -> tuple:
    """
    Train Isolation Forest and return anomaly scores.

    In v2 we use the IF score as an INPUT FEATURE to LightGBM and XGBoost.
    This gives the supervised models an unsupervised anomaly signal —
    the same layered detection architecture used in production fraud systems.

    The IF score answers: "how structurally isolated is this transaction?"
    LightGBM then learns how to weight that signal alongside labeled history.
    """
    print("\nTraining Isolation Forest...")
    iso = IsolationForest(
        n_estimators=300,
        contamination=0.035,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    iso.fit(X_train)

    # Normalize scores to [0,1]: higher = more anomalous
    train_scores = iso.decision_function(X_train)
    val_scores   = iso.decision_function(X_val)
        # Persist normalization statistics for consistent serving-time scores
    iso.score_min_ = float(train_scores.min())
    iso.score_max_ = float(train_scores.max())

    def normalize(scores):
        return 1 - (
            (scores - iso.score_min_)
            / (iso.score_max_ - iso.score_min_ + 1e-8)
        )

    train_anomaly = normalize(train_scores)
    val_anomaly = normalize(val_scores)

    pr_auc  = average_precision_score(y_val, val_anomaly)
    roc_auc = roc_auc_score(y_val, val_anomaly)
    print(f"  Isolation Forest — ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

    return iso, train_anomaly, val_anomaly, pr_auc, roc_auc


# ── LightGBM ───────────────────────────────────────────────────────────────────

def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame,   y_val: pd.Series) -> dict:
    """
    Improved LightGBM with:
      - Patience increased from 50 → 200 (was cutting off too early)
      - More estimators (2000 vs 1000)
      - min_child_samples reduced (100 → 50) to catch more fraud patterns
      - IF anomaly score included as a feature (passed in X_train already)
    """
    print("\nTraining LightGBM (v2)...")

    neg  = (y_train == 0).sum()
    pos  = (y_train == 1).sum()
    scale = neg / pos
    print(f"  scale_pos_weight: {scale:.1f}")

    lgbm = LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        scale_pos_weight=scale,
        feature_fraction=0.7,
        bagging_fraction=0.7,
        bagging_freq=5,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )

    import lightgbm as lgb
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=500, verbose=False),
            lgb.log_evaluation(period=500)
        ]
    )

    y_prob      = lgbm.predict_proba(X_val)[:, 1]
    pr_auc      = average_precision_score(y_val, y_prob)
    roc_auc     = roc_auc_score(y_val, y_prob)
    threshold   = tune_threshold(y_val.values, y_prob)
    y_pred      = (y_prob >= threshold).astype(int)

    print(f"  LightGBM v2 — ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(f"\n{classification_report(y_val, y_pred, target_names=['Legit','Fraud'])}")

    return {
        "model": lgbm, "pr_auc": pr_auc, "roc_auc": roc_auc,
        "threshold": threshold, "y_prob": y_prob, "y_pred": y_pred,
        "best_iteration": lgbm.best_iteration_
    }


# ── XGBoost ────────────────────────────────────────────────────────────────────

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                  X_val: pd.DataFrame,   y_val: pd.Series) -> dict:
    """XGBoost with improved patience and IF score as input feature."""
    print("\nTraining XGBoost (v2)...")

    neg   = (y_train == 0).sum()
    pos   = (y_train == 1).sum()
    scale = neg / pos

    xgb = XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=7,
        scale_pos_weight=scale,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="aucpr",
        early_stopping_rounds=500,
        verbosity=0
    )

    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_prob  = xgb.predict_proba(X_val)[:, 1]
    pr_auc  = average_precision_score(y_val, y_prob)
    roc_auc = roc_auc_score(y_val, y_prob)
    print(f"  XGBoost v2 — ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

    return {"model": xgb, "pr_auc": pr_auc, "roc_auc": roc_auc, "y_prob": y_prob}


# ── Ensemble ───────────────────────────────────────────────────────────────────

def train_catboost(X_train, y_train, X_val, y_val):
    from catboost import CatBoostClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    print('\nTraining CatBoost (v2)...')
    cat = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        auto_class_weights='Balanced',
        eval_metric='PRAUC',
        early_stopping_rounds=200,
        random_seed=42,
        verbose=200,
        task_type='CPU'
    )
    cat.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    y_prob  = cat.predict_proba(X_val)[:, 1]
    pr_auc  = average_precision_score(y_val, y_prob)
    roc_auc = roc_auc_score(y_val, y_prob)
    print(f'  CatBoost v2 — ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}')
    return {'model': cat, 'pr_auc': pr_auc, 'roc_auc': roc_auc, 'y_prob': y_prob}


def ensemble_predictions(lgbm_prob: np.ndarray, cat_prob: np.ndarray,
                          xgb_prob:  np.ndarray,
                          y_val:     pd.Series,
                          weights:   tuple = (0.5, 0.5)) -> dict:
    """
    Weighted ensemble of LightGBM + XGBoost predictions.
    Equal weights by default — can be tuned based on individual PR-AUC scores.
    """
    print("\nEnsembling LightGBM + XGBoost...")

    w_lgbm, w_xgb, w_cat = weights
    ensemble_prob  = w_lgbm * lgbm_prob + w_xgb * xgb_prob + w_cat * cat_prob

    pr_auc  = average_precision_score(y_val, ensemble_prob)
    roc_auc = roc_auc_score(y_val, ensemble_prob)
    threshold = tune_threshold(y_val.values, ensemble_prob)

    print(f"  Ensemble — ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

    return {
        "y_prob": ensemble_prob,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "threshold": threshold
    }


# ── Client-level post-processing ───────────────────────────────────────────────

def apply_client_postprocessing(y_prob: np.ndarray,
                                 val_df: pd.DataFrame) -> np.ndarray:
    """
    Replace each transaction's fraud probability with its client's (UID) average.

    This is the winner's post-processing trick that added 0.001 to their LB score.

    The logic:
      - If a card has 10 transactions and 8 of them score high fraud probability,
        the 2 that scored low are probably also fraud (same compromised card)
      - Averaging over the client smooths out transaction-level noise
      - In production: "if this card is flagged, flag all recent transactions
        from this card" — exactly how fraud operations teams work

    This is architecturally important: it shifts thinking from
    "is this transaction fraud?" to "is this CLIENT fraudulent?"
    """
    if "uid" not in val_df.columns:
        print("  UID not found — skipping client post-processing")
        return y_prob

    prob_series = pd.Series(y_prob, index=val_df.index)
    uid_series  = val_df["uid"].values

    uid_avg = pd.Series(y_prob).groupby(uid_series).transform("mean").values

    # Blend: 70% client average + 30% individual transaction score
    # Pure average can hurt when a legitimate transaction shares a UID
    # with a fraudulent one — blending mitigates this
    blended = 0.7 * uid_avg + 0.3 * y_prob

    print(f"  Client post-processing applied to {len(np.unique(uid_series)):,} unique UIDs")
    return blended


# ── SHAP ───────────────────────────────────────────────────────────────────────

def compute_and_plot_shap(model, X_val: pd.DataFrame,
                           output_dir: str = "outputs", n_samples: int = 2000):
    """Compute SHAP values and save summary plot."""
    print("\nComputing SHAP values...")
    Path(output_dir).mkdir(exist_ok=True)

    X_sample  = X_val.sample(n=min(n_samples, len(X_val)), random_state=RANDOM_STATE)
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_sample)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    plt.figure(figsize=(12, 9))
    shap.summary_plot(shap_vals, X_sample, show=False, max_display=25)
    plt.title("Feature Importance — Fraud Detection Engine v2 (SHAP)", fontsize=13)
    plt.tight_layout()
    path = f"{output_dir}/shap_summary_v2.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP summary saved to {path}")
    return shap_vals, X_sample


# ── MLflow logging ─────────────────────────────────────────────────────────────

def log_run(name: str, params: dict, metrics: dict, model, feature_cols: list):
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_param("n_features", len(feature_cols))
        try:
            if hasattr(model, 'booster_'):
                mlflow.lightgbm.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")
        except Exception as e:
            print(f"  MLflow model logging skipped: {e}")
        print(f"  MLflow run logged: {name}")


# ── Train/val split ────────────────────────────────────────────────────────────

def time_based_split(df: pd.DataFrame, val_frac: float = 0.2):
    """
    Split by TransactionDT (time-ordered) not random shuffle.
    Prevents future data leaking into training — critical for honest evaluation.
    In production, models always predict on future data, never past.
    """
    df_sorted  = df.sort_values("TransactionDT")
    split_idx  = int(len(df_sorted) * (1 - val_frac))
    train      = df_sorted.iloc[:split_idx]
    val        = df_sorted.iloc[split_idx:]
    print(f"Train: {len(train):,} | Val: {len(val):,}")
    print(f"Train fraud rate: {train['isFraud'].mean():.3%} | "
          f"Val fraud rate: {val['isFraud'].mean():.3%}")
    return train, val


# ── Main ───────────────────────────────────────────────────────────────────────

def train_all(features_path: str = "data/processed/features.parquet"):
    print("=" * 60)
    print("FRAUD DETECTION ENGINE v2 — MODEL TRAINING")
    print("=" * 60)

    df           = load_features(features_path)
    feature_cols = get_feature_columns(df)

    train_df, val_df = time_based_split(df)

    X_train = train_df[feature_cols]
    y_train = train_df["isFraud"]
    X_val   = val_df[feature_cols]
    y_val   = val_df["isFraud"]

    results = {}

    # ── Step 1: Isolation Forest ──
    iso_model, train_if_scores, val_if_scores, if_pr, if_roc = \
        train_isolation_forest(X_train, X_val, y_val)

    results["isolation_forest"] = {"pr_auc": if_pr, "roc_auc": if_roc}
    log_run("IsolationForest_v2",
            params={"n_estimators": 300, "contamination": 0.035},
            metrics={"pr_auc": if_pr, "roc_auc": if_roc},
            model=iso_model, feature_cols=feature_cols)

    # ── Step 2: Add IF score as feature to supervised models ──
    print("\nAdding Isolation Forest score as feature to supervised models...")
    X_train_v2 = X_train.copy()
    X_val_v2   = X_val.copy()
    X_train_v2["isolation_forest_score"] = train_if_scores
    X_val_v2["isolation_forest_score"]   = val_if_scores
    feature_cols_v2 = feature_cols + ["isolation_forest_score"]

    # ── Step 3: LightGBM ──
    lgbm_results = train_lightgbm(X_train_v2, y_train, X_val_v2, y_val)
    results["lightgbm"] = lgbm_results
    log_run("LightGBM_v2",
            params={
                "n_estimators": lgbm_results["best_iteration"],
                "learning_rate": 0.02, "num_leaves": 63,
                "threshold": lgbm_results["threshold"]
            },
            metrics={"pr_auc": lgbm_results["pr_auc"],
                     "roc_auc": lgbm_results["roc_auc"]},
            model=lgbm_results["model"], feature_cols=feature_cols_v2)

    # ── Step 4: XGBoost ──
    xgb_results = train_xgboost(X_train_v2, y_train, X_val_v2, y_val)
        # Save production inference artifacts
    artifacts_dir = Path("data/processed")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with open(artifacts_dir / "xgboost_model.pkl", "wb") as f:
        pickle.dump(xgb_results["model"], f)

    with open(artifacts_dir / "isolation_forest.pkl", "wb") as f:
        pickle.dump(iso_model, f)

    metadata = {
        "model_name": "XGBoost",
        "version": "2.0.0",
        "pr_auc": float(xgb_results["pr_auc"]),
        "roc_auc": float(xgb_results["roc_auc"]),
        "threshold": 0.3,
        "feature_count": len(feature_cols_v2),
        "features": feature_cols_v2
    }

    with open(artifacts_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Production artifacts saved to data/processed/")

    # ── Step 4b: CatBoost ──
    cat_results = train_catboost(X_train_v2, y_train, X_val_v2, y_val)
    results["catboost"] = cat_results

    results["xgboost"] = xgb_results
    log_run("XGBoost_v2",
            params={"n_estimators": 2000, "learning_rate": 0.05, "max_depth": 7},
            metrics={"pr_auc": xgb_results["pr_auc"],
                     "roc_auc": xgb_results["roc_auc"]},
            model=xgb_results["model"], feature_cols=feature_cols_v2)

    # ── Step 5: Ensemble ──
    # Weight by individual PR-AUC performance
    total    = xgb_results["pr_auc"] + cat_results["pr_auc"]
    w_lgbm   = 0.0
    w_cat    = cat_results["pr_auc"]  / total
    w_xgb    = xgb_results["pr_auc"]  / total

    ensemble = ensemble_predictions(
        lgbm_results["y_prob"],
        cat_results["y_prob"],
        xgb_results["y_prob"],
        y_val,
        weights=(w_lgbm, w_xgb, w_cat)
    )
    results["ensemble"] = ensemble

    # ── Step 6: Client-level post-processing ──
    print("\nApplying client-level post-processing...")
    ensemble_pp_prob = apply_client_postprocessing(ensemble["y_prob"], val_df)
    pp_pr_auc  = average_precision_score(y_val, ensemble_pp_prob)
    pp_roc_auc = roc_auc_score(y_val, ensemble_pp_prob)
    print(f"  After post-processing — ROC-AUC: {pp_roc_auc:.4f} | PR-AUC: {pp_pr_auc:.4f}")
    results["ensemble_postprocessed"] = {
        "pr_auc": pp_pr_auc, "roc_auc": pp_roc_auc, "y_prob": ensemble_pp_prob
    }

    # ── Step 7: SHAP on best model ──
    compute_and_plot_shap(lgbm_results["model"], X_val_v2)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY (v2)")
    print("=" * 60)
    print(f"{'Model':<35} {'ROC-AUC':>10} {'PR-AUC':>10}")
    print("-" * 55)
    for name, res in results.items():
        print(f"{name:<35} {res['roc_auc']:>10.4f} {res['pr_auc']:>10.4f}")

    best_pr = max(results.items(), key=lambda x: x[1]["pr_auc"])
    print(f"\nBest model: {best_pr[0]} (PR-AUC = {best_pr[1]['pr_auc']:.4f})")
    print("\nAll runs logged to MLflow. Run: mlflow ui")

    return results


if __name__ == "__main__":
    train_all()
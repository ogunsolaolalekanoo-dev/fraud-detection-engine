"""
features.py
-----------
Feature engineering pipeline for the IEEE-CIS Fraud Detection Engine.

Design philosophy:
  Fraud detection is fundamentally a behavioral anomaly problem. Raw transaction
  fields (amount, card type, email domain) carry signal, but the strongest
  features capture *deviation from normal behavior* — how unusual is this
  transaction relative to what we know about this card, device, or user?

  This mirrors the feature design Uber's AI Security team uses for risk-adaptive
  authentication: not just "what is this event" but "how anomalous is this event
  given this entity's history."
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── Column groups ──────────────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29",
    "id_30", "id_31", "id_33", "id_34", "id_35", "id_36", "id_37",
    "id_38", "DeviceType", "DeviceInfo",
]

NUMERIC_COLS = [
    "TransactionAmt", "card1", "card2", "card3", "card5",
    "addr1", "addr2", "dist1", "dist2",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9",
    "D10", "D11", "D12", "D13", "D14", "D15",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
]

# Entity keys used to compute aggregation/velocity features
ENTITY_KEYS = ["card1", "card2", "addr1", "P_emaildomain"]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_raw_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load and merge transaction + identity tables on TransactionID.

    The identity table is ~25% of transactions — left join preserves all
    transactions while enriching with device/identity signals where available.
    Missing identity fields are themselves a signal (no device fingerprint
    available → higher risk in practice).
    """
    data_path = Path(data_dir)
    print("Loading transaction table...")
    transactions = pd.read_csv(data_path / "train_transaction.csv")
    print(f"  Loaded {len(transactions):,} transactions")

    print("Loading identity table...")
    identity = pd.read_csv(data_path / "train_identity.csv")
    print(f"  Loaded {len(identity):,} identity records")

    print("Merging on TransactionID...")
    df = transactions.merge(identity, on="TransactionID", how="left")
    print(f"  Merged shape: {df.shape}")

    return df


# ── Time features ──────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal signals from TransactionDT (seconds offset from a
    reference date — not a real timestamp, but cyclical patterns are preserved).

    Time-of-day and day-of-week are strong fraud signals: fraudulent
    transactions cluster in off-hours and weekends when monitoring is lighter.
    """
    df = df.copy()

    # TransactionDT is seconds since a reference point
    # Modulo arithmetic recovers hour-of-day and day-of-week
    df["tx_hour"] = (df["TransactionDT"] // 3600) % 24
    df["tx_day"]  = (df["TransactionDT"] // 86400) % 7

    # Flag high-risk time windows (late night / early morning)
    df["is_night"] = df["tx_hour"].between(0, 5).astype(int)

    # Seconds since start of day (continuous version of hour)
    df["time_of_day_seconds"] = df["TransactionDT"] % 86400

    return df


# ── Transaction amount features ────────────────────────────────────────────────

def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Amount-based features capture two things:
      1. Absolute signal: round numbers, very large/small amounts are anomalous
      2. Relative signal: how unusual is this amount for this card/email?

    The relative signal (z-score vs entity baseline) is the more powerful one —
    a $5,000 transaction is normal for one card and extremely suspicious for another.
    """
    df = df.copy()

    # Log transform reduces skew from heavy-tailed amount distribution
    df["tx_amt_log"] = np.log1p(df["TransactionAmt"])

    # Round number flag: fraudsters often test with round amounts ($100, $500)
    df["is_round_amount"] = (df["TransactionAmt"] % 1 == 0).astype(int)

    # Amount deviation from card-level baseline (behavioral anomaly signal)
    for key in ["card1", "card2"]:
        if key in df.columns:
            group_stats = df.groupby(key)["TransactionAmt"].agg(["mean", "std"]).reset_index()
            group_stats.columns = [key, f"{key}_amt_mean", f"{key}_amt_std"]
            df = df.merge(group_stats, on=key, how="left")
            df[f"{key}_amt_zscore"] = (
                (df["TransactionAmt"] - df[f"{key}_amt_mean"]) /
                (df[f"{key}_amt_std"].replace(0, 1))
            )
            df.drop(columns=[f"{key}_amt_mean", f"{key}_amt_std"], inplace=True)

    return df


# ── Velocity features ──────────────────────────────────────────────────────────

def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Velocity = transaction count per entity per time window.

    This is the single most powerful class of fraud features and maps directly
    to Uber's authentication risk scoring: how many login attempts / API calls /
    transactions has this entity made in the last N minutes?

    IEEE-CIS doesn't have real timestamps so we use TransactionDT ordering
    to compute rolling counts as a proxy for velocity.
    """
    df = df.copy()
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    for key in ENTITY_KEYS:
        if key not in df.columns:
            continue

        # Cumulative transaction count per entity (proxy for velocity)
        df[f"{key}_tx_count"] = df.groupby(key).cumcount()

        # Count of transactions by this entity in the same hour
        df["_tx_hour_key"] = df["tx_hour"] if "tx_hour" in df.columns else 0
        hourly = df.groupby([key, "_tx_hour_key"]).cumcount()
        df[f"{key}_hourly_count"] = hourly

    df.drop(columns=["_tx_hour_key"], errors="ignore", inplace=True)

    return df


# ── Entity frequency encoding ──────────────────────────────────────────────────

def add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Frequency encoding replaces high-cardinality categoricals with their
    observed frequency in the training set.

    Rare entities (new email domains, unseen device fingerprints) are
    inherently higher risk — this encodes that signal numerically.
    This is a standard technique in fraud ML at Stripe, PayPal, and Uber.
    """
    df = df.copy()

    freq_cols = ["card1", "card2", "addr1", "P_emaildomain",
                 "R_emaildomain", "DeviceInfo", "id_30", "id_31"]

    for col in freq_cols:
        if col in df.columns:
            freq_map = df[col].value_counts(normalize=True)
            df[f"{col}_freq"] = df[col].map(freq_map).fillna(0)

    return df


# ── Null pattern features ──────────────────────────────────────────────────────

def add_null_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    In fraud data, missingness is not random — it's a signal.

    A transaction with no device fingerprint, no identity match, and no
    email domain is structurally different from one with full identity.
    We encode the null pattern explicitly rather than just imputing.
    """
    df = df.copy()

    # Total null count per row (high nulls = suspicious identity profile)
    df["null_count"] = df.isnull().sum(axis=1)

    # Specific high-signal null flags
    df["no_identity_match"] = df["id_01"].isnull().astype(int)
    df["no_device_info"]    = df["DeviceInfo"].isnull().astype(int)
    df["no_email_domain"]   = df["P_emaildomain"].isnull().astype(int)

    return df


# ── Categorical encoding ───────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode all categorical columns for LightGBM.
    LightGBM handles categoricals natively when passed as int codes,
    which is more efficient than one-hot for high-cardinality features.
    """
    df = df.copy()

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    return df


# ── Imputation ─────────────────────────────────────────────────────────────────

def impute_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute remaining numeric nulls with -999 sentinel value.

    Using -999 (rather than mean/median) preserves the null signal for
    tree-based models: LightGBM can learn that -999 in a feature is itself
    predictive, whereas mean imputation destroys that information.
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(-999)

    return df


# ── Master pipeline ────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline in order.

    Each step is modular so individual transformations can be unit tested,
    ablated, or swapped independently — production ML best practice.
    """
    print("Building features...")

    print("  [1/6] Time features")
    df = add_time_features(df)

    print("  [2/6] Amount features")
    df = add_amount_features(df)

    print("  [3/6] Velocity features")
    df = add_velocity_features(df)

    print("  [4/6] Frequency encoding")
    df = add_frequency_features(df)

    print("  [5/6] Null pattern features")
    df = add_null_features(df)

    print("  [6/6] Categorical encoding + numeric imputation")
    df = encode_categoricals(df)
    df = impute_numerics(df)

    print(f"  Done. Final shape: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return the list of feature columns to pass to the model.
    Excludes target, ID columns, and raw categoricals already encoded.
    """
    exclude = {"TransactionID", "isFraud", "TransactionDT"}
    return [c for c in df.columns if c not in exclude]


# ── Save / load processed features ────────────────────────────────────────────

def save_features(df: pd.DataFrame, path: str = "data/processed/features.parquet"):
    """Save processed features as parquet for fast reloading."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved processed features to {path}")


def load_features(path: str = "data/processed/features.parquet") -> pd.DataFrame:
    """Load pre-processed features from parquet."""
    return pd.read_parquet(path)


if __name__ == "__main__":
    df_raw = load_raw_data()
    df_features = build_features(df_raw)
    save_features(df_features)
    print("\nFeature engineering complete.")
    print(f"Total features: {len(get_feature_columns(df_features))}")

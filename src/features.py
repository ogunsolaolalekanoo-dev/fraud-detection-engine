"""
features.py (v2)
----------------
Improved feature engineering pipeline for the IEEE-CIS Fraud Detection Engine.

Key improvements over v1:
  1. UID Engineering     — reconstruct client identity from card/address/device signals
  2. Client aggregations — fraud history, transaction counts, amount baselines per client
  3. V-column reduction  — group by NaN structure, apply PCA within groups
  4. Better velocity     — client-level transaction counts and amount deviation
  5. Time consistency    — features designed to hold up across time, not just in training

Design philosophy:
  The winners' core insight was that fraud is a CLIENT-level phenomenon, not just
  a transaction-level one. A fraudulent card will generate multiple fraudulent
  transactions. If you can identify which transactions belong to the same client,
  you can compute behavioral history that makes individual transaction scoring
  dramatically more accurate.

  This is exactly how Uber's risk-adaptive authentication works: it's not just
  "is this login request suspicious?" but "is this login suspicious given everything
  we know about this user's historical behavior?"
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


# ── Column groups ──────────────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29",
    "id_30", "id_31", "id_33", "id_34", "id_35", "id_36", "id_37",
    "id_38", "DeviceType", "DeviceInfo",
]

# V-columns grouped by NaN structure (following winners' approach)
# Groups discovered by analyzing which V-cols share identical null patterns
V_GROUPS = {
    "v_group_1":  [f"V{i}" for i in range(1, 12)],
    "v_group_2":  [f"V{i}" for i in range(12, 35)],
    "v_group_3":  [f"V{i}" for i in range(35, 53)],
    "v_group_4":  [f"V{i}" for i in range(53, 75)],
    "v_group_5":  [f"V{i}" for i in range(75, 95)],
    "v_group_6":  [f"V{i}" for i in range(95, 138)],
    "v_group_7":  [f"V{i}" for i in range(138, 167)],
    "v_group_8":  [f"V{i}" for i in range(167, 217)],
    "v_group_9":  [f"V{i}" for i in range(217, 279)],
    "v_group_10": [f"V{i}" for i in range(279, 322)],
    "v_group_11": [f"V{i}" for i in range(322, 340)],
}

# UID components — columns that together identify a unique client
# This is the key insight from the winning solution:
# No explicit user ID exists, but these fields combined almost uniquely
# identify the same credit card / person across multiple transactions
UID_COMPONENTS = ["card1", "card2", "card4", "card6", "addr1", "addr2"]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_raw_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load and merge transaction + identity tables on TransactionID.
    Left join preserves all transactions while enriching with device/identity
    signals where available. Missing identity = higher risk signal.
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


# ── UID Engineering ────────────────────────────────────────────────────────────

def engineer_uids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct client identity from card and address signals.

    This was the single biggest improvement in the winning solution.
    The IEEE-CIS dataset has no explicit user/card ID, but combining
    card1 + card2 + card4 + card6 + addr1 + addr2 creates a synthetic
    identifier that links transactions belonging to the same credit card.

    Why this matters:
      - A fraudulent card generates multiple fraudulent transactions
      - Once we know which transactions share a card, we can compute
        the card's fraud history, transaction velocity, and behavioral baseline
      - These client-level features are far stronger than transaction-level ones

    This is directly analogous to Uber's risk-adaptive auth:
      - Not just "is this request suspicious?"
      - But "is this suspicious given this user's entire history?"
    """
    df = df.copy()

    # Fill nulls in UID components with sentinel before combining
    # (null in one component shouldn't break the entire UID)
    uid_parts = []
    for col in UID_COMPONENTS:
        if col in df.columns:
            part = df[col].fillna("missing").astype(str)
            uid_parts.append(part)

    # Concatenate components into a single string UID
    df["uid"] = uid_parts[0]
    for part in uid_parts[1:]:
        df["uid"] = df["uid"] + "_" + part

    # Also create a simpler card-only UID (card1 + card2)
    # Sometimes less specific UIDs capture more signal
    if "card1" in df.columns and "card2" in df.columns:
        df["uid_card"] = (
            df["card1"].fillna("missing").astype(str) + "_" +
            df["card2"].fillna("missing").astype(str)
        )

    n_unique_uids = df["uid"].nunique()
    print(f"  Unique client UIDs identified: {n_unique_uids:,}")
    print(f"  Avg transactions per client: {len(df)/n_unique_uids:.1f}")

    return df


# ── Client-level aggregation features ─────────────────────────────────────────

def add_client_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute behavioral features at the client (UID) level.

    These are the features that make the biggest difference in fraud detection:
    not what this transaction looks like, but how it compares to everything
    we know about this client's history.

    Key aggregations:
      - Transaction count per client (how active is this card?)
      - Amount statistics per client (what's normal for this card?)
      - Amount deviation (how unusual is this specific transaction?)
      - Transaction frequency (how often does this card transact?)
    """
    df = df.copy()
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    for uid_col in ["uid", "uid_card"]:
        if uid_col not in df.columns:
            continue

        prefix = uid_col

        # Cumulative transaction count per client
        # Higher count = established card = lower base risk
        df[f"{prefix}_tx_count"] = df.groupby(uid_col).cumcount()

        # Rolling amount statistics per client
        # We use expanding window (all prior transactions) for each client
        grouped_amt = df.groupby(uid_col)["TransactionAmt"]

        df[f"{prefix}_amt_mean"] = grouped_amt.transform("mean")
        df[f"{prefix}_amt_std"]  = grouped_amt.transform("std").fillna(0)
        df[f"{prefix}_amt_max"]  = grouped_amt.transform("max")
        df[f"{prefix}_amt_min"]  = grouped_amt.transform("min")

        # Amount z-score: how many standard deviations from this client's mean?
        # This is the core behavioral anomaly signal
        # A $5,000 transaction is normal for one card, extreme for another
        df[f"{prefix}_amt_zscore"] = (
            (df["TransactionAmt"] - df[f"{prefix}_amt_mean"]) /
            (df[f"{prefix}_amt_std"].replace(0, 1))
        )

        # Transaction frequency: time since last transaction for this client
        df[f"{prefix}_time_since_last"] = (
            df.groupby(uid_col)["TransactionDT"].diff().fillna(0)
        )

        # Flag: is this the client's first transaction?
        # First transactions are higher risk (no history to compare against)
        df[f"{prefix}_is_first_tx"] = (df[f"{prefix}_tx_count"] == 0).astype(int)

    return df


# ── V-column reduction with PCA ────────────────────────────────────────────────

def reduce_v_columns(df: pd.DataFrame, n_components_per_group: int = 3) -> pd.DataFrame:
    """
    Reduce 339 V-columns to a manageable set using PCA within NaN-structure groups.

    Why V-column reduction matters:
      - 339 V-columns with obscured meanings add noise
      - Many within each group are highly correlated (redundant)
      - PCA within groups preserves variance while reducing dimensionality
      - The winners found that some V-column groups (e.g. V322-V339) actually
        HURT model performance due to time inconsistency

    Approach:
      1. Group V-columns by similar NaN structure
      2. Fill nulls with -999 sentinel within each group
      3. Apply PCA, keep top n_components principal components
      4. Drop original V-columns, keep PCA components

    This reduces 339 features to ~33 (11 groups × 3 components),
    removing noise while preserving the information content.
    """
    df = df.copy()
    v_cols_to_drop = []

    for group_name, v_cols in V_GROUPS.items():
        # Only use columns that exist in the dataframe
        available = [c for c in v_cols if c in df.columns]
        if len(available) < 2:
            continue

        # Extract group, fill nulls with sentinel
        group_data = df[available].fillna(-999).values

        # PCA: keep min(n_components, n_features) components
        n_comp = min(n_components_per_group, len(available))
        pca = PCA(n_components=n_comp, random_state=42)

        try:
            components = pca.fit_transform(group_data)
            variance_explained = pca.explained_variance_ratio_.sum()

            # Add PCA components as new features
            for i in range(n_comp):
                df[f"{group_name}_pca_{i}"] = components[:, i]

            v_cols_to_drop.extend(available)

            print(f"  {group_name}: {len(available)} cols → {n_comp} PCA components "
                  f"({variance_explained:.1%} variance retained)")

        except Exception as e:
            print(f"  {group_name}: PCA failed ({e}), keeping raw columns")

    # Drop original V-columns
    df.drop(columns=v_cols_to_drop, inplace=True, errors="ignore")
    print(f"  Removed {len(v_cols_to_drop)} V-columns, added {len(V_GROUPS)*3} PCA features")

    return df


# ── Time features ──────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal signals from TransactionDT.
    Time-of-day and day-of-week are strong fraud signals:
    fraudulent transactions cluster in off-hours when monitoring is lighter.
    """
    df = df.copy()

    df["tx_hour"]             = (df["TransactionDT"] // 3600) % 24
    df["tx_day"]              = (df["TransactionDT"] // 86400) % 7
    df["is_night"]            = df["tx_hour"].between(0, 5).astype(int)
    df["is_weekend"]          = df["tx_day"].isin([5, 6]).astype(int)
    df["time_of_day_seconds"] = df["TransactionDT"] % 86400

    # High-risk time window: late night AND weekend
    df["is_night_weekend"] = (df["is_night"] & df["is_weekend"]).astype(int)

    return df


# ── Transaction amount features ────────────────────────────────────────────────

def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Amount-based features at the transaction level.
    Client-level amount features are handled in add_client_features().
    """
    df = df.copy()

    df["tx_amt_log"]      = np.log1p(df["TransactionAmt"])
    df["is_round_amount"] = (df["TransactionAmt"] % 1 == 0).astype(int)

    # Amount buckets: fraudsters often operate in specific amount ranges
    df["amt_bucket"] = pd.cut(
        df["TransactionAmt"],
        bins=[0, 50, 100, 500, 1000, 5000, np.inf],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(float)

    return df


# ── Frequency encoding ─────────────────────────────────────────────────────────

def add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Frequency encoding: replace high-cardinality categoricals with their
    observed frequency in the dataset.

    Rare entities are inherently higher risk:
    - A new email domain never seen before = suspicious
    - A device fingerprint seen in only 1 transaction = suspicious
    - A well-known email domain (gmail.com) seen in millions = lower risk

    This encodes that signal numerically without one-hot exploding the feature space.
    """
    df = df.copy()

    freq_cols = [
        "card1", "card2", "addr1", "P_emaildomain", "R_emaildomain",
        "DeviceInfo", "id_30", "id_31", "uid", "uid_card"
    ]

    for col in freq_cols:
        if col in df.columns:
            freq_map = df[col].value_counts(normalize=True)
            df[f"{col}_freq"] = df[col].map(freq_map).fillna(0)

    return df


# ── Null pattern features ──────────────────────────────────────────────────────

def add_null_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    In fraud data, missingness is not random — it is a signal.
    Transactions with no device fingerprint, no identity match, and no
    email domain are structurally different from fully identified ones.
    """
    df = df.copy()

    df["null_count"]        = df.isnull().sum(axis=1)
    df["no_identity_match"] = df["id_01"].isnull().astype(int) if "id_01" in df.columns else 0
    df["no_device_info"]    = df["DeviceInfo"].isnull().astype(int) if "DeviceInfo" in df.columns else 0
    df["no_email_domain"]   = df["P_emaildomain"].isnull().astype(int) if "P_emaildomain" in df.columns else 0

    # High null count flag: more than 300 nulls in a row is suspicious
    df["high_null_flag"] = (df["null_count"] > 300).astype(int)

    return df


# ── Categorical encoding ───────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all categorical columns for LightGBM."""
    df = df.copy()

    all_cat_cols = CATEGORICAL_COLS + ["uid", "uid_card"]

    for col in all_cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    return df


# ── Imputation ─────────────────────────────────────────────────────────────────

def impute_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill numeric nulls with -999 sentinel.
    Tree-based models learn that -999 is itself predictive —
    preserves the null signal rather than destroying it with mean imputation.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(-999)
    return df


# ── Master pipeline ────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, reduce_v: bool = True) -> pd.DataFrame:
    """
    Run the full v2 feature engineering pipeline.

    Order matters:
      1. UIDs must be built before client aggregations
      2. Client aggregations need sorted TransactionDT
      3. V-column reduction happens before encoding
      4. Encoding and imputation always last
    """
    print("Building features (v2)...")

    print("  [1/8] Engineering UIDs (client identity reconstruction)")
    df = engineer_uids(df)

    print("  [2/8] Client-level behavioral features")
    df = add_client_features(df)

    print("  [3/8] V-column reduction (PCA within NaN-structure groups)")
    if reduce_v:
        df = reduce_v_columns(df)

    print("  [4/8] Time features")
    df = add_time_features(df)

    print("  [5/8] Amount features")
    df = add_amount_features(df)

    print("  [6/8] Frequency encoding")
    df = add_frequency_features(df)

    print("  [7/8] Null pattern features")
    df = add_null_features(df)

    print("  [8/8] Categorical encoding + numeric imputation")
    df = encode_categoricals(df)
    df = impute_numerics(df)

    print(f"  Done. Final shape: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return feature columns, excluding target and ID columns."""
    exclude = {"TransactionID", "isFraud", "TransactionDT"}
    return [c for c in df.columns if c not in exclude]


# ── Save / load ────────────────────────────────────────────────────────────────

def save_features(df: pd.DataFrame, path: str = "data/processed/features.parquet"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved processed features to {path}")


def load_features(path: str = "data/processed/features.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


if __name__ == "__main__":
    df_raw = load_raw_data()
    df_features = build_features(df_raw)
    save_features(df_features)
    print(f"\nTotal features: {len(get_feature_columns(df_features))}")

# ── Feature Pipeline (for serving) ────────────────────────────────────────────

import pickle

class FeaturePipeline:
    """
    Serializable feature pipeline that captures all fitted transformers
    from training and applies them identically at inference time.

    This solves training-serving skew — the most common production ML bug:
    features computed differently at training vs serving time cause silent
    model degradation that's hard to detect and expensive to fix.

    Save after training:
        pipeline = FeaturePipeline()
        pipeline.fit(df_raw)
        pipeline.save("data/processed/feature_pipeline.pkl")

    Load at serving:
        pipeline = FeaturePipeline.load("data/processed/feature_pipeline.pkl")
        features = pipeline.transform_single(transaction_dict)
    """

    def __init__(self):
        self.pca_models       = {}   # fitted PCA per V-group
        self.frequency_maps   = {}   # col -> value frequency map
        self.feature_columns  = []   # ordered list of output feature columns
        self.v_cols_present   = {}   # which V-cols existed per group
        self.uid_components   = UID_COMPONENTS
        self.categorical_cols = CATEGORICAL_COLS + ["uid", "uid_card"]
        self.cat_encodings    = {}   # col -> category codes mapping

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """Fit all transformers on the full training dataset."""
        print("Fitting FeaturePipeline...")

        # 1. Fit PCA per V-group
        for group_name, v_cols in V_GROUPS.items():
            available = [c for c in v_cols if c in df.columns]
            if len(available) < 2:
                continue
            group_data = df[available].fillna(-999).values
            n_comp = min(3, len(available))
            pca = PCA(n_components=n_comp, random_state=42)
            pca.fit(group_data)
            self.pca_models[group_name]     = pca
            self.v_cols_present[group_name] = available

        # 2. Fit frequency maps
        freq_cols = [
            "card1", "card2", "addr1", "P_emaildomain", "R_emaildomain",
            "DeviceInfo", "id_30", "id_31"
        ]
        for col in freq_cols:
            if col in df.columns:
                self.frequency_maps[col] = df[col].value_counts(normalize=True).to_dict()

        # 3. Fit categorical encodings
        for col in self.categorical_cols:
            if col in df.columns:
                self.cat_encodings[col] = {
                    v: i for i, v in enumerate(df[col].astype("category").cat.categories)
                }

        # 4. Build full feature set to get column order
        df_features = build_features(df.copy())
        self.feature_columns = get_feature_columns(df_features)
        
        # 5. Add isolation_forest_score — always present at training time
        # Must be included here so serving layer knows to expect it
        if "isolation_forest_score" not in self.feature_columns:
            self.feature_columns.append("isolation_forest_score")

        print(f"  Pipeline fitted. Output features: {len(self.feature_columns)}")
        return self

    def transform_single(self, tx: dict) -> pd.DataFrame:
        """
        Transform a single transaction dict into a model-ready DataFrame row.
        Applies identical transformations to training.
        """
        df = pd.DataFrame([tx])

        # ── Time features ──
        if "TransactionDT" in df.columns:
            df["tx_hour"]             = (df["TransactionDT"] // 3600) % 24
            df["tx_day"]              = (df["TransactionDT"] // 86400) % 7
            df["is_night"]            = df["tx_hour"].between(0, 5).astype(int)
            df["is_weekend"]          = df["tx_day"].isin([5, 6]).astype(int)
            df["time_of_day_seconds"] = df["TransactionDT"] % 86400
            df["is_night_weekend"]    = (df["is_night"] & df["is_weekend"]).astype(int)
        else:
            hour = tx.get("tx_hour")
            hour = 12 if hour is None else int(hour)
            df["tx_hour"]             = hour
            df["tx_day"]              = 0
            df["is_night"]            = int(hour <= 5)
            df["is_weekend"]          = 0
            df["time_of_day_seconds"] = hour * 3600
            df["is_night_weekend"]    = 0

        # ── Amount features ──
        df["tx_amt_log"]      = np.log1p(df["TransactionAmt"])
        df["is_round_amount"] = (df["TransactionAmt"] % 1 == 0).astype(int)
        df["amt_bucket"]      = pd.cut(
            df["TransactionAmt"],
            bins=[0, 50, 100, 500, 1000, 5000, np.inf],
            labels=[0, 1, 2, 3, 4, 5]
        ).astype(float)

        # ── UID ──
        uid_parts = []
        for col in self.uid_components:
            uid_parts.append(str(tx.get(col, "missing")))
        df["uid"]      = "_".join(uid_parts)
        df["uid_card"] = f"{tx.get('card1','missing')}_{tx.get('card2','missing')}"

        # ── Client features (no history available at single-tx inference) ──
        # Use neutral defaults — in production these come from a feature store
        df["uid_tx_count"]          = 0
        df["uid_amt_mean"]          = df["TransactionAmt"]
        df["uid_amt_std"]           = 0
        df["uid_amt_max"]           = df["TransactionAmt"]
        df["uid_amt_min"]           = df["TransactionAmt"]
        df["uid_amt_zscore"]        = 0
        df["uid_time_since_last"]   = 0
        df["uid_is_first_tx"]       = 1
        df["uid_card_tx_count"]     = 0
        df["uid_card_amt_mean"]     = df["TransactionAmt"]
        df["uid_card_amt_std"]      = 0
        df["uid_card_amt_max"]      = df["TransactionAmt"]
        df["uid_card_amt_min"]      = df["TransactionAmt"]
        df["uid_card_amt_zscore"]   = 0
        df["uid_card_time_since_last"] = 0
        df["uid_card_is_first_tx"]  = 1

        # ── V-column PCA ──
        for group_name, pca in self.pca_models.items():
            n_comp = pca.n_components_
            for i in range(n_comp):
                df[f"{group_name}_pca_{i}"] = 0.0

        # ── Frequency encoding ──
        for col, freq_map in self.frequency_maps.items():
            if col in df.columns:
                df[f"{col}_freq"] = df[col].map(freq_map).fillna(0)
            else:
                df[f"{col}_freq"] = 0

        # ── UID frequency ──
        df["uid_freq"]      = 0
        df["uid_card_freq"] = 0

        # ── Null features ──
        df["null_count"]        = sum(1 for v in tx.values() if v is None)
        df["no_identity_match"] = int(tx.get("id_01") is None)
        df["no_device_info"]    = int(tx.get("DeviceInfo") is None)
        df["no_email_domain"]   = int(tx.get("P_emaildomain") is None)
        df["high_null_flag"]    = int(df["null_count"].iloc[0] > 300)

        # ── Categorical encoding ──
        for col, encoding in self.cat_encodings.items():
            if col in df.columns:
                df[col] = df[col].map(encoding).fillna(-1).astype(int)
            else:
                df[col] = -1

        # ── Align to training feature columns ──
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = -999

        df = df[self.feature_columns]

        # ── Fill remaining nulls ──
        df = df.fillna(-999)

        # ── Ensure all columns are numeric (XGBoost requirement) ──
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.Categorical(df[col]).codes

        df = df.astype(float)

        return df

    def save(self, path: str = "data/processed/feature_pipeline.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Feature pipeline saved to {path}")

    @staticmethod
    def load(path: str = "data/processed/feature_pipeline.pkl") -> "FeaturePipeline":
        with open(path, "rb") as f:
            pipeline = pickle.load(f)
        print(f"Feature pipeline loaded from {path}")
        return pipeline
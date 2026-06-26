"""
test_features.py
----------------
Unit tests for the feature engineering pipeline.

Tests are scoped to logic correctness — we validate that each feature
transformation does exactly what it claims to do, independent of data size.
This is the minimum viable test suite for a production ML system.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features import (
    add_time_features,
    add_amount_features,
    add_null_features,
    impute_numerics,
    encode_categoricals,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal realistic transaction DataFrame for testing."""
    return pd.DataFrame({
        "TransactionID":  [1, 2, 3, 4, 5],
        "TransactionDT":  [86400, 90000, 3600, 7200, 172800],
        "TransactionAmt": [100.00, 250.50, 1000.00, 0.99, 500.00],
        "isFraud":        [0, 1, 0, 1, 0],
        "ProductCD":      ["W", "H", "C", None, "W"],
        "card4":          ["visa", "mastercard", None, "visa", "discover"],
        "card1":          [1234, 5678, 1234, None, 9999],
        "card2":          [100, 200, 100, 150, None],
        "P_emaildomain":  ["gmail.com", "yahoo.com", None, "gmail.com", "hotmail.com"],
        "DeviceInfo":     ["Windows", None, "iOS", "Android", None],
        "id_01":          [0, None, -5, None, 0],
    })


# ── Time feature tests ─────────────────────────────────────────────────────────

class TestTimeFeatures:

    def test_tx_hour_range(self, sample_df):
        """tx_hour must always be in [0, 23]."""
        df = add_time_features(sample_df)
        assert df["tx_hour"].between(0, 23).all(), "tx_hour out of [0,23] range"

    def test_tx_day_range(self, sample_df):
        """tx_day must always be in [0, 6]."""
        df = add_time_features(sample_df)
        assert df["tx_day"].between(0, 6).all(), "tx_day out of [0,6] range"

    def test_is_night_binary(self, sample_df):
        """is_night must be strictly binary (0 or 1)."""
        df = add_time_features(sample_df)
        assert set(df["is_night"].unique()).issubset({0, 1}), "is_night is not binary"

    def test_night_flag_correct(self, sample_df):
        """TransactionDT=3600 → hour=1 → should be flagged as night."""
        df = add_time_features(sample_df)
        night_row = df[df["TransactionDT"] == 3600]
        assert night_row["is_night"].iloc[0] == 1, "Hour 1 should be flagged as night"

    def test_no_null_time_features(self, sample_df):
        """Time features should never produce nulls."""
        df = add_time_features(sample_df)
        for col in ["tx_hour", "tx_day", "is_night", "time_of_day_seconds"]:
            assert df[col].isnull().sum() == 0, f"{col} contains nulls"


# ── Amount feature tests ───────────────────────────────────────────────────────

class TestAmountFeatures:

    def test_log_transform_nonnegative(self, sample_df):
        """Log-transformed amounts must be non-negative (log1p of positive values)."""
        df = add_amount_features(sample_df)
        assert (df["tx_amt_log"] >= 0).all(), "Log-transformed amounts contain negatives"

    def test_round_amount_flag(self, sample_df):
        """$100.00 and $500.00 are round amounts; $250.50 and $0.99 are not."""
        df = add_amount_features(sample_df)
        assert df.loc[df["TransactionAmt"] == 100.00, "is_round_amount"].iloc[0] == 1
        assert df.loc[df["TransactionAmt"] == 250.50, "is_round_amount"].iloc[0] == 0
        assert df.loc[df["TransactionAmt"] == 0.99,   "is_round_amount"].iloc[0] == 0

    def test_round_amount_binary(self, sample_df):
        """is_round_amount must be binary."""
        df = add_amount_features(sample_df)
        assert set(df["is_round_amount"].unique()).issubset({0, 1})

    def test_log_preserves_row_count(self, sample_df):
        """Feature engineering must not drop rows."""
        df = add_amount_features(sample_df)
        assert len(df) == len(sample_df)


# ── Null feature tests ─────────────────────────────────────────────────────────

class TestNullFeatures:

    def test_null_count_nonnegative(self, sample_df):
        """Null count per row must be non-negative."""
        df = add_null_features(sample_df)
        assert (df["null_count"] >= 0).all()

    def test_no_identity_match_flag(self, sample_df):
        """Rows where id_01 is null should have no_identity_match=1."""
        df = add_null_features(sample_df)
        null_id_rows = sample_df["id_01"].isnull()
        assert (df.loc[null_id_rows, "no_identity_match"] == 1).all()

    def test_no_device_info_flag(self, sample_df):
        """Rows where DeviceInfo is null should have no_device_info=1."""
        df = add_null_features(sample_df)
        null_device_rows = sample_df["DeviceInfo"].isnull()
        assert (df.loc[null_device_rows, "no_device_info"] == 1).all()

    def test_null_flags_binary(self, sample_df):
        """All null flags must be binary."""
        df = add_null_features(sample_df)
        for col in ["no_identity_match", "no_device_info", "no_email_domain"]:
            assert set(df[col].unique()).issubset({0, 1}), f"{col} is not binary"


# ── Imputation tests ───────────────────────────────────────────────────────────

class TestImputation:

    def test_no_nulls_after_imputation(self, sample_df):
        """After imputation, no numeric column should contain nulls."""
        df = impute_numerics(sample_df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert df[numeric_cols].isnull().sum().sum() == 0

    def test_sentinel_value_used(self, sample_df):
        """Nulls should be filled with -999, not 0 or mean."""
        df = impute_numerics(sample_df)
        # card1 had a null — should now be -999
        assert -999 in df["card1"].values


# ── Categorical encoding tests ─────────────────────────────────────────────────

class TestCategoricalEncoding:

    def test_categoricals_are_numeric(self, sample_df):
        """After encoding, categorical columns should be numeric (int codes)."""
        df = encode_categoricals(sample_df)
        for col in ["ProductCD", "card4", "P_emaildomain"]:
            if col in df.columns:
                assert pd.api.types.is_numeric_dtype(df[col]), \
                    f"{col} is not numeric after encoding"

    def test_encoding_handles_nulls(self, sample_df):
        """Encoding should handle null categoricals without raising errors."""
        df = encode_categoricals(sample_df)
        # Should not throw; null categories get code -1 in pandas
        assert True


# ── Integration test ───────────────────────────────────────────────────────────

class TestPipelineIntegration:

    def test_pipeline_preserves_row_count(self, sample_df):
        """Full feature pipeline must not drop any rows."""
        from features import build_features
        df = build_features(sample_df.copy())
        assert len(df) == len(sample_df)

    def test_pipeline_adds_features(self, sample_df):
        """Full pipeline must add new columns beyond the original set."""
        from features import build_features
        original_cols = set(sample_df.columns)
        df = build_features(sample_df.copy())
        new_cols = set(df.columns) - original_cols
        assert len(new_cols) > 5, f"Expected >5 new features, got {len(new_cols)}"

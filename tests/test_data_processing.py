import pandas as pd
import pytest
from src.data_processing import build_features, build_preprocessor

def test_build_features_columns():
    # Sample transaction-level data
    df = pd.DataFrame({
        "CustomerId": [1, 1, 2],
        "TransactionId": ["t1", "t2", "t3"],
        "Value": [100, 200, 50],
        "TransactionStartTime": pd.to_datetime(["2023-01-01 10:00", "2023-01-02 11:00", "2023-01-01 12:00"]),
        "ChannelId": ["c1", "c1", "c2"],
        "ProductCategory": ["cat1", "cat1", "cat2"],
        "is_high_risk": [0, 0, 1]
    })

    customer_df = build_features(df)
    expected_cols = [
        "CustomerId",
        "total_transaction_value",
        "avg_transaction_value",
        "transaction_count",
        "std_transaction_value",
        "most_common_channel",
        "most_common_category",
        "is_high_risk"
    ]

    assert all(col in customer_df.columns for col in expected_cols), "Missing expected columns"
    # Check std_transaction_value NaN handled
    assert customer_df["std_transaction_value"].isnull().sum() == 0, "NaN values in std_transaction_value"

def test_build_preprocessor_transform():
    sample_df = pd.DataFrame({
        "total_transaction_value": [300, 50],
        "avg_transaction_value": [150, 50],
        "transaction_count": [2, 1],
        "std_transaction_value": [70.71, 0.0],
        "most_common_channel": ["c1", "c2"],
        "most_common_category": ["cat1", "cat2"]
    })

    numerical_features = ["total_transaction_value", "avg_transaction_value", "transaction_count", "std_transaction_value"]
    categorical_features = ["most_common_channel", "most_common_category"]

    preprocessor = build_preprocessor(numerical_features, categorical_features)
    X_processed = preprocessor.fit_transform(sample_df)

    # Check output shape
    expected_num_features = len(numerical_features)
    expected_cat_features = sum(len(sample_df[col].unique()) for col in categorical_features)
    expected_total_features = expected_num_features + expected_cat_features

    assert X_processed.shape[1] == expected_total_features, "Processed feature shape mismatch"

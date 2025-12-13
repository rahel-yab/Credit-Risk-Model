import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df["transaction_hour"] = df["TransactionStartTime"].dt.hour
    df["transaction_day"] = df["TransactionStartTime"].dt.day
    df["transaction_month"] = df["TransactionStartTime"].dt.month
    df["transaction_year"] = df["TransactionStartTime"].dt.year

    # Aggregate customer-level features
    customer_features = (
        df.groupby("CustomerId")
        .agg(
            total_transaction_value=("Value", "sum"),
            avg_transaction_value=("Value", "mean"),
            transaction_count=("TransactionId", "count"),
            std_transaction_value=("Value", "std"),
            most_common_channel=("ChannelId", lambda x: x.mode()[0]),
            most_common_category=("ProductCategory", lambda x: x.mode()[0]),
            is_high_risk=("is_high_risk", "max")
        )
        .reset_index()
    )

    # Handle missing values
    customer_features["std_transaction_value"] = customer_features[
        "std_transaction_value"
    ].fillna(0)

    return customer_features

def build_preprocessor(numerical_features: list, categorical_features: list) -> ColumnTransformer:

    # Numeric pipeline
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor

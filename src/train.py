import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


DATA_PATH = "data/processed/transactions_with_target.csv"
MODEL_DIR = "models"
TARGET_COL = "FraudResult"


# Columns to drop (identifiers, not features)
DROP_COLS = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "CustomerId",
    "TransactionStartTime"
]


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")
    return pd.read_csv(path)


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [
        "Amount",
        "Value",
        "PricingStrategy"
    ]

    categorical_features = [
        "CurrencyCode",
        "CountryCode",
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId"
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor


def build_model(preprocessor: ColumnTransformer) -> Pipeline:
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                n_jobs=-1
            ))
        ]
    )
    return model


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    df = df.drop(columns=DROP_COLS, errors="ignore")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("Building model...")
    preprocessor = build_preprocessor(df)
    model = build_model(preprocessor)

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "logistic_regression.pkl")
    joblib.dump(model, model_path)

    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()

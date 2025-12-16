import pandas as pd
import joblib
from pathlib import Path
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

DATA_PATH = Path("data/processed/transactions_with_target.csv")
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(exist_ok=True)

TARGET_COL = "is_high_risk"
DROP_COLS = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "CustomerId",
    "TransactionStartTime",
    "FraudResult",
]

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path}")
    return pd.read_csv(path)

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object", "category"]).columns

    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

def main():
    mlflow.set_experiment("Credit Risk Model")

    with mlflow.start_run():
        print("Loading data...")
        df = load_data(DATA_PATH)

        print("Preparing features and target...")
        X = df.drop(columns=[TARGET_COL] + DROP_COLS, errors="ignore")
        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Building model pipeline...")
        pipeline = build_pipeline(X_train)

        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)

        print("Training model...")
        pipeline.fit(X_train, y_train)

        print("Evaluating model...")
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_proba)

        print(classification_report(y_test, y_pred))
        print("ROC-AUC:", roc_auc)

        # Log metrics
        mlflow.log_metric("accuracy", report["accuracy"])
        mlflow.log_metric("precision", report["weighted avg"]["precision"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])
        mlflow.log_metric("roc_auc", roc_auc)

        # Log model
        mlflow.sklearn.log_model(pipeline, "model")

        model_file = MODEL_PATH / "credit_risk_logistic.pkl"
        joblib.dump(pipeline, model_file)
        print(f"Model saved to {model_file}")

        # Register the model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, "CreditRiskLogisticRegression")

if __name__ == "__main__":
    main()

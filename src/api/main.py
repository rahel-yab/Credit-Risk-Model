from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
from .pydantic_models import TransactionData, PredictionResponse

app = FastAPI(title="Credit Risk Prediction API", version="1.0.0")

# Load the model from MLflow registry
model_name = "CreditRiskLogisticRegression"
model_version = "latest"  # or specify version
model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(transaction: TransactionData):
    # Convert to DataFrame
    data = pd.DataFrame([transaction.dict()])
    
    # Predict probability
    proba = model.predict_proba(data)[:, 1][0]
    
    # Predict class
    prediction = model.predict(data)[0]
    
    return PredictionResponse(
        risk_probability=float(proba),
        is_high_risk=bool(prediction)
    )

@app.get("/")
def read_root():
    return {"message": "Credit Risk Prediction API"}
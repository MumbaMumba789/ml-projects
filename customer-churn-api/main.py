from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using a trained ML model.",
    version="1.0.0",
)

MODEL_PATH = Path(__file__).parent / "churn_model.joblib"
model = joblib.load(MODEL_PATH)


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=1000)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(customer: Customer):
    customer_data = pd.DataFrame([customer.model_dump()])

    probability = float(model.predict_proba(customer_data)[0][1])
    prediction = "Churn" if probability >= 0.5 else "No churn"

    return {
        "prediction": prediction,
        "churn_probability": round(probability, 4),
    }

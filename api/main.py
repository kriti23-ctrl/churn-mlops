from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Churn Prediction API")

model = joblib.load("model.pkl")

class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: float
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running!"}

@app.post("/predict")
def predict(data: CustomerData):
    df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return {
        "churn": bool(prediction),
        "churn_probability": round(float(probability), 4)
    }

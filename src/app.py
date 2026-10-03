from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import joblib
import uuid
from datetime import datetime
import json
import time

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "logistic_regression.pkl"
AUDIT_LOG = BASE_DIR / "logs" / "prediction_audit.jsonl"

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Responsible MLOps API for credit risk prediction",
    version="1.0.0"
)

# Load validated model
try:
    model = joblib.load(MODEL_PATH)
    MODEL_STATUS = "loaded"
except Exception:
    model = None
    MODEL_STATUS = "failed"


class CreditApplication(BaseModel):
    checking_status: str
    duration_months: int
    credit_history: str
    purpose: str
    credit_amount: int
    savings_status: str
    employment_since: str
    installment_rate: int
    personal_status_sex: str
    other_debtors: str
    residence_since: int
    property: str
    age: int
    other_installment_plans: str
    housing: str
    existing_credits: int
    job: str
    dependents: int
    telephone: str
    foreign_worker: str


@app.get("/")
def root():
    return {
        "message": "Credit Risk Prediction API",
        "model": "Logistic Regression",
        "model_status": MODEL_STATUS
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_status": MODEL_STATUS
    }


@app.post("/predict")
def predict(application: CreditApplication):

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not available."
        )

    start_time = time.time()

    try:
        input_data = application.model_dump()

        # Convert input into DataFrame
        import pandas as pd
        input_df = pd.DataFrame([input_data])

        # Get probability of bad credit risk
        probability = float(model.predict_proba(input_df)[0][1])

        prediction = int(model.predict(input_df)[0])

        # Risk classification
        if probability >= 0.70:
            risk_level = "HIGH_RISK"
            decision = "HUMAN_REVIEW_REQUIRED"

        elif probability >= 0.40:
            risk_level = "MEDIUM_RISK"
            decision = "HUMAN_REVIEW_REQUIRED"

        else:
            risk_level = "LOW_RISK"
            decision = "AUTOMATED_DECISION"

        request_id = str(uuid.uuid4())
        processing_time = time.time() - start_time

        # Audit trail
        audit_record = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "model": "logistic_regression",
            "prediction": prediction,
            "risk_probability": round(probability, 4),
            "risk_level": risk_level,
            "decision": decision,
            "processing_time_seconds": round(processing_time, 4)
        }

        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)

        with open(AUDIT_LOG, "a", encoding="utf-8") as file:
            file.write(json.dumps(audit_record) + "\n")

        return {
            "request_id": request_id,
            "prediction": prediction,
            "risk_probability": round(probability, 4),
            "risk_level": risk_level,
            "decision": decision,
            "model": "Logistic Regression",
            "processing_time_seconds": round(processing_time, 4)
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )
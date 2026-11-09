"""FastAPI backend.

Run from the repo root:  uvicorn src.api:app --reload
Docs: http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import CATEGORICAL
from src.predict import load_meta, load_model, predict_one

app = FastAPI(title="Netflix Churn Prediction API", version="1.0")

try:
    MODEL, META = load_model(), load_meta()
except FileNotFoundError:
    MODEL, META = None, None


class Customer(BaseModel):
    age: int = Field(ge=10, le=110)
    gender: str
    subscription_type: str
    watch_hours: float = Field(ge=0)
    last_login_days: int = Field(ge=0)
    region: str
    device: str
    monthly_fee: float = Field(ge=0)
    payment_method: str
    number_of_profiles: int = Field(ge=1)
    avg_watch_time_per_day: float = Field(ge=0)
    favorite_genre: str


def _require_model():
    if MODEL is None:
        raise HTTPException(503, "Model not trained yet. Run: python -m src.train")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.get("/meta")
def meta():
    _require_model()
    return META


@app.post("/predict")
def predict(customer: Customer):
    _require_model()
    record = customer.model_dump()
    for col in CATEGORICAL:
        if record[col] not in META["categories"][col]:
            raise HTTPException(422, f"Unknown {col}: '{record[col]}'. Allowed: {META['categories'][col]}")
    result = predict_one(MODEL, record)
    result["model_used"] = META["best_model"]
    return result

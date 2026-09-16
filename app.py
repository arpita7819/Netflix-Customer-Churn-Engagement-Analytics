"""
app.py — FastAPI backend for Netflix Churn Prediction.
Run: uvicorn app:app --reload
"""

import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# ── Load artefacts ────────────────────────────────────────────────────────────
def _load(name: str):
    with open(os.path.join(MODEL_DIR, name), "rb") as f:
        return pickle.load(f)

try:
    model    = _load("model.pkl")
    encoders = _load("encoders.pkl")
    features = _load("feature_columns.pkl")
    meta     = _load("meta.pkl")
except FileNotFoundError:
    raise RuntimeError("Run train.py first to generate model artefacts.")

app = FastAPI(
    title="Netflix Churn Prediction API",
    version="1.0",
    description="Predicts whether a Netflix customer will churn.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request schema ─────────────────────────────────────────────────────────────
class CustomerData(BaseModel):
    age:                    int   = Field(..., ge=1, le=120, example=35)
    gender:                 Literal["Male", "Female", "Other"]
    subscription_type:      Literal["Basic", "Standard", "Premium"]
    watch_hours:            float = Field(..., ge=0, example=10.5)
    last_login_days:        int   = Field(..., ge=0, example=15)
    region:                 Literal["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]
    device:                 Literal["Mobile", "TV", "Tablet", "Laptop"]
    monthly_fee:            float = Field(..., ge=0, example=13.99)
    payment_method:         Literal["Credit Card", "Debit Card", "PayPal", "Gift Card", "Crypto"]
    number_of_profiles:     int   = Field(..., ge=1, le=10, example=2)
    avg_watch_time_per_day: float = Field(..., ge=0, example=1.2)
    favorite_genre:         Literal["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Documentary", "Thriller"]


# ── Prediction endpoint ────────────────────────────────────────────────────────
@app.post("/predict")
def predict(data: CustomerData):
    row = data.dict()
    encoded = {}
    for col, val in row.items():
        if col in encoders:
            le = encoders[col]
            if val not in le.classes_:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unknown value '{val}' for column '{col}'. "
                           f"Allowed: {list(le.classes_)}"
                )
            encoded[col] = int(le.transform([val])[0])
        else:
            encoded[col] = val

    X = pd.DataFrame([[encoded[f] for f in features]], columns=features)
    pred  = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])

    risk_level = "High" if proba >= 0.7 else ("Medium" if proba >= 0.4 else "Low")

    return {
        "churned":    bool(pred),
        "churn_probability": round(proba, 4),
        "risk_level": risk_level,
        "model_used": meta["best_model"],
    }


# ── Metadata endpoints ─────────────────────────────────────────────────────────
@app.get("/meta")
def get_meta():
    return {
        "best_model":    meta["best_model"],
        "best_roc_auc":  round(meta["best_auc"], 4),
        "all_results":   {k: {m: round(v, 4) for m, v in r.items()}
                          for k, r in meta["results"].items()},
        "feature_columns": meta["feature_columns"],
        "encoder_classes": meta["encoder_classes"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}

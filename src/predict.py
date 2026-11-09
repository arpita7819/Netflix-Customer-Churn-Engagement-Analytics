"""Shared prediction helpers used by both the API and the Streamlit app."""
import json

import joblib
import pandas as pd

from src.config import FEATURES, MODEL_DIR

MODEL_PATH = MODEL_DIR / "model.joblib"
META_PATH = MODEL_DIR / "metrics.json"


def risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "High"
    if prob >= 0.4:
        return "Medium"
    return "Low"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python -m src.train")
    return joblib.load(MODEL_PATH)


def load_meta() -> dict:
    if not META_PATH.exists():
        raise FileNotFoundError("metrics.json not found. Run: python -m src.train")
    with open(META_PATH) as f:
        return json.load(f)


def predict_one(model, record: dict) -> dict:
    X = pd.DataFrame([record])[FEATURES]
    prob = float(model.predict_proba(X)[0, 1])
    return {
        "churned": prob >= 0.5,
        "churn_probability": round(prob, 4),
        "risk_level": risk_level(prob),
    }

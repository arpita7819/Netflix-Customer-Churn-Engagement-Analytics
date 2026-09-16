"""
train.py — Trains a Random Forest churn classifier and saves the model + encoder.
Run: python train.py
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "netflix_customer_churn.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 1. Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.drop(columns=["customer_id"], inplace=True)

# ── 2. Encode categoricals ───────────────────────────────────────────────────
CATEGORICAL = ["gender", "subscription_type", "region", "device",
               "payment_method", "favorite_genre"]

encoders: dict[str, LabelEncoder] = {}
for col in CATEGORICAL:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ── 3. Split ─────────────────────────────────────────────────────────────────
X = df.drop(columns=["churned"])
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Train models & pick best ──────────────────────────────────────────────
candidates = {
    "RandomForest":  RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "GradientBoost": GradientBoostingClassifier(n_estimators=150, random_state=42),
    "Logistic":      LogisticRegression(max_iter=1000, random_state=42),
}

best_name, best_model, best_auc = None, None, 0.0
results = {}

for name, model in candidates.items():
    model.fit(X_train, y_train)
    preds  = model.predict(X_test)
    proba  = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, proba)
    acc    = accuracy_score(y_test, preds)
    results[name] = {"accuracy": acc, "roc_auc": auc}
    print(f"\n{'='*50}")
    print(f"  {name}  |  Accuracy: {acc:.4f}  |  ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, preds))
    if auc > best_auc:
        best_auc, best_name, best_model = auc, name, model

print(f"\nBest model: {best_name}  (ROC-AUC = {best_auc:.4f})")

# ── 5. Feature importance (for RF / GB) ─────────────────────────────────────
if hasattr(best_model, "feature_importances_"):
    fi = pd.Series(best_model.feature_importances_, index=X.columns)
    print("\nTop-10 feature importances:")
    print(fi.sort_values(ascending=False).head(10))

# ── 6. Save artefacts ────────────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
    pickle.dump(best_model, f)

with open(os.path.join(MODEL_DIR, "encoders.pkl"), "wb") as f:
    pickle.dump(encoders, f)

with open(os.path.join(MODEL_DIR, "feature_columns.pkl"), "wb") as f:
    pickle.dump(list(X.columns), f)

meta = {
    "best_model": best_name,
    "best_auc":   best_auc,
    "results":    results,
    "feature_columns": list(X.columns),
    "categorical_columns": CATEGORICAL,
    "encoder_classes": {col: list(le.classes_) for col, le in encoders.items()},
}
with open(os.path.join(MODEL_DIR, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("\nArtefacts saved to ./models/")

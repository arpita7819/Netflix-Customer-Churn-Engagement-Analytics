"""Train, evaluate and save the churn model.

Run from the repo root:
    python -m src.train
    python -m src.train --exclude avg_watch_time_per_day   # retrain without a suspect feature

Outputs
    models/model.joblib   full sklearn Pipeline (preprocessing + model)
    models/metrics.json   metrics, feature importance, leakage checks, category values
    images/06-08_*.png    confusion matrix, ROC curves, feature importance
"""
import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import load_data
from src.config import (CATEGORICAL, ENGAGEMENT, FEATURES, IMAGES_DIR, MODEL_DIR, NUMERIC,
                        RANDOM_STATE, TARGET, TEST_SIZE)


def build_pipeline(model, features):
    numeric = [c for c in NUMERIC if c in features]
    categorical = [c for c in CATEGORICAL if c in features]
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def score(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def main(exclude):
    df = load_data()
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in CSV: {missing}")

    features = [f for f in FEATURES if f not in exclude]
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    results, fitted = {}, {}
    for name, model in candidates.items():
        pipe = build_pipeline(model, features).fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        cv_auc = cross_val_score(build_pipeline(clone(model), features), X_train, y_train,
                                 cv=cv, scoring="roc_auc")
        results[name] = {**score(y_test, proba), "cv_auc_mean": cv_auc.mean(), "cv_auc_std": cv_auc.std()}
        fitted[name] = pipe
        print(f"{name:20s} " + "  ".join(f"{k}={v:.4f}" for k, v in results[name].items()))

    best = max(results, key=lambda n: results[n]["roc_auc"])
    best_pipe = fitted[best]
    print(f"\nBest model (test ROC-AUC): {best}")

    # ---- Feature importance (permutation, on original columns) ----
    perm = permutation_importance(best_pipe, X_test, y_test, scoring="roc_auc",
                                  n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    importance = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
    importance = importance[importance.index.isin(features)]

    # ---- Leakage diagnostics ----
    single_auc = {}
    for c in NUMERIC:
        a = roc_auc_score(y, df[c])
        single_auc[c] = float(max(a, 1 - a))
    top_feature = importance.index[0]
    reduced = [f for f in features if f != top_feature]
    ablate = build_pipeline(clone(candidates[best]), reduced).fit(X_train, y_train)
    auc_without = roc_auc_score(y_test, ablate.predict_proba(X_test)[:, 1])
    no_eng = [f for f in features if f not in ENGAGEMENT]
    auc_no_eng = None
    if no_eng:
        abl2 = build_pipeline(clone(candidates[best]), no_eng).fit(X_train, y_train)
        auc_no_eng = float(roc_auc_score(y_test, abl2.predict_proba(X_test)[:, 1]))
    leakage = {
        "engagement_features": ENGAGEMENT,
        "auc_without_engagement_features": auc_no_eng,
        "single_feature_auc": single_auc,
        "top_feature": top_feature,
        "auc_with_top_feature": results[best]["roc_auc"],
        "auc_without_top_feature": float(auc_without),
        "watch_hours_vs_avg_watch_time_corr": (
            float(df["watch_hours"].corr(df["avg_watch_time_per_day"])) if
            {"watch_hours", "avg_watch_time_per_day"} <= set(df.columns) else None),
    }
    print(f"\nTop feature: {top_feature}. AUC with = {results[best]['roc_auc']:.4f}, "
          f"without = {auc_without:.4f}")
    print(f"AUC without ALL engagement features {ENGAGEMENT}: {auc_no_eng}")
    print("Single-feature AUC (>0.9 is suspicious):",
          {k: round(v, 3) for k, v in single_auc.items()})

    # ---- Plots ----
    IMAGES_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(best_pipe, X_test, y_test, ax=ax, cmap="Reds")
    ax.set_title(f"Confusion Matrix - {best}")
    fig.tight_layout(); fig.savefig(IMAGES_DIR / "06_confusion_matrix.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC Curves")
    fig.tight_layout(); fig.savefig(IMAGES_DIR / "07_roc_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    importance.sort_values().plot.barh(ax=ax, color="#E50914")
    ax.set_title("Permutation Importance (drop in ROC-AUC)")
    fig.tight_layout(); fig.savefig(IMAGES_DIR / "08_feature_importance.png", dpi=150); plt.close(fig)

    # ---- Save artifacts ----
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_pipe, MODEL_DIR / "model.joblib")
    meta = {
        "best_model": best,
        "features": features,
        "excluded_features": exclude,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "overall_churn_rate": float(y.mean()),
        "models": {k: {m: float(v) for m, v in r.items()} for k, r in results.items()},
        "feature_importance": {k: float(v) for k, v in importance.items()},
        "leakage_check": leakage,
        "categories": {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL},
        "numeric_ranges": {c: [float(df[c].min()), float(df[c].max())] for c in NUMERIC},
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved model + metrics to {MODEL_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=[], help="features to leave out of the model")
    main(ap.parse_args().exclude)

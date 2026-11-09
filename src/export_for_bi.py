"""Create an enriched CSV for Power BI / Tableau.

Run from the repo root:  python -m src.export_for_bi
Output: data/processed/netflix_churn_enriched.csv
Adds: age_group, inactivity_bucket, engagement_quartile, at_risk_flag,
      and (if a model is trained) churn_probability + risk_level.
"""
import numpy as np
import pandas as pd

from src.config import FEATURES, PROCESSED_DIR, TARGET
from src.data import load_data
from src.predict import MODEL_PATH, load_model, risk_level


def main():
    df = load_data()
    df["age_group"] = pd.cut(df["age"], [0, 24, 34, 44, 54, 200],
                             labels=["18-24", "25-34", "35-44", "45-54", "55+"])
    df["inactivity_bucket"] = pd.cut(df["last_login_days"], [-1, 7, 14, 30, np.inf],
                                     labels=["0-7 days", "8-14 days", "15-30 days", "30+ days"])
    df["engagement_quartile"] = pd.qcut(df["avg_watch_time_per_day"].rank(method="first"),
                                        4, labels=[1, 2, 3, 4]).astype(int)
    df["at_risk_flag"] = ((df[TARGET] == 0) & (df["engagement_quartile"] <= 2)
                          & (df["last_login_days"] >= 15)).astype(int)
    if MODEL_PATH.exists():
        proba = load_model().predict_proba(df[FEATURES])[:, 1]
        df["churn_probability"] = proba.round(4)
        df["risk_level"] = [risk_level(p) for p in proba]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "netflix_churn_enriched.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()

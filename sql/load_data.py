"""Load the Netflix churn CSV into a local SQLite database.

Usage (from repo root):
    python sql/load_data.py
Creates: netflix.db with table `netflix_customers`
"""
import sqlite3
from pathlib import Path

import pandas as pd

CSV_PATH = Path("data/netflix_customer_churn.csv")
DB_PATH = Path("netflix.db")

df = pd.read_csv(CSV_PATH)
with sqlite3.connect(DB_PATH) as conn:
    df.to_sql("netflix_customers", conn, if_exists="replace", index=False)
    rows = conn.execute("SELECT COUNT(*) FROM netflix_customers").fetchone()[0]
print(f"Loaded {rows} rows into {DB_PATH} (table: netflix_customers)")

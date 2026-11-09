"""Central configuration: paths, column groups, constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "netflix_customer_churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
IMAGES_DIR = ROOT / "images"

TARGET = "churned"
ID_COL = "customer_id"

CATEGORICAL = ["gender", "subscription_type", "region", "device", "payment_method", "favorite_genre"]
NUMERIC = ["age", "watch_hours", "last_login_days", "monthly_fee",
           "number_of_profiles", "avg_watch_time_per_day"]
FEATURES = CATEGORICAL + NUMERIC

# Behavioural features that overlap heavily (avg_watch_time_per_day ~ watch_hours / days)
ENGAGEMENT = ["watch_hours", "avg_watch_time_per_day", "last_login_days"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

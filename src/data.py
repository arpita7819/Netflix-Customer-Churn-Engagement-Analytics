"""Single place where the dataset is loaded and cleaned."""
import pandas as pd

from src.config import DATA_PATH

MAX_HOURS_PER_DAY = 24  # physically impossible to watch more than 24 h/day


def load_data() -> pd.DataFrame:
    """Load the CSV and apply the documented cleaning step.

    Cleaning: `avg_watch_time_per_day` has a handful of rows above 24 hours/day
    (impossible; they all have last_login_days = 0 and avg == total watch_hours,
    i.e. a division artefact). Values are capped at 24.
    """
    df = pd.read_csv(DATA_PATH)
    df["avg_watch_time_per_day"] = df["avg_watch_time_per_day"].clip(upper=MAX_HOURS_PER_DAY)
    return df

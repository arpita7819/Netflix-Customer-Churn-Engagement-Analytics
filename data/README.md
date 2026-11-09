# Data

Place your dataset here as:

```
data/netflix_customer_churn.csv
```

(5,000 rows x 14 columns.)

## Data dictionary

| Column | Type | Description |
|---|---|---|
| customer_id | string | Unique customer UUID (not used as a feature) |
| age | int | Customer age |
| gender | category | Male / Female / Other |
| subscription_type | category | Basic / Standard / Premium |
| watch_hours | float | Total hours watched |
| last_login_days | int | Days since last login |
| region | category | Geographic region |
| device | category | Primary viewing device |
| monthly_fee | float | Monthly subscription cost |
| payment_method | category | Payment method used |
| number_of_profiles | int | Profiles under the account |
| avg_watch_time_per_day | float | Average hours watched per day |
| favorite_genre | category | Most-watched genre |
| churned | int | **Target**: 1 = churned, 0 = retained |

> Note: this is a synthetic/public practice dataset, so results are illustrative.

`data/processed/` is filled by `python -m src.export_for_bi` and holds the enriched file for Power BI / Tableau.

# Power BI / Tableau Dashboard Guide

This is the piece that turns the project from "ML demo" into "analyst portfolio". Build one dashboard and add the screenshot to `images/` and the link to the main README.

## 1. Get the data
```bash
python -m src.export_for_bi
```
Load `data/processed/netflix_churn_enriched.csv` into Power BI (Get Data -> Text/CSV) or Tableau.
`at_risk_flag` = still subscribed + engagement quartile 1-2 + inactive 15+ days.
It already includes `age_group`, `inactivity_bucket`, `engagement_quartile`, `at_risk_flag`, `churn_probability` and `risk_level`.

## 2. DAX measures (Power BI)
```DAX
Total Customers = COUNTROWS(netflix_churn_enriched)
Churned Customers = CALCULATE([Total Customers], netflix_churn_enriched[churned] = 1)
Churn Rate % = DIVIDE([Churned Customers], [Total Customers])
Monthly Revenue Lost = CALCULATE(SUM(netflix_churn_enriched[monthly_fee]), netflix_churn_enriched[churned] = 1)
At-Risk Customers = CALCULATE([Total Customers], netflix_churn_enriched[at_risk_flag] = 1)
Annual Revenue at Risk = CALCULATE(SUM(netflix_churn_enriched[monthly_fee]), netflix_churn_enriched[at_risk_flag] = 1) * 12
Avg Watch Time = AVERAGE(netflix_churn_enriched[avg_watch_time_per_day])
```

## 3. Suggested pages

**Page 1 - Executive Overview**
- KPI cards: Total Customers, Churn Rate %, Monthly Revenue Lost, Annual Revenue at Risk
- Bar: Churn Rate % by subscription_type
- Map or bar: Churn Rate % by region
- Donut: churned vs retained
- Slicers: subscription_type, region, device

**Page 2 - Churn Drivers**
- Bar: Churn Rate % by payment_method
- Column: Churn Rate % by engagement_quartile
- Column: Churn Rate % by inactivity_bucket
- Matrix heatmap: subscription_type x engagement_quartile (conditional formatting on Churn Rate %)
- Column: Churn Rate % by age_group

**Page 3 - Retention Targets**
- Card: At-Risk Customers and Annual Revenue at Risk
- Table: customer_id, subscription_type, region, last_login_days, avg_watch_time_per_day, churn_probability (filter at_risk_flag = 1, sort by churn_probability)
- Bar: Annual Revenue at Risk by subscription_type

## 4. Design tips
- Netflix palette: red `#E50914`, dark grey `#564D4D`, black background optional.
- One message per chart; title each chart with the insight (e.g. "Basic plan churns 18 points more than Premium").
- Export a screenshot of each page to `images/` and publish to Power BI Service ("Publish to web") for a shareable link.

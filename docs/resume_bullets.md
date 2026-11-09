# Resume Bullets (numbers verified on your dataset)

## Project entry

**Netflix Customer Churn & Engagement Analytics** | Python, SQL, Pandas, Scikit-learn, Streamlit, FastAPI | [GitHub link]

- Analyzed 5,000 subscriber records (14 attributes, 50.3% churn, ~$33K/month revenue lost) with SQL (CTEs, window functions) and Python; identified Basic-plan churn of 61.8% vs 43.7% for Premium.
- Segmented customers by engagement and inactivity: churn fell from 93.6% in the lowest engagement quartile to 1.0% in the highest, and jumped to 75.1% after 30+ days inactive, which gave a clear day-15 re-engagement trigger.
- Built a retention target list of 337 at-risk active customers worth ~$62.7K/year in recurring revenue, with 7 data-backed recommendations (payment-method migration, multi-profile plans, Premium retention).
- Cleaned data (capped 10 impossible >24 h/day values) and compared 3 classifiers with 5-fold CV; ran feature-group ablations showing engagement drives the signal (ROC-AUC 0.998 → 0.63 without it).
- Deployed an interactive Streamlit app with a FastAPI scoring endpoint for real-time churn risk.

> Add this bullet **only after you build the dashboard:**
> - Built a 3-page Power BI dashboard (KPIs, churn drivers, retention target list) to track churn and revenue at risk.

## Shorter version
- Analyzed churn for 5,000 subscribers with SQL and Python; found engagement is the dominant driver (93.6% → 1.0% churn across quartiles) and quantified **$62.7K/year** of at-risk revenue across 337 customers.
- Delivered 7 retention recommendations and a Streamlit churn-scoring app.

## Skills line
SQL (CTEs, window functions) | Python (Pandas, NumPy, Scikit-learn) | Data Cleaning | EDA | Segmentation | Data Visualization | Power BI | Streamlit | FastAPI | Git

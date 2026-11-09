# Interview Prep: Questions You Will Be Asked About This Project

## "Walk me through the project."
Business problem → data cleaning → SQL/EDA findings → recommendations → model as a supporting tool. Lead with the insight (engagement is the dominant driver; here is the retention list and what it is worth), not the algorithm.

## "Your model has 99% accuracy. Isn't that suspicious?"
Yes, and I checked. The dataset is synthetic, so the signal is cleaner than real data. `avg_watch_time_per_day` had a single-feature AUC of 0.92 and is essentially `watch_hours` divided by days, so the engagement features overlap. Dropping one feature barely changed the AUC (0.998 → 0.993), so I tested the whole engagement group: without it the ROC-AUC fell to 0.63. That tells me churn here is almost entirely engagement-driven. On real data I'd expect lower scores and I'd validate on a time-based split.

## "Why ROC-AUC / F1 and not just accuracy?"
The business cares about catching churners (recall) without wasting offers on loyal users (precision). Accuracy hides that trade-off, so I report all of them plus cross-validation.

## "How did you define 'at-risk'?"
Still subscribed, bottom-half engagement (quartiles 1-2, ≤0.29 hrs/day, where 74-94% churn), and inactive 15+ days. Customers with that profile churn at 85.2% across the full base, so the definition is evidence-based. The thresholds are adjustable, and I'd tune them against campaign ROI.

## "Why is Premium your biggest revenue problem if it has the lowest churn?"
Rate and revenue differ: Premium churns at 43.7% but each customer pays more, so it loses $13.3K/month versus $9.2K for Basic. I'd prioritise by revenue at risk, not just rate.

## "What data quality issues did you find?"
10 rows had average watch time above 24 hours/day (impossible). They all had a login within the last day and an average equal to (or half of) total hours, so I treated it as a division artefact and capped at 24. I documented it in one place (`src/data.py`) so SQL, EDA and the model stay consistent.

## "What did you find that was NOT a driver?"
Region, device, gender and genre all sit within about ±3 points of the average churn rate, so I'd tell the business not to spend effort segmenting on them. Saying what doesn't matter is as useful as what does.

## "What are the limitations?"
Synthetic data; no time dimension (no cohorts or survival analysis); no offer-cost data (no ROI); engagement is measured alongside churn so the model is descriptive, not a true early-warning system.

## "What would you do next?"
A/B test the win-back offer with a control group, estimate customer lifetime value, add signup/cancel dates for cohort retention, and use a time-based validation split.

## SQL questions to rehearse (all in `sql/churn_analysis.sql`)
- Churn rate by segment with `GROUP BY` + `AVG(churned)`
- `NTILE` for engagement quartiles; `RANK() OVER (PARTITION BY ...)` for ranking regions within each plan
- A CTE that builds the at-risk list, then a second query that aggregates its revenue

-- =====================================================================
-- Netflix Customer Churn & Engagement Analysis
-- Dialect: SQLite 3.25+ (window functions). Also works in PostgreSQL
-- with minor tweaks (ROUND needs ::numeric casts).
-- Table: netflix_customers  |  Target: churned (1 = churned, 0 = retained)
-- =====================================================================


-- Q1. Headline KPIs ----------------------------------------------------
SELECT
    COUNT(*)                                   AS total_customers,
    SUM(churned)                               AS churned_customers,
    ROUND(100.0 * AVG(churned), 1)             AS churn_rate_pct,
    ROUND(AVG(monthly_fee), 2)                 AS avg_monthly_fee,
    ROUND(SUM(CASE WHEN churned = 1 THEN monthly_fee ELSE 0 END), 2) AS monthly_revenue_lost
FROM netflix_customers;


-- Q2. Churn by subscription plan ---------------------------------------
SELECT
    subscription_type,
    COUNT(*)                          AS customers,
    ROUND(100.0 * AVG(churned), 1)    AS churn_rate_pct,
    ROUND(SUM(CASE WHEN churned = 1 THEN monthly_fee ELSE 0 END), 2) AS monthly_revenue_lost
FROM netflix_customers
GROUP BY subscription_type
ORDER BY churn_rate_pct DESC;


-- Q3. Churn by region, with gap vs. overall average --------------------
SELECT
    region,
    COUNT(*)                                   AS customers,
    ROUND(100.0 * AVG(churned), 1)             AS churn_rate_pct,
    ROUND(100.0 * AVG(churned)
          - (SELECT 100.0 * AVG(churned) FROM netflix_customers), 1) AS pts_vs_overall
FROM netflix_customers
GROUP BY region
ORDER BY churn_rate_pct DESC;


-- Q4. Churn by payment method ------------------------------------------
SELECT
    payment_method,
    COUNT(*)                          AS customers,
    ROUND(100.0 * AVG(churned), 1)    AS churn_rate_pct
FROM netflix_customers
GROUP BY payment_method
ORDER BY churn_rate_pct DESC;


-- Q5. Churn by device and favorite genre --------------------------------
SELECT
    device,
    favorite_genre,
    COUNT(*)                          AS customers,
    ROUND(100.0 * AVG(churned), 1)    AS churn_rate_pct
FROM netflix_customers
GROUP BY device, favorite_genre
HAVING COUNT(*) >= 30                 -- ignore tiny segments
ORDER BY churn_rate_pct DESC
LIMIT 10;


-- Q6. Churn by age group -------------------------------------------------
SELECT
    CASE
        WHEN age < 25 THEN '18-24'
        WHEN age < 35 THEN '25-34'
        WHEN age < 45 THEN '35-44'
        WHEN age < 55 THEN '45-54'
        ELSE '55+'
    END                               AS age_group,
    COUNT(*)                          AS customers,
    ROUND(100.0 * AVG(churned), 1)    AS churn_rate_pct
FROM netflix_customers
GROUP BY age_group
ORDER BY age_group;


-- Q7. Engagement quartiles (NTILE window function) ---------------------
-- Quartile 1 = least engaged, 4 = most engaged (by avg daily watch time)
WITH engagement AS (
    SELECT
        *,
        NTILE(4) OVER (ORDER BY avg_watch_time_per_day) AS engagement_quartile
    FROM netflix_customers
)
SELECT
    engagement_quartile,
    COUNT(*)                                       AS customers,
    ROUND(MIN(avg_watch_time_per_day), 2)          AS min_hours_per_day,
    ROUND(MAX(avg_watch_time_per_day), 2)          AS max_hours_per_day,
    ROUND(100.0 * AVG(churned), 1)                 AS churn_rate_pct
FROM engagement
GROUP BY engagement_quartile
ORDER BY engagement_quartile;


-- Q8. Inactivity buckets (days since last login) -----------------------
SELECT
    CASE
        WHEN last_login_days <= 7  THEN '0-7 days'
        WHEN last_login_days <= 14 THEN '8-14 days'
        WHEN last_login_days <= 30 THEN '15-30 days'
        ELSE '30+ days'
    END                               AS inactivity_bucket,
    COUNT(*)                          AS customers,
    ROUND(100.0 * AVG(churned), 1)    AS churn_rate_pct
FROM netflix_customers
GROUP BY inactivity_bucket
ORDER BY MIN(last_login_days);


-- Q9. Rank regions by churn WITHIN each plan (RANK window function) ---
WITH seg AS (
    SELECT
        subscription_type,
        region,
        COUNT(*)                       AS customers,
        100.0 * AVG(churned)           AS churn_rate_pct
    FROM netflix_customers
    GROUP BY subscription_type, region
)
SELECT
    subscription_type,
    region,
    customers,
    ROUND(churn_rate_pct, 1) AS churn_rate_pct,
    RANK() OVER (PARTITION BY subscription_type ORDER BY churn_rate_pct DESC) AS risk_rank
FROM seg
ORDER BY subscription_type, risk_rank;


-- Q10. At-risk ACTIVE customers (retention target list) ----------------
-- Quartiles are computed over ALL customers (same basis as Q7). At-risk =
-- still subscribed + bottom-half engagement (quartiles 1-2, ~<=0.29 hrs/day,
-- where 74-94% of customers churn) + inactive for 15+ days.
WITH ranked AS (
    SELECT
        *,
        NTILE(4) OVER (ORDER BY avg_watch_time_per_day) AS engagement_quartile
    FROM netflix_customers
)
SELECT
    customer_id,
    subscription_type,
    region,
    last_login_days,
    avg_watch_time_per_day,
    monthly_fee
FROM ranked
WHERE churned = 0
  AND engagement_quartile <= 2
  AND last_login_days >= 15
ORDER BY monthly_fee DESC, last_login_days DESC;


-- Q11. Revenue at risk from that at-risk group --------------------------
WITH ranked AS (
    SELECT
        *,
        NTILE(4) OVER (ORDER BY avg_watch_time_per_day) AS engagement_quartile
    FROM netflix_customers
),
at_risk AS (
    SELECT * FROM ranked
    WHERE churned = 0
      AND engagement_quartile <= 2
      AND last_login_days >= 15
)
SELECT
    subscription_type,
    COUNT(*)                          AS at_risk_customers,
    ROUND(SUM(monthly_fee), 2)        AS monthly_revenue_at_risk,
    ROUND(SUM(monthly_fee) * 12, 2)   AS annual_revenue_at_risk
FROM at_risk
GROUP BY subscription_type
ORDER BY annual_revenue_at_risk DESC;


-- Q12. Does a shared account (more profiles) reduce churn? -------------
SELECT
    number_of_profiles,
    COUNT(*)                          AS customers,
    ROUND(100.0 * AVG(churned), 1)    AS churn_rate_pct
FROM netflix_customers
GROUP BY number_of_profiles
ORDER BY number_of_profiles;

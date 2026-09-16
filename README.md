# 🎬 Netflix Customer Churn Prediction

End-to-end machine learning project — data science, REST API backend, and interactive Streamlit frontend.

## Project Structure

```
netflix_churn_project/
├── data/
│   └── netflix_customer_churn.csv   # Raw dataset (5 000 rows × 14 cols)
├── models/                          # Auto-generated after training
│   ├── model.pkl                    # Best trained model
│   ├── encoders.pkl                 # Label encoders for categoricals
│   ├── feature_columns.pkl          # Ordered feature list
│   └── meta.pkl                     # Training metrics + encoder classes
├── train.py                         # Model training script
├── app.py                           # FastAPI backend
├── frontend.py                      # Streamlit frontend
├── reports.docx                     # Full project report (9 sections, 10 tables)
├── requirements.txt
└── README.md
```

## Dataset

| Column | Description |
|---|---|
| `customer_id` | Unique UUID |
| `age` | Customer age |
| `gender` | Male / Female / Other |
| `subscription_type` | Basic / Standard / Premium |
| `watch_hours` | Total hours watched |
| `last_login_days` | Days since last login |
| `region` | Geographic region |
| `device` | Primary viewing device |
| `monthly_fee` | Monthly subscription cost |
| `payment_method` | Payment method used |
| `number_of_profiles` | Profiles under account |
| `avg_watch_time_per_day` | Average hours/day |
| `favorite_genre` | Most-watched genre |
| `churned` | **Target** — 1 = churned, 0 = retained |

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```
Trains three classifiers (Random Forest, Gradient Boosting, Logistic Regression) and saves the best one to `models/`.

### 3. Start the backend API
```bash
uvicorn app:app --reload
```
API available at **http://localhost:8000**  
Swagger docs at **http://localhost:8000/docs**

### 4. Launch the frontend
```bash
streamlit run frontend.py
```
App opens at **http://localhost:8501**

---

## Frontend Pages

| Page | Description |
|---|---|
| 🏠 Dashboard | KPI metrics, churn rate overview, by-region and by-subscription charts |
| 📊 EDA & Insights | Demographics, viewing behaviour, financials, correlation heatmap |
| 🔮 Predict Churn | Real-time prediction form → FastAPI → gauge + risk signals |
| 📈 Model Performance | Accuracy/AUC comparison, feature importances, encoder info |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Predict churn for a single customer |
| `GET` | `/meta` | Model metadata and performance |
| `GET` | `/health` | Health check |

### Example prediction request
```json
POST /predict
{
  "age": 35,
  "gender": "Male",
  "subscription_type": "Standard",
  "watch_hours": 10.5,
  "last_login_days": 15,
  "region": "Europe",
  "device": "TV",
  "monthly_fee": 13.99,
  "payment_method": "Credit Card",
  "number_of_profiles": 2,
  "avg_watch_time_per_day": 1.2,
  "favorite_genre": "Action"
}
```

### Example response
```json
{
  "churned": false,
  "churn_probability": 0.3421,
  "risk_level": "Low",
  "model_used": "RandomForest"
}
```

---

## Project Report

A comprehensive Word document report (`reports.docx`) is included in the project root. It covers all aspects of the project in 9 sections with 10 data tables:

| Section | Contents |
|---|---|
| 1. Executive Summary | Project overview, key metrics at a glance |
| 2. Dataset Overview | Column descriptions (14 features), class distribution |
| 3. Exploratory Data Analysis | Churn by subscription, region, payment method, viewing behaviour |
| 4. Feature Engineering & Preprocessing | Label encoding, train/test split strategy |
| 5. Model Training & Evaluation | Comparison of 3 classifiers (accuracy + ROC-AUC) |
| 6. Feature Importances | Ranked importance table for all 12 features |
| 7. System Architecture | Three-layer architecture (ML → API → Frontend) |
| 8. Key Findings & Recommendations | 5 findings + 5 business action items |
| 9. How to Run the Project | Step-by-step run commands table |

### Report Highlights

- **Best model:** Gradient Boosting Classifier — **99.1% accuracy**, **ROC-AUC 0.9986**
- **Top predictor:** `avg_watch_time_per_day` drives **65.1%** of model decisions
- **Highest-risk segment:** Basic plan subscribers churn at **61.8%** vs 43.7% for Premium
- **Payment risk:** Crypto and Gift Card users churn **~16% more** than card-based users
- **Top 4 features** account for **91.5%** of total model importance

Open the report:
```bash
# Windows
start reports.docx

# macOS
open reports.docx

# Linux
xdg-open reports.docx
```

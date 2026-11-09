"""Streamlit app: dashboard, EDA, live churn prediction, model performance.

Run from the repo root:  streamlit run src/frontend.py
Deploy: Streamlit Community Cloud -> main file path = src/frontend.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import DATA_PATH, IMAGES_DIR, NUMERIC, TARGET
from src.data import load_data as load_clean_data
from src.predict import load_meta, load_model, predict_one

st.set_page_config(page_title="Netflix Churn Analytics", page_icon="🎬", layout="wide")
RED = "#E50914"


@st.cache_data
def load_data():
    return load_clean_data()


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_meta():
    return load_meta()


def churn_by(df, col):
    s = df.groupby(col)[TARGET].mean().mul(100).round(1).sort_values(ascending=False)
    return s.reset_index().rename(columns={TARGET: "churn_rate_pct"})


def bar(df, col, title):
    d = churn_by(df, col)
    fig = px.bar(d, x=col, y="churn_rate_pct", title=title, color_discrete_sequence=[RED], text="churn_rate_pct")
    fig.update_layout(yaxis_title="Churn %", xaxis_title="")
    return fig


def page_dashboard(df):
    st.title("🎬 Netflix Churn Dashboard")
    lost = df.loc[df[TARGET] == 1, "monthly_fee"].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churn rate", f"{df[TARGET].mean()*100:.1f}%")
    c3.metric("Avg monthly fee", f"${df['monthly_fee'].mean():.2f}")
    c4.metric("Monthly revenue lost", f"${lost:,.0f}")
    a, b = st.columns(2)
    a.plotly_chart(bar(df, "subscription_type", "Churn % by Plan"), width="stretch")
    b.plotly_chart(bar(df, "region", "Churn % by Region"), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(bar(df, "payment_method", "Churn % by Payment Method"), width="stretch")
    q = pd.qcut(df["avg_watch_time_per_day"].rank(method="first"), 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    d = df.groupby(q, observed=True)[TARGET].mean().mul(100).round(1).reset_index()
    d.columns = ["engagement", "churn_rate_pct"]
    fig = px.bar(d, x="engagement", y="churn_rate_pct", title="Churn % by Engagement Quartile",
                 color_discrete_sequence=[RED], text="churn_rate_pct")
    b.plotly_chart(fig, width="stretch")


def page_eda(df):
    st.title("📊 EDA & Insights")
    col = st.selectbox("Segment churn rate by", ["gender", "device", "favorite_genre", "number_of_profiles",
                                                 "subscription_type", "region", "payment_method"])
    st.plotly_chart(bar(df, col, f"Churn % by {col}"), width="stretch")
    num = st.selectbox("Compare a numeric feature across churned vs retained", NUMERIC)
    fig = px.box(df.assign(status=df[TARGET].map({0: "Retained", 1: "Churned"})), x="status", y=num,
                 color="status", color_discrete_map={"Retained": "#564D4D", "Churned": RED})
    st.plotly_chart(fig, width="stretch")
    st.subheader("Correlation heatmap")
    corr = df[NUMERIC + [TARGET]].corr().round(2)
    st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1),
                    width="stretch")


def page_predict(df):
    st.title("🔮 Predict Churn")
    try:
        model, meta = get_model(), get_meta()
    except FileNotFoundError as e:
        st.error(str(e)); return
    cats, rng = meta["categories"], meta["numeric_ranges"]
    with st.form("customer"):
        c1, c2, c3 = st.columns(3)
        rec = {
            "age": c1.number_input("Age", 10, 110, 35),
            "gender": c1.selectbox("Gender", cats["gender"]),
            "subscription_type": c1.selectbox("Subscription", cats["subscription_type"]),
            "region": c2.selectbox("Region", cats["region"]),
            "device": c2.selectbox("Device", cats["device"]),
            "payment_method": c2.selectbox("Payment method", cats["payment_method"]),
            "favorite_genre": c2.selectbox("Favorite genre", cats["favorite_genre"]),
            "watch_hours": c3.number_input("Total watch hours", 0.0, float(rng["watch_hours"][1]) * 2, 20.0),
            "avg_watch_time_per_day": c3.number_input("Avg watch hrs/day", 0.0, 24.0, 1.5),
            "last_login_days": c3.number_input("Days since last login", 0, 365, 5),
            "monthly_fee": c3.number_input("Monthly fee", 0.0, 100.0, 13.99),
            "number_of_profiles": c3.number_input("Profiles", 1, 10, 2),
        }
        go_btn = st.form_submit_button("Predict")
    if not go_btn:
        return
    res = predict_one(model, rec)
    p = res["churn_probability"]
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%"},
        title={"text": f"Churn probability - {res['risk_level']} risk"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": RED},
               "steps": [{"range": [0, 40], "color": "#d9f2d9"}, {"range": [40, 70], "color": "#fff3cd"},
                         {"range": [70, 100], "color": "#f8d7da"}]}))
    st.plotly_chart(gauge, width="stretch")

    signals = []
    q1 = df["avg_watch_time_per_day"].quantile(0.25)
    if rec["last_login_days"] >= 15:
        signals.append("Inactive for 15+ days")
    if rec["avg_watch_time_per_day"] <= q1:
        signals.append(f"Low engagement (bottom 25% of users, <= {q1:.2f} hrs/day)")
    plan_rates = df.groupby("subscription_type")[TARGET].mean()
    if rec["subscription_type"] == plan_rates.idxmax():
        signals.append(f"{rec['subscription_type']} is the highest-churn plan in this data")
    pay_rates = df.groupby("payment_method")[TARGET].mean()
    if pay_rates[rec["payment_method"]] > df[TARGET].mean() + 0.05:
        signals.append(f"{rec['payment_method']} payers churn above average")
    st.subheader("Risk signals")
    for s in signals or ["No major risk signals detected"]:
        st.write("•", s)


def page_model():
    st.title("📈 Model Performance")
    try:
        meta = get_meta()
    except FileNotFoundError as e:
        st.error(str(e)); return
    st.write(f"**Best model:** {meta['best_model']}  |  Train rows: {meta['train_rows']:,}  |  "
             f"Test rows: {meta['test_rows']:,}")
    st.dataframe(pd.DataFrame(meta["models"]).T.round(4), width="stretch")
    lk = meta["leakage_check"]
    st.subheader("Leakage check")
    st.write(f"Top feature: **{lk['top_feature']}**. ROC-AUC with it: **{lk['auc_with_top_feature']:.4f}**, "
             f"without it: **{lk['auc_without_top_feature']:.4f}**.")
    if lk.get("auc_without_engagement_features") is not None:
        st.write(f"Without **all** engagement features ({', '.join(lk['engagement_features'])}): "
                 f"**{lk['auc_without_engagement_features']:.4f}**. "
                 "The engagement features overlap, so removing one barely changes the score; "
                 "removing the whole group shows how much of the signal comes from engagement.")
    fi = pd.Series(meta["feature_importance"]).sort_values()
    st.plotly_chart(px.bar(fi, orientation="h", title="Permutation importance (drop in ROC-AUC)",
                           color_discrete_sequence=[RED]).update_layout(showlegend=False),
                    width="stretch")
    for img in ["06_confusion_matrix.png", "07_roc_curves.png"]:
        if (IMAGES_DIR / img).exists():
            st.image(str(IMAGES_DIR / img), width=450)


if not DATA_PATH.exists():
    st.error(f"Dataset not found at {DATA_PATH}. Put netflix_customer_churn.csv in the data/ folder.")
    st.stop()

data = load_data()
page = st.sidebar.radio("Navigate", ["Dashboard", "EDA & Insights", "Predict Churn", "Model Performance"])
if page == "Dashboard":
    page_dashboard(data)
elif page == "EDA & Insights":
    page_eda(data)
elif page == "Predict Churn":
    page_predict(data)
else:
    page_model()

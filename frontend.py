"""
frontend.py — Streamlit frontend for Netflix Churn Prediction.
Run: streamlit run frontend.py
"""

import os
import pickle
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Churn Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL   = "http://localhost:8000"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "netflix_customer_churn.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #141414; }
  [data-testid="stSidebar"]          { background: #1c1c1c; }
  h1, h2, h3, h4                     { color: #E50914; }
  .metric-card {
      background: #1c1c1c; border-radius: 10px; padding: 20px;
      border-left: 4px solid #E50914; margin: 8px 0;
  }
  .churn-high   { background:#4a1010; border-left:4px solid #E50914; border-radius:8px; padding:15px; }
  .churn-medium { background:#4a3010; border-left:4px solid #f59e0b; border-radius:8px; padding:15px; }
  .churn-low    { background:#0f2e1a; border-left:4px solid #22c55e; border-radius:8px; padding:15px; }
  .stMetric     { background:#1c1c1c; border-radius:8px; padding:10px; }
</style>
""", unsafe_allow_html=True)

NETFLIX_RED  = "#E50914"
COLORS       = [NETFLIX_RED, "#831010", "#f59e0b", "#22c55e", "#3b82f6", "#a855f7", "#ec4899", "#14b8a6"]

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_meta():
    try:
        with open(os.path.join(MODEL_DIR, "meta.pkl"), "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def api_predict(payload: dict):
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "⚠️ Backend not running. Start it with: `uvicorn app:app --reload`"
    except Exception as e:
        return None, str(e)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=150)
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "📊 EDA & Insights", "🔮 Predict Churn", "📈 Model Performance"],
        index=0,
    )
    st.markdown("---")
    st.caption("Netflix Customer Churn Predictor v1.0")


df = load_data()
meta = load_meta()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("🎬 Netflix Customer Churn Dashboard")
    st.markdown("An end-to-end ML platform for churn prediction and customer intelligence.")
    st.markdown("---")

    # KPI cards
    total     = len(df)
    churned   = df["churned"].sum()
    retained  = total - churned
    churn_rate = churned / total * 100
    avg_fee   = df["monthly_fee"].mean()
    avg_watch = df["watch_hours"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Churned", f"{int(churned):,}", delta=f"{churn_rate:.1f}%", delta_color="inverse")
    c3.metric("Retained", f"{int(retained):,}")
    c4.metric("Avg Monthly Fee", f"${avg_fee:.2f}")
    c5.metric("Avg Watch Hours", f"{avg_watch:.1f} h")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Rate Overview")
        fig = go.Figure(go.Pie(
            labels=["Churned", "Retained"],
            values=[churned, retained],
            hole=0.55,
            marker_colors=[NETFLIX_RED, "#2d2d2d"],
            textinfo="label+percent",
        ))
        fig.update_layout(
            paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
            font_color="white", showlegend=True, height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn by Subscription Type")
        ct = df.groupby("subscription_type")["churned"].agg(["sum", "count"]).reset_index()
        ct.columns = ["Subscription", "Churned", "Total"]
        ct["Rate"] = ct["Churned"] / ct["Total"] * 100
        fig = px.bar(ct, x="Subscription", y="Rate", color="Subscription",
                     color_discrete_sequence=COLORS,
                     labels={"Rate": "Churn Rate (%)"}, text_auto=".1f")
        fig.update_layout(paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
                          font_color="white", showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Churn by Region")
        rg = df.groupby("region")["churned"].mean().reset_index()
        rg.columns = ["Region", "Churn Rate"]
        rg["Churn Rate"] = rg["Churn Rate"] * 100
        fig = px.bar(rg.sort_values("Churn Rate", ascending=True),
                     x="Churn Rate", y="Region", orientation="h",
                     color="Churn Rate", color_continuous_scale=["#2d2d2d", NETFLIX_RED])
        fig.update_layout(paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
                          font_color="white", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Monthly Fee Distribution")
        fig = px.histogram(df, x="monthly_fee", color="churned",
                           barmode="overlay", nbins=30,
                           color_discrete_map={0: "#22c55e", 1: NETFLIX_RED},
                           labels={"churned": "Churned"})
        fig.update_layout(paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
                          font_color="white", height=300)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA & INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Insights":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["👥 Demographics", "📺 Viewing Behaviour", "💳 Financials", "🔗 Correlations"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Age Distribution by Churn")
            fig = px.histogram(df, x="age", color="churned", nbins=30, barmode="overlay",
                               color_discrete_map={0: "#22c55e", 1: NETFLIX_RED},
                               labels={"churned": "Churned"})
            fig.update_layout(paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
                               font_color="white", height=320)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Gender vs Churn")
            gdf = df.groupby("gender")["churned"].mean().reset_index()
            gdf.columns = ["Gender", "Churn Rate"]
            gdf["Churn Rate"] = gdf["Churn Rate"] * 100
            fig = px.bar(gdf, x="Gender", y="Churn Rate", color="Gender",
                         color_discrete_sequence=COLORS, text_auto=".1f")
            fig.update_layout(paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
                               font_color="white", showlegend=False, height=320)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Churn by Device")
            dv = df.groupby("device")["churned"].mean().reset_index()
            dv.columns = ["Device", "Churn Rate"]
            dv["Churn Rate"] = dv["Churn Rate"] * 100
            fig = px.pie(dv, values="Churn Rate", names="Device",
                         color_discrete_sequence=COLORS, hole=0.4)
            fig.update_layout(paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
                               font_color="white", height=320)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.subheader("Favorite Genre vs Churn")
            fg = df.groupby("favorite_genre")["churned"].mean().reset_index()
            fg.columns = ["Genre", "Churn Rate"]
            fg["Churn Rate"] = fg["Churn Rate"] * 100
            fig = px.bar(fg.sort_values("Churn Rate", ascending=False),
                         x="Genre", y="Churn Rate", color="Genre",
                         color_discrete_sequence=COLORS, text_auto=".1f")
            fig.update_layout(paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
                               font_color="white", showlegend=False, height=320)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Watch Hours vs Churn")
            fig = px.box(df, x="churned", y="watch_hours",
                         color="churned",
                         color_discrete_map={0: "#22c55e", 1: NETFLIX_RED},
                         labels={"churned": "Churned", "watch_hours": "Watch Hours"})
            fig.update_layout(paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
                               font_color="white", height=320, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Last Login Days vs Churn")
            fig = px.box(df, x="churned", y="last_login_days",
                         color="churned",
                         color_discrete_map={0: "#22c55e", 1: NETFLIX_RED},
                         labels={"churned": "Churned", "last_login_days": "Days Since Last Login"})
            fig.update_layout(paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
                               font_color="white", height=320, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Avg Watch Time per Day vs Last Login (scatter by churn)")
        fig = px.scatter(df.sample(1000, random_state=42), x="avg_watch_time_per_day",
                         y="last_login_days", color="churned",
                         color_discrete_map={0: "#22c55e", 1: NETFLIX_RED},
                         opacity=0.6, size_max=6,
                         labels={"churned": "Churned",
                                 "avg_watch_time_per_day": "Avg Watch Time/Day (h)",
                                 "last_login_days": "Days Since Last Login"})
        fig.update_layout(paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
                           font_color="white", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Monthly Fee by Subscription")
            fig = px.box(df, x="subscription_type", y="monthly_fee",
                         color="subscription_type", color_discrete_sequence=COLORS)
            fig.update_layout(paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
                               font_color="white", height=320, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Payment Method vs Churn Rate")
            pm = df.groupby("payment_method")["churned"].mean().reset_index()
            pm.columns = ["Payment", "Churn Rate"]
            pm["Churn Rate"] = pm["Churn Rate"] * 100
            fig = px.bar(pm.sort_values("Churn Rate", ascending=False),
                         x="Payment", y="Churn Rate", color="Payment",
                         color_discrete_sequence=COLORS, text_auto=".1f")
            fig.update_layout(paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
                               font_color="white", showlegend=False, height=320)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Feature Correlation Heatmap")
        num_df = df.copy()
        for col in ["gender", "subscription_type", "region", "device",
                    "payment_method", "favorite_genre"]:
            num_df[col] = pd.factorize(num_df[col])[0]
        corr = num_df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        aspect="auto", zmin=-1, zmax=1)
        fig.update_layout(paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
                           font_color="white", height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Raw Data Sample")
        st.dataframe(df.head(100), use_container_width=True, height=300)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICT CHURN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Churn":
    st.title("🔮 Predict Customer Churn")
    st.markdown("Fill in customer details to get a real-time churn prediction from the ML model.")
    st.markdown("---")

    col_form, col_result = st.columns([1.2, 1])

    with col_form:
        with st.form("predict_form"):
            st.subheader("Customer Profile")
            c1, c2 = st.columns(2)
            age             = c1.number_input("Age",              min_value=1,  max_value=120, value=35)
            gender          = c2.selectbox("Gender",              ["Male", "Female", "Other"])
            subscription    = c1.selectbox("Subscription Type",   ["Basic", "Standard", "Premium"])
            region          = c2.selectbox("Region",              ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"])
            device          = c1.selectbox("Device",              ["Mobile", "TV", "Tablet", "Laptop"])
            favorite_genre  = c2.selectbox("Favorite Genre",      ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Documentary", "Thriller"])
            payment_method  = c1.selectbox("Payment Method",      ["Credit Card", "Debit Card", "PayPal", "Gift Card", "Crypto"])
            num_profiles    = c2.number_input("Number of Profiles", min_value=1, max_value=10, value=2)

            st.subheader("Usage & Financials")
            c3, c4 = st.columns(2)
            watch_hours     = c3.number_input("Total Watch Hours",        min_value=0.0, max_value=500.0, value=10.0, step=0.5)
            last_login      = c4.number_input("Days Since Last Login",    min_value=0,   max_value=365,   value=15)
            monthly_fee     = c3.number_input("Monthly Fee ($)",          min_value=0.0, max_value=50.0,  value=13.99, step=0.01)
            avg_watch_time  = c4.number_input("Avg Watch Time/Day (h)",   min_value=0.0, max_value=24.0,  value=1.2,   step=0.1)

            submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    with col_result:
        st.subheader("Prediction Result")
        if submitted:
            payload = {
                "age":                    age,
                "gender":                 gender,
                "subscription_type":      subscription,
                "watch_hours":            watch_hours,
                "last_login_days":        last_login,
                "region":                 region,
                "device":                 device,
                "monthly_fee":            monthly_fee,
                "payment_method":         payment_method,
                "number_of_profiles":     num_profiles,
                "avg_watch_time_per_day": avg_watch_time,
                "favorite_genre":         favorite_genre,
            }
            with st.spinner("Calling prediction API…"):
                result, err = api_predict(payload)

            if err:
                st.error(err)
            else:
                prob       = result["churn_probability"]
                risk       = result["risk_level"]
                churned    = result["churned"]
                model_used = result["model_used"]

                color_map  = {"High": "#E50914", "Medium": "#f59e0b", "Low": "#22c55e"}
                class_map  = {"High": "churn-high", "Medium": "churn-medium", "Low": "churn-low"}
                icon_map   = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

                st.markdown(f"""
                <div class="{class_map[risk]}">
                    <h2 style="color:{color_map[risk]};margin:0">{icon_map[risk]} {risk} Risk</h2>
                    <p style="color:white;font-size:18px;margin:8px 0">
                        Churn Probability: <strong>{prob*100:.1f}%</strong>
                    </p>
                    <p style="color:#aaa;margin:0">
                        Prediction: {"⚠️ Will Churn" if churned else "✅ Will Stay"}<br>
                        Model: {model_used}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"color": "white"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "white"},
                        "bar":  {"color": color_map[risk]},
                        "bgcolor": "#1c1c1c",
                        "steps": [
                            {"range": [0,   40],  "color": "#0f2e1a"},
                            {"range": [40,  70],  "color": "#4a3010"},
                            {"range": [70, 100],  "color": "#4a1010"},
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 3},
                            "thickness": 0.75,
                            "value": prob * 100,
                        },
                    },
                    title={"text": "Churn Probability", "font": {"color": "white"}},
                ))
                fig.update_layout(
                    paper_bgcolor="#141414", font_color="white", height=260, margin=dict(t=60, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Risk factors callout
                st.markdown("**Key Risk Signals:**")
                signals = []
                if last_login > 30:
                    signals.append(f"🔴 Last login was {last_login} days ago (inactivity risk)")
                if watch_hours < 5:
                    signals.append(f"🔴 Low total watch hours: {watch_hours} h")
                if avg_watch_time < 0.5:
                    signals.append(f"🟡 Very low avg watch time/day: {avg_watch_time} h")
                if subscription == "Basic":
                    signals.append("🟡 Basic plan — higher churn tendency")
                if not signals:
                    signals.append("🟢 No strong risk signals detected")
                for s in signals:
                    st.markdown(s)
        else:
            st.info("Fill in the form and click **Predict Churn** to see results.")
            st.markdown("""
            **How it works:**
            1. Enter customer profile details on the left
            2. The form calls the FastAPI backend
            3. The trained ML model returns a churn probability
            4. Risk level is assigned: Low < 40% · Medium 40–70% · High ≥ 70%
            """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.markdown("---")

    if meta is None:
        st.warning("Run `python train.py` first to generate model metrics.")
        st.stop()

    st.success(f"✅ Best Model: **{meta['best_model']}**  |  ROC-AUC: **{meta['best_auc']:.4f}**")

    # Model comparison bar chart
    results = meta["results"]
    models  = list(results.keys())
    acc     = [results[m]["accuracy"] for m in models]
    auc     = [results[m]["roc_auc"]  for m in models]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Accuracy Comparison")
        fig = go.Figure(go.Bar(
            x=models, y=acc, text=[f"{v:.4f}" for v in acc],
            marker_color=[NETFLIX_RED if m == meta["best_model"] else "#2d2d2d" for m in models],
            textposition="outside",
        ))
        fig.update_layout(paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
                           font_color="white", yaxis_range=[0, 1.05], height=320,
                           yaxis_title="Accuracy")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("ROC-AUC Comparison")
        fig = go.Figure(go.Bar(
            x=models, y=auc, text=[f"{v:.4f}" for v in auc],
            marker_color=[NETFLIX_RED if m == meta["best_model"] else "#2d2d2d" for m in models],
            textposition="outside",
        ))
        fig.update_layout(paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
                           font_color="white", yaxis_range=[0, 1.05], height=320,
                           yaxis_title="ROC-AUC")
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    try:
        with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
            model_obj = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "feature_columns.pkl"), "rb") as f:
            feat_cols = pickle.load(f)

        if hasattr(model_obj, "feature_importances_"):
            fi = pd.Series(model_obj.feature_importances_, index=feat_cols)
            fi = fi.sort_values(ascending=True)
            st.subheader(f"Feature Importances — {meta['best_model']}")
            fig = go.Figure(go.Bar(
                x=fi.values, y=fi.index, orientation="h",
                marker=dict(
                    color=fi.values,
                    colorscale=[[0, "#2d2d2d"], [1, NETFLIX_RED]],
                ),
                text=[f"{v:.4f}" for v in fi.values],
                textposition="outside",
            ))
            fig.update_layout(paper_bgcolor="#1c1c1c", plot_bgcolor="#1c1c1c",
                               font_color="white", height=420, xaxis_title="Importance")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importances not available for this model type.")
    except Exception as e:
        st.warning(f"Could not load feature importances: {e}")

    # Encoder classes table
    st.subheader("Categorical Encodings")
    for col, classes in meta["encoder_classes"].items():
        with st.expander(f"📌 {col}"):
            st.write(", ".join(classes))

"""Generate the EDA / dashboard images used in the README.

Run from the repo root:  python -m src.eda
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import IMAGES_DIR, NUMERIC, TARGET
from src.data import load_data

RED, GREY = "#E50914", "#564D4D"
sns.set_theme(style="whitegrid")


def churn_rate(df, col):
    return df.groupby(col)[TARGET].mean().mul(100).sort_values(ascending=False)


def save(fig, name):
    IMAGES_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / name, dpi=150)
    plt.close(fig)
    print("saved", name)


def kpi_dashboard(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]; ax.axis("off")
    lost = df.loc[df[TARGET] == 1, "monthly_fee"].sum()
    kpis = [("Customers", f"{len(df):,}"), ("Churn rate", f"{df[TARGET].mean()*100:.1f}%"),
            ("Avg monthly fee", f"${df['monthly_fee'].mean():.2f}"), ("Monthly revenue lost", f"${lost:,.0f}")]
    for i, (k, v) in enumerate(kpis):
        ax.text(0.02 + (i % 2) * 0.5, 0.7 - (i // 2) * 0.5, v, fontsize=24, weight="bold", color=RED)
        ax.text(0.02 + (i % 2) * 0.5, 0.58 - (i // 2) * 0.5, k, fontsize=11, color=GREY)
    ax.set_title("Key Metrics", loc="left", weight="bold")
    for ax, col, title in [(axes[0, 1], "subscription_type", "Churn % by Plan"),
                           (axes[1, 0], "region", "Churn % by Region"),
                           (axes[1, 1], "payment_method", "Churn % by Payment Method")]:
        s = churn_rate(df, col)
        sns.barplot(x=s.index, y=s.values, ax=ax, color=RED)
        ax.set_title(title, weight="bold"); ax.set_xlabel(""); ax.set_ylabel("Churn %")
        ax.tick_params(axis="x", rotation=30)
    save(fig, "01_kpi_dashboard.png")


def demographics(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(data=df, x="age", hue=TARGET, multiple="stack", bins=20, ax=axes[0, 0],
                 palette=[GREY, RED]); axes[0, 0].set_title("Age Distribution by Churn", weight="bold")
    for ax, col in [(axes[0, 1], "gender"), (axes[1, 0], "device"), (axes[1, 1], "favorite_genre")]:
        s = churn_rate(df, col)
        sns.barplot(x=s.index, y=s.values, ax=ax, color=RED)
        ax.set_title(f"Churn % by {col.replace('_', ' ').title()}", weight="bold")
        ax.set_xlabel(""); ax.set_ylabel("Churn %"); ax.tick_params(axis="x", rotation=30)
    save(fig, "02_eda_demographics.png")


def behaviour(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in [(axes[0, 0], "watch_hours"), (axes[0, 1], "last_login_days"),
                    (axes[1, 0], "avg_watch_time_per_day")]:
        sns.boxplot(data=df, x=TARGET, y=col, ax=ax, palette=[GREY, RED], hue=TARGET, legend=False)
        ax.set_title(f"{col.replace('_', ' ').title()} vs Churn", weight="bold")
        ax.set_ylim(0, df[col].quantile(0.99))  # zoom past extreme outliers
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Retained", "Churned"]); ax.set_xlabel("")
    s = churn_rate(df, "number_of_profiles").sort_index()
    sns.barplot(x=s.index, y=s.values, ax=axes[1, 1], color=RED)
    axes[1, 1].set_title("Churn % by Number of Profiles", weight="bold")
    axes[1, 1].set_xlabel("Profiles"); axes[1, 1].set_ylabel("Churn %")
    save(fig, "03_eda_behaviour_financials.png")


def correlation(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[NUMERIC + [TARGET]].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlation Heatmap", weight="bold")
    save(fig, "04_correlation_heatmap.png")


if __name__ == "__main__":
    data = load_data()
    kpi_dashboard(data); demographics(data); behaviour(data); correlation(data)

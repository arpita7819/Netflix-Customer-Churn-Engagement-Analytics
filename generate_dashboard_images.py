"""
generate_dashboard_images.py
Generates all dashboard diagram PNGs for insertion into reports.docx.
Run: python generate_dashboard_images.py
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(__file__)
DATA   = os.path.join(BASE, "data", "netflix_customer_churn.csv")
MDIR   = os.path.join(BASE, "models")
IMGDIR = os.path.join(BASE, "dashboard_images")
os.makedirs(IMGDIR, exist_ok=True)

df    = pd.read_csv(DATA)
meta  = pickle.load(open(os.path.join(MDIR, "meta.pkl"),    "rb"))
model = pickle.load(open(os.path.join(MDIR, "model.pkl"),   "rb"))
feats = pickle.load(open(os.path.join(MDIR, "feature_columns.pkl"), "rb"))

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#141414"
SURFACE = "#1c1c1c"
RED     = "#E50914"
GREEN   = "#22c55e"
AMBER   = "#f59e0b"
BLUE    = "#3b82f6"
MUTED   = "#888888"
WHITE   = "#f0f0f0"
COLORS  = [RED, "#831010", AMBER, GREEN, BLUE, "#a855f7", "#ec4899", "#14b8a6"]

def dark_fig(w=14, h=7):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    return fig

def save(name):
    path = os.path.join(IMGDIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {name}")
    return path

# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE 1 — KPI Dashboard Overview
# ═══════════════════════════════════════════════════════════════════════════════
fig = dark_fig(16, 9)
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4,
                        left=0.04, right=0.97, top=0.92, bottom=0.06)

fig.text(0.5, 0.96, "Page 1 — KPI Dashboard", color=WHITE,
         fontsize=18, fontweight="bold", ha="center", va="top")
fig.text(0.5, 0.925, "Netflix Customer Churn Overview", color=MUTED,
         fontsize=11, ha="center", va="top")

# KPI cards
kpis = [
    ("Total Customers", "5,000",      WHITE),
    ("Churned",         "2,515",      RED),
    ("Retained",        "2,485",      GREEN),
    ("Avg Monthly Fee", "$13.68",     BLUE),
    ("Avg Watch Hours", "11.65 h",    AMBER),
]
for i, (label, val, color) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i if i < 4 else 3])
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.9,
                 boxstyle="round,pad=0.03", facecolor=SURFACE,
                 edgecolor=color, linewidth=2))
    ax.text(0.5, 0.72, val,  color=color,  fontsize=20, fontweight="bold",
            ha="center", va="center")
    ax.text(0.5, 0.28, label, color=MUTED, fontsize=9,
            ha="center", va="center", wrap=True)

# Donut chart
ax1 = fig.add_subplot(gs[1:, 0])
ax1.set_facecolor(BG)
sizes  = [2515, 2485]
colors = [RED, "#2d2d2d"]
wedges, texts, autotexts = ax1.pie(
    sizes, labels=["Churned\n50.3%", "Retained\n49.7%"],
    colors=colors, autopct="", startangle=90,
    wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2),
    textprops={"color": WHITE, "fontsize": 9}
)
ax1.text(0, 0, "50.3%\nChurn\nRate", color=RED, fontsize=11,
         fontweight="bold", ha="center", va="center")
ax1.set_title("Churn Rate Overview", color=WHITE, fontsize=10, pad=8)

# Churn by subscription bar
ax2 = fig.add_subplot(gs[1, 1:3])
ax2.set_facecolor(SURFACE)
subs  = ["Basic", "Standard", "Premium"]
rates = [61.83, 45.44, 43.71]
bars  = ax2.bar(subs, rates, color=[RED, AMBER, GREEN], width=0.5, edgecolor=BG)
for b, r in zip(bars, rates):
    ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8,
             f"{r:.1f}%", ha="center", color=WHITE, fontsize=9, fontweight="bold")
ax2.set_facecolor(SURFACE); ax2.set_ylim(0, 80)
ax2.tick_params(colors=WHITE); ax2.set_ylabel("Churn Rate (%)", color=MUTED, fontsize=9)
for spine in ax2.spines.values(): spine.set_color("#333333")
ax2.set_title("Churn by Subscription Type", color=WHITE, fontsize=10, pad=6)

# Churn by region horizontal bar
ax3 = fig.add_subplot(gs[2, 1:3])
ax3.set_facecolor(SURFACE)
regions = ["Africa", "Asia", "Europe", "N. America", "Oceania", "S. America"]
rrates  = [48.32, 50.65, 51.67, 49.47, 50.07, 51.43]
norm    = plt.Normalize(min(rrates), max(rrates))
rcols   = plt.cm.RdYlGn_r(norm(rrates))
bars3   = ax3.barh(regions, rrates, color=rcols, edgecolor=BG)
for b, r in zip(bars3, rrates):
    ax3.text(r + 0.2, b.get_y() + b.get_height()/2,
             f"{r:.1f}%", va="center", color=WHITE, fontsize=8)
ax3.set_facecolor(SURFACE); ax3.set_xlim(44, 56)
ax3.tick_params(colors=WHITE, labelsize=8)
ax3.set_xlabel("Churn Rate (%)", color=MUTED, fontsize=8)
for spine in ax3.spines.values(): spine.set_color("#333333")
ax3.set_title("Churn Rate by Region", color=WHITE, fontsize=10, pad=6)

# Monthly fee histogram
ax4 = fig.add_subplot(gs[1:, 3])
ax4.set_facecolor(SURFACE)
churned_fees  = df[df["churned"] == 1]["monthly_fee"]
retained_fees = df[df["churned"] == 0]["monthly_fee"]
ax4.hist(retained_fees, bins=20, alpha=0.6, color=GREEN, label="Retained", edgecolor=BG)
ax4.hist(churned_fees,  bins=20, alpha=0.6, color=RED,   label="Churned",  edgecolor=BG)
ax4.set_facecolor(SURFACE)
ax4.tick_params(colors=WHITE, labelsize=8)
ax4.set_xlabel("Monthly Fee ($)", color=MUTED, fontsize=8)
ax4.set_ylabel("Count", color=MUTED, fontsize=8)
ax4.legend(facecolor=SURFACE, labelcolor=WHITE, fontsize=8)
for spine in ax4.spines.values(): spine.set_color("#333333")
ax4.set_title("Monthly Fee Distribution", color=WHITE, fontsize=10, pad=6)

save("01_kpi_dashboard.png")

# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE 2 — EDA Demographics Tab
# ═══════════════════════════════════════════════════════════════════════════════
fig = dark_fig(16, 8)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.35,
                        left=0.06, right=0.97, top=0.91, bottom=0.08)
fig.text(0.5, 0.96, "Page 2 — EDA: Demographics Tab", color=WHITE,
         fontsize=16, fontweight="bold", ha="center")

# Age histogram
ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(SURFACE)
ax.hist(df[df["churned"]==0]["age"], bins=25, alpha=0.6, color=GREEN, label="Retained", edgecolor=BG)
ax.hist(df[df["churned"]==1]["age"], bins=25, alpha=0.6, color=RED,   label="Churned",  edgecolor=BG)
ax.set_facecolor(SURFACE); ax.tick_params(colors=WHITE, labelsize=8)
ax.set_xlabel("Age", color=MUTED, fontsize=9); ax.set_ylabel("Count", color=MUTED, fontsize=9)
ax.legend(facecolor=SURFACE, labelcolor=WHITE, fontsize=8)
for sp in ax.spines.values(): sp.set_color("#333333")
ax.set_title("Age Distribution by Churn", color=WHITE, fontsize=11, pad=6)

# Gender bar
ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor(SURFACE)
gdf = df.groupby("gender")["churned"].mean().reset_index()
bars = ax2.bar(gdf["gender"], gdf["churned"]*100, color=COLORS[:3], edgecolor=BG, width=0.5)
for b in bars:
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.4,
             f"{b.get_height():.1f}%", ha="center", color=WHITE, fontsize=9)
ax2.set_facecolor(SURFACE); ax2.set_ylim(0, 65)
ax2.tick_params(colors=WHITE, labelsize=9); ax2.set_ylabel("Churn Rate (%)", color=MUTED, fontsize=9)
for sp in ax2.spines.values(): sp.set_color("#333333")
ax2.set_title("Churn Rate by Gender", color=WHITE, fontsize=11, pad=6)

# Device donut
ax3 = fig.add_subplot(gs[1, 0]); ax3.set_facecolor(BG)
ddf = df.groupby("device")["churned"].mean()
ax3.pie(ddf.values*100, labels=ddf.index, colors=COLORS[:len(ddf)],
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(width=0.55, edgecolor=BG),
        textprops={"color": WHITE, "fontsize": 8},
        pctdistance=0.75)
ax3.set_title("Churn Rate by Device", color=WHITE, fontsize=11, pad=6)

# Genre bar
ax4 = fig.add_subplot(gs[1, 1]); ax4.set_facecolor(SURFACE)
fdf = df.groupby("favorite_genre")["churned"].mean().sort_values(ascending=False)
bars4 = ax4.bar(range(len(fdf)), fdf.values*100, color=COLORS[:len(fdf)], edgecolor=BG)
ax4.set_xticks(range(len(fdf))); ax4.set_xticklabels(fdf.index, rotation=35, ha="right", fontsize=7)
ax4.set_facecolor(SURFACE); ax4.set_ylim(0, 65)
ax4.tick_params(colors=WHITE, labelsize=7); ax4.set_ylabel("Churn Rate (%)", color=MUTED, fontsize=9)
for sp in ax4.spines.values(): sp.set_color("#333333")
ax4.set_title("Churn Rate by Favorite Genre", color=WHITE, fontsize=11, pad=6)

save("02_eda_demographics.png")

# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE 3 — EDA Viewing Behaviour + Financials
# ═══════════════════════════════════════════════════════════════════════════════
fig = dark_fig(16, 8)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38,
                        left=0.06, right=0.97, top=0.91, bottom=0.08)
fig.text(0.5, 0.96, "Page 2 — EDA: Viewing Behaviour & Financials", color=WHITE,
         fontsize=16, fontweight="bold", ha="center")

# Watch hours box
ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(SURFACE)
d0 = df[df["churned"]==0]["watch_hours"]; d1 = df[df["churned"]==1]["watch_hours"]
bp = ax.boxplot([d0, d1], labels=["Retained", "Churned"], patch_artist=True,
                medianprops=dict(color=WHITE, linewidth=2))
bp["boxes"][0].set_facecolor(GREEN); bp["boxes"][1].set_facecolor(RED)
for w in bp["whiskers"]+bp["caps"]: w.set_color("#888888")
for fl in bp["fliers"]: fl.set_markerfacecolor("#555555"); fl.set_markersize(3)
ax.set_facecolor(SURFACE); ax.tick_params(colors=WHITE, labelsize=9)
ax.set_ylabel("Watch Hours", color=MUTED, fontsize=9)
for sp in ax.spines.values(): sp.set_color("#333333")
ax.set_title("Watch Hours vs Churn", color=WHITE, fontsize=11, pad=6)

# Last login box
ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor(SURFACE)
d0b = df[df["churned"]==0]["last_login_days"]; d1b = df[df["churned"]==1]["last_login_days"]
bp2 = ax2.boxplot([d0b, d1b], labels=["Retained", "Churned"], patch_artist=True,
                  medianprops=dict(color=WHITE, linewidth=2))
bp2["boxes"][0].set_facecolor(GREEN); bp2["boxes"][1].set_facecolor(RED)
for w in bp2["whiskers"]+bp2["caps"]: w.set_color("#888888")
for fl in bp2["fliers"]: fl.set_markerfacecolor("#555555"); fl.set_markersize(3)
ax2.set_facecolor(SURFACE); ax2.tick_params(colors=WHITE, labelsize=9)
ax2.set_ylabel("Days Since Last Login", color=MUTED, fontsize=9)
for sp in ax2.spines.values(): sp.set_color("#333333")
ax2.set_title("Last Login Days vs Churn", color=WHITE, fontsize=11, pad=6)

# Scatter
ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor(SURFACE)
samp = df.sample(800, random_state=42)
ax3.scatter(samp[samp["churned"]==0]["avg_watch_time_per_day"],
            samp[samp["churned"]==0]["last_login_days"],
            alpha=0.4, color=GREEN, s=10, label="Retained")
ax3.scatter(samp[samp["churned"]==1]["avg_watch_time_per_day"],
            samp[samp["churned"]==1]["last_login_days"],
            alpha=0.4, color=RED, s=10, label="Churned")
ax3.set_facecolor(SURFACE); ax3.tick_params(colors=WHITE, labelsize=8)
ax3.set_xlabel("Avg Watch Time/Day (h)", color=MUTED, fontsize=8)
ax3.set_ylabel("Days Since Last Login", color=MUTED, fontsize=8)
ax3.legend(facecolor=SURFACE, labelcolor=WHITE, fontsize=8)
for sp in ax3.spines.values(): sp.set_color("#333333")
ax3.set_title("Engagement vs Recency", color=WHITE, fontsize=11, pad=6)

# Monthly fee by subscription
ax4 = fig.add_subplot(gs[1, 0]); ax4.set_facecolor(SURFACE)
sub_data = [df[df["subscription_type"]==s]["monthly_fee"] for s in ["Basic","Standard","Premium"]]
bp4 = ax4.boxplot(sub_data, labels=["Basic","Standard","Premium"], patch_artist=True,
                  medianprops=dict(color=WHITE, linewidth=2))
for box, c in zip(bp4["boxes"], [RED, AMBER, GREEN]): box.set_facecolor(c)
for w in bp4["whiskers"]+bp4["caps"]: w.set_color("#888888")
ax4.set_facecolor(SURFACE); ax4.tick_params(colors=WHITE, labelsize=9)
ax4.set_ylabel("Monthly Fee ($)", color=MUTED, fontsize=9)
for sp in ax4.spines.values(): sp.set_color("#333333")
ax4.set_title("Monthly Fee by Subscription", color=WHITE, fontsize=11, pad=6)

# Payment method churn bar
ax5 = fig.add_subplot(gs[1, 1:]); ax5.set_facecolor(SURFACE)
pmdf = df.groupby("payment_method")["churned"].mean().sort_values(ascending=False)
bar_colors = [RED if r > 0.55 else AMBER if r > 0.45 else GREEN for r in pmdf.values]
bars5 = ax5.bar(pmdf.index, pmdf.values*100, color=bar_colors, edgecolor=BG)
for b in bars5:
    ax5.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
             f"{b.get_height():.1f}%", ha="center", color=WHITE, fontsize=9)
ax5.set_facecolor(SURFACE); ax5.set_ylim(0, 75)
ax5.tick_params(colors=WHITE, labelsize=9); ax5.set_ylabel("Churn Rate (%)", color=MUTED, fontsize=9)
ax5.set_title("Churn Rate by Payment Method", color=WHITE, fontsize=11, pad=6)
for sp in ax5.spines.values(): sp.set_color("#333333")

save("03_eda_behaviour_financials.png")

# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE 4 — Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
fig = dark_fig(12, 9)
fig.text(0.5, 0.97, "Page 2 — EDA: Feature Correlation Heatmap", color=WHITE,
         fontsize=15, fontweight="bold", ha="center")
ax = fig.add_subplot(111); ax.set_facecolor(BG)
num_df = df.copy()
for col in ["gender","subscription_type","region","device","payment_method","favorite_genre"]:
    num_df[col] = pd.factorize(num_df[col])[0]
num_df.drop(columns=["customer_id"], errors="ignore", inplace=True)
corr = num_df.corr(numeric_only=True)
cmap = LinearSegmentedColormap.from_list("rdbu", ["#1e40af","#1c1c1c","#E50914"])
im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", color=WHITE, fontsize=8)
ax.set_yticklabels(corr.columns, color=WHITE, fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr.columns)):
        v = corr.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=6.5, color=WHITE if abs(v) > 0.3 else MUTED)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.ax.tick_params(colors=WHITE, labelsize=8)
cb.ax.yaxis.label.set_color(WHITE)
ax.set_facecolor(BG)
plt.tight_layout(rect=[0, 0, 1, 0.95])
save("04_correlation_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE 5 — Predict Churn Page (gauge diagram)
# ═══════════════════════════════════════════════════════════════════════════════
fig = dark_fig(14, 7)
fig.text(0.5, 0.97, "Page 3 — Predict Churn: Output Panel", color=WHITE,
         fontsize=15, fontweight="bold", ha="center")

# Gauge
ax = fig.add_axes([0.05, 0.08, 0.44, 0.82])
ax.set_facecolor(BG); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.2, 1.3); ax.axis("off")
theta = np.linspace(np.pi, 0, 300)
# zones
for (t1, t2, col) in [(np.pi, np.pi*0.6, "#0f2e1a"),
                       (np.pi*0.6, np.pi*0.3, "#4a3010"),
                       (np.pi*0.3, 0, "#4a1010")]:
    ts = np.linspace(t1, t2, 100)
    for r in np.linspace(0.6, 0.95, 15):
        ax.plot(r*np.cos(ts), r*np.sin(ts), color=col, lw=3, alpha=0.8)
# needle at 73%
needle_angle = np.pi * (1 - 0.73)
ax.annotate("", xy=(0.78*np.cos(needle_angle), 0.78*np.sin(needle_angle)),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=WHITE, lw=2))
ax.add_patch(plt.Circle((0, 0), 0.06, color=WHITE, zorder=5))
ax.text(0, -0.1, "73%", color=RED, fontsize=28, fontweight="bold", ha="center")
ax.text(0, -0.17, "Churn Probability", color=MUTED, fontsize=10, ha="center")
# labels
for ang, lbl in [(np.pi, "0%"), (np.pi*0.7, "Low"), (np.pi*0.4, "Med"),
                 (np.pi*0.15, "High"), (0, "100%")]:
    ax.text(1.1*np.cos(ang), 1.1*np.sin(ang), lbl, color=WHITE,
            fontsize=8, ha="center", va="center")
ax.set_title("Churn Probability Gauge", color=WHITE, fontsize=12, pad=4)

# Risk panel
ax2 = fig.add_axes([0.54, 0.08, 0.43, 0.82])
ax2.set_facecolor(BG); ax2.axis("off")
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

# High-risk badge
ax2.add_patch(FancyBboxPatch((0.02, 0.72), 0.96, 0.24,
    boxstyle="round,pad=0.03", facecolor="#4a1010", edgecolor=RED, lw=2))
ax2.text(0.5, 0.88, "HIGH RISK", color=RED, fontsize=18, fontweight="bold", ha="center")
ax2.text(0.5, 0.79, "Prediction: Will Churn  |  Model: GradientBoost", color=WHITE, fontsize=9, ha="center")

# Risk signals
signals = [
    ("Last login: 45 days ago",          RED),
    ("Watch hours: 2.1 h (very low)",    RED),
    ("Avg watch time/day: 0.3 h",        AMBER),
    ("Subscription: Basic plan",          AMBER),
]
y = 0.62
ax2.text(0.04, y+0.04, "Risk Signals Detected:", color=WHITE, fontsize=10, fontweight="bold")
for sig, col in signals:
    y -= 0.10
    ax2.add_patch(plt.Circle((0.07, y+0.012), 0.018, color=col))
    ax2.text(0.12, y, sig, color=WHITE, fontsize=9, va="bottom")

# Risk threshold legend
ax2.text(0.04, y-0.08, "Risk Thresholds:", color=WHITE, fontsize=10, fontweight="bold")
for j, (lbl, rng, col) in enumerate([("Low", "< 40%", GREEN),
                                      ("Medium", "40–70%", AMBER),
                                      ("High", "> 70%", RED)]):
    yy = y - 0.17 - j*0.085
    ax2.add_patch(FancyBboxPatch((0.04, yy), 0.92, 0.07,
        boxstyle="round,pad=0.02", facecolor=col+"22", edgecolor=col, lw=1.5))
    ax2.text(0.5, yy+0.035, f"{lbl}  ({rng})", color=col, fontsize=9,
             fontweight="bold", ha="center", va="center")

save("05_predict_churn_page.png")

# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE 6 — Model Performance Page
# ═══════════════════════════════════════════════════════════════════════════════
fig = dark_fig(16, 8)
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4,
                        left=0.06, right=0.97, top=0.89, bottom=0.1)
fig.text(0.5, 0.96, "Page 4 — Model Performance", color=WHITE,
         fontsize=16, fontweight="bold", ha="center")

models_  = ["GradientBoost", "RandomForest", "Logistic"]
accs     = [99.1, 96.8, 88.5]
aucs     = [0.9986, 0.9964, 0.9599]
bar_cols = [RED, AMBER, "#555555"]

# Accuracy
ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(SURFACE)
bars = ax.bar(models_, accs, color=bar_cols, edgecolor=BG, width=0.5)
for b, v in zip(bars, accs):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
            f"{v:.1f}%", ha="center", color=WHITE, fontsize=10, fontweight="bold")
ax.set_facecolor(SURFACE); ax.set_ylim(0, 110)
ax.tick_params(colors=WHITE, labelsize=9); ax.set_ylabel("Accuracy (%)", color=MUTED)
for sp in ax.spines.values(): sp.set_color("#333333")
ax.set_title("Model Accuracy Comparison", color=WHITE, fontsize=12, pad=8)
ax.axhline(y=99.1, color=RED, linestyle="--", linewidth=1, alpha=0.5)

# ROC-AUC + Feature importance stacked
ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor(SURFACE)
fi = pd.Series(model.feature_importances_, index=feats).sort_values()
cmap_fi = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(fi)))
ax2.barh(fi.index, fi.values*100, color=cmap_fi, edgecolor=BG)
ax2.set_facecolor(SURFACE); ax2.tick_params(colors=WHITE, labelsize=8)
ax2.set_xlabel("Importance (%)", color=MUTED, fontsize=9)
for sp in ax2.spines.values(): sp.set_color("#333333")
ax2.set_title("Feature Importances — GradientBoost", color=WHITE, fontsize=12, pad=8)
for i, (v, name) in enumerate(zip(fi.values, fi.index)):
    if v*100 > 1:
        ax2.text(v*100+0.3, i, f"{v*100:.1f}%", va="center", color=WHITE, fontsize=7.5)

save("06_model_performance.png")

# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE 7 — System / Architecture Flow Diagram
# ═══════════════════════════════════════════════════════════════════════════════
fig = dark_fig(14, 6)
fig.text(0.5, 0.95, "System Architecture — End-to-End Data Flow", color=WHITE,
         fontsize=15, fontweight="bold", ha="center")
ax = fig.add_axes([0, 0, 1, 0.9]); ax.set_facecolor(BG); ax.axis("off")
ax.set_xlim(0, 14); ax.set_ylim(0, 6)

boxes = [
    (1.0,  2.5, 2.2, 1.2, "CSV Dataset\n5,000 rows",       BLUE,  "Data Layer"),
    (4.2,  2.5, 2.2, 1.2, "train.py\nML Training",          AMBER, "ML Layer"),
    (7.4,  2.5, 2.2, 1.2, "app.py\nFastAPI Backend",        GREEN, "API Layer"),
    (10.6, 2.5, 2.2, 1.2, "frontend.py\nStreamlit UI",      RED,   "Frontend"),
]
for (x, y, w, h, label, col, sub) in boxes:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                 facecolor=col+"33", edgecolor=col, lw=2.5))
    lines = label.split("\n")
    ax.text(x+w/2, y+h/2+0.1, lines[0], color=WHITE, fontsize=10,
            fontweight="bold", ha="center", va="center")
    ax.text(x+w/2, y+h/2-0.2, lines[1], color=col, fontsize=8.5,
            ha="center", va="center")
    ax.text(x+w/2, y-0.35, sub, color=MUTED, fontsize=8, ha="center")

# Arrows
for x_start, x_end, lbl in [
    (3.2, 4.2, "reads"), (6.4, 7.4, "saves\n.pkl"), (9.6, 10.6, "POST\n/predict")]:
    ax.annotate("", xy=(x_end, 3.1), xytext=(x_start, 3.1),
                arrowprops=dict(arrowstyle="-|>", color=WHITE, lw=1.8))
    ax.text((x_start+x_end)/2, 3.45, lbl, color=MUTED, fontsize=8, ha="center")

# Models box below train.py
ax.add_patch(FancyBboxPatch((4.2, 0.6), 2.2, 0.9, boxstyle="round,pad=0.1",
             facecolor="#2d2d2d", edgecolor="#555555", lw=1.5))
ax.text(5.3, 1.05, "models/  (.pkl files)", color=MUTED, fontsize=8, ha="center")
ax.annotate("", xy=(5.3, 1.5), xytext=(5.3, 2.5),
            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.5))
ax.annotate("", xy=(8.5, 2.5), xytext=(8.5, 1.5),
            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.5,
                            connectionstyle="arc3,rad=0"))
ax.annotate("", xy=(8.5, 1.5), xytext=(6.4, 1.5),
            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.5))
ax.text(7.4, 1.1, "loads at startup", color=MUTED, fontsize=7.5, ha="center")

save("07_architecture_flow.png")

print("\nAll 7 dashboard images saved to:", IMGDIR)

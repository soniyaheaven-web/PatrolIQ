import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

IN_PATH = "data/processed/crimes_features.csv"
OUT_DIR = "outputs/temporal"
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
sns.set_theme(style="whitegrid")

# ---------------------------------------------------
# 1. Load + cyclical encoding
# ---------------------------------------------------
df = pd.read_csv(
    IN_PATH,
    usecols=["id", "year", "hour", "day_of_week", "day_of_week_num", "month",
             "is_weekend", "season", "primary_type", "arrest",
             "crime_severity_score"],
)
print("Loaded:", df.shape)

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week_num"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week_num"] / 7)
df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

FEATS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
T = df[FEATS].values

# ---------------------------------------------------
# 2. K selection (3 to 6), final k 3-5 la irundhu
# ---------------------------------------------------
print("\n=== Temporal K-Means: k selection ===")
K_RANGE = range(3, 7)
inertias, sils, dbis = [], [], []
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(T)
    s = silhouette_score(T, km.labels_, sample_size=20000, random_state=SEED)
    d = davies_bouldin_score(T, km.labels_)
    inertias.append(km.inertia_)
    sils.append(s)
    dbis.append(d)
    print(f"k={k}: inertia={km.inertia_:,.0f} | silhouette={s:.4f} | DBI={d:.4f}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(list(K_RANGE), inertias, marker="o"); ax[0].set_title("Elbow (Inertia)")
ax[1].plot(list(K_RANGE), sils, marker="o", color="green"); ax[1].set_title("Silhouette")
ax[2].plot(list(K_RANGE), dbis, marker="o", color="red"); ax[2].set_title("Davies-Bouldin")
for a in ax:
    a.set_xlabel("k")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/temporal_k_selection.png", dpi=120)
plt.close()

# Project requirement: 3-5 patterns
valid = [(k, s) for k, s in zip(K_RANGE, sils) if k <= 5]
best_k = max(valid, key=lambda x: x[1])[0]
print("\nSelected k (3-5 range, best silhouette):", best_k)

km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(T)
df["time_cluster"] = km.labels_
final_sil = silhouette_score(T, km.labels_, sample_size=20000, random_state=SEED)
final_dbi = davies_bouldin_score(T, km.labels_)
print(f"Final: silhouette={final_sil:.4f}, DBI={final_dbi:.4f}")

# ---------------------------------------------------
# 3. Cluster profiles (auto label)
# ---------------------------------------------------
def circular_mean_hour(g):
    ang = np.arctan2(g["hour_sin"].mean(), g["hour_cos"].mean())
    return (ang * 24 / (2 * np.pi)) % 24


def hour_band(h):
    if 5 <= h < 12:
        return "Morning"
    if 12 <= h < 17:
        return "Afternoon"
    if 17 <= h < 21:
        return "Evening"
    return "Late Night"


rows = []
for c, g in df.groupby("time_cluster"):
    wk = g["is_weekend"].mean() * 100
    mh = circular_mean_hour(g)
    day_text = "Weekend-heavy" if wk > 40 else ("Weekday-heavy" if wk < 20 else "Mixed days")
    rows.append({
        "cluster": c,
        "label": f"{hour_band(mh)} | {day_text}",
        "total_crimes": len(g),
        "share_%": round(len(g) / len(df) * 100, 1),
        "avg_hour": round(mh, 1),
        "top3_hours": ", ".join(map(str, g["hour"].value_counts().head(3).index)),
        "weekend_%": round(wk, 1),
        "top3_months": ", ".join(map(str, g["month"].value_counts().head(3).index)),
        "top_season": g["season"].value_counts().idxmax(),
        "dominant_crime": g["primary_type"].value_counts().idxmax(),
        "arrest_rate_%": round(g["arrest"].mean() * 100, 1),
        "avg_severity": round(g["crime_severity_score"].mean(), 2),
    })
prof = pd.DataFrame(rows).sort_values("total_crimes", ascending=False)
print("\n=== Temporal Cluster Profile ===")
print(prof.to_string(index=False))
prof.to_csv(f"{OUT_DIR}/temporal_cluster_profile.csv", index=False)

label_map = dict(zip(prof["cluster"], prof["label"]))
df["time_cluster_label"] = df["time_cluster"].map(label_map)

# ---------------------------------------------------
# 4. Chart: hourly distribution per cluster
# ---------------------------------------------------
plt.figure(figsize=(11, 5))
for c in sorted(df["time_cluster"].unique()):
    share = (df[df["time_cluster"] == c]["hour"]
             .value_counts(normalize=True).sort_index() * 100)
    plt.plot(share.index, share.values, marker="o", label=f"C{c}: {label_map[c]}")
plt.xticks(range(0, 24))
plt.xlabel("Hour"); plt.ylabel("% of cluster crimes")
plt.title("Hourly Profile of Each Temporal Cluster")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/cluster_hourly_profile.png", dpi=120)
plt.close()

# ---------------------------------------------------
# 5. Chart: cluster share by month
# ---------------------------------------------------
ct = pd.crosstab(df["month"], df["time_cluster"], normalize="index") * 100
ct.columns = [f"C{c}" for c in ct.columns]
ct.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="tab10")
plt.title("Temporal Cluster Mix by Month (%)")
plt.xlabel("Month"); plt.ylabel("%")
plt.legend(title="Cluster", bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/cluster_by_month.png", dpi=120)
plt.close()

# ---------------------------------------------------
# 6. Peak time slots
# ---------------------------------------------------
print("\n=== Peak Danger Time Slots ===")
slots = df.groupby(["day_of_week", "hour"]).size().sort_values(ascending=False)
print("Top 10 (day, hour) slots:")
print(slots.head(10))
print("\nTop 10 slots excluding hour 0 (midnight recording artifact):")
print(slots[slots.index.get_level_values("hour") != 0].head(10))

# Severity-weighted peak hours (violent crimes: severity >= 7)
violent = df[df["crime_severity_score"] >= 7]
print("\nTop 5 hours for severe crimes (severity>=7):")
print(violent["hour"].value_counts().head(5))

# ---------------------------------------------------
# 7. Seasonal trend (full years 2021-2025 only)
# ---------------------------------------------------
full = df[df["year"].between(2021, 2025)]
monthly_avg = full.groupby("month").size() / 5
print("\n=== Avg crimes per month (2021-2025) ===")
print(monthly_avg.round(0).astype(int).to_string())

plt.figure(figsize=(9, 4))
sns.barplot(x=monthly_avg.index, y=monthly_avg.values, color="#ff7f0e")
plt.title("Average Crimes per Month (2021-2025, sample)")
plt.xlabel("Month"); plt.ylabel("Avg crimes")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/seasonal_monthly_avg.png", dpi=120)
plt.close()

season_share = df["season"].value_counts(normalize=True).mul(100).round(1)
print("\nSeason share %:\n", season_share.to_string())

# ---------------------------------------------------
# 8. Weekday vs Weekend hourly pattern
# ---------------------------------------------------
plt.figure(figsize=(11, 5))
for flag, name, col in [(0, "Weekday", "#1f77b4"), (1, "Weekend", "#d62728")]:
    s = (df[df["is_weekend"] == flag]["hour"]
         .value_counts(normalize=True).sort_index() * 100)
    plt.plot(s.index, s.values, marker="o", label=name, color=col)
plt.xticks(range(0, 24))
plt.xlabel("Hour"); plt.ylabel("% of crimes")
plt.title("Weekday vs Weekend: Hourly Crime Pattern")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/weekday_vs_weekend.png", dpi=120)
plt.close()

# ---------------------------------------------------
# 9. Temporal profile by crime type (top 10)
# ---------------------------------------------------
top10 = df["primary_type"].value_counts().head(10).index
sub = df[df["primary_type"].isin(top10)]

hour_prof = pd.crosstab(sub["primary_type"], sub["hour"], normalize="index") * 100
plt.figure(figsize=(13, 5))
sns.heatmap(hour_prof, cmap="YlOrRd")
plt.title("Hourly Profile by Crime Type (% of each type)")
plt.xlabel("Hour"); plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/crime_type_hourly_heatmap.png", dpi=120)
plt.close()

type_cluster = pd.crosstab(sub["primary_type"], sub["time_cluster"], normalize="index") * 100
type_cluster.columns = [f"C{c}" for c in type_cluster.columns]
plt.figure(figsize=(8, 5))
sns.heatmap(type_cluster, annot=True, fmt=".0f", cmap="Blues")
plt.title("Crime Type vs Temporal Cluster (% of type)")
plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/crime_type_vs_cluster.png", dpi=120)
plt.close()

peak_rows = []
for t in top10:
    hrs = df[df["primary_type"] == t]["hour"].value_counts()
    hrs_no0 = hrs.drop(0, errors="ignore")
    peak_rows.append({
        "crime_type": t,
        "peak_hour": hrs.idxmax(),
        "peak_hour_excl_midnight": hrs_no0.idxmax(),
        "weekend_%": round(df[df["primary_type"] == t]["is_weekend"].mean() * 100, 1),
    })
peak_df = pd.DataFrame(peak_rows)
print("\n=== Peak hour by crime type ===")
print(peak_df.to_string(index=False))
peak_df.to_csv(f"{OUT_DIR}/crime_type_peak_hours.csv", index=False)

# ---------------------------------------------------
# 10. Save labels
# ---------------------------------------------------
df[["id", "time_cluster", "time_cluster_label"]].to_csv(
    "data/processed/temporal_labels.csv", index=False)
print("\nAll outputs saved in:", OUT_DIR)
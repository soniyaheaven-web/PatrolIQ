import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

IN_PATH = "data/processed/crimes_features.csv"
OUT_DIR = "outputs/eda"
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]


def save(name):
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{name}.png", dpi=120)
    plt.close()
    print("Saved:", name)


# ---------------------------------------------------
# Load
# ---------------------------------------------------
df = pd.read_csv(IN_PATH, low_memory=False)
df["date"] = pd.to_datetime(df["date"])
print("Loaded:", df.shape)

# ---------------------------------------------------
# 1. Statistical summary
# ---------------------------------------------------
print("\n=== Overall Summary ===")
print("Total records     :", f"{len(df):,}")
print("Date range        :", df["date"].min(), "to", df["date"].max())
print("Crime types       :", df["primary_type"].nunique())
print("Districts         :", df["district"].nunique())
print("Arrest rate       : {:.1f}%".format(df["arrest"].mean() * 100))
print("Domestic rate     : {:.1f}%".format(df["domestic"].mean() * 100))
print("Avg severity      : {:.2f}".format(df["crime_severity_score"].mean()))

# ---------------------------------------------------
# 2. Crime type distribution (Top 15)
# ---------------------------------------------------
plt.figure(figsize=(10, 6))
top_types = df["primary_type"].value_counts().head(15)
sns.barplot(x=top_types.values, y=top_types.index, color="#d62728")
plt.title("Top 15 Crime Types")
plt.xlabel("Number of Crimes")
plt.ylabel("")
save("01_top_crime_types")

# ---------------------------------------------------
# 3. Hourly pattern
# ---------------------------------------------------
plt.figure(figsize=(10, 5))
hourly = df["hour"].value_counts().sort_index()
plt.plot(hourly.index, hourly.values, marker="o", color="#1f77b4")
plt.xticks(range(0, 24))
plt.title("Crimes by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Number of Crimes")
save("02_hourly_pattern")
print("Peak hour:", hourly.idxmax(), "| Safest hour:", hourly.idxmin())

# ---------------------------------------------------
# 4. Day of week
# ---------------------------------------------------
plt.figure(figsize=(9, 5))
dow = df["day_of_week"].value_counts().reindex(DAY_ORDER)
sns.barplot(x=dow.index, y=dow.values, color="#2ca02c")
plt.xticks(rotation=30)
plt.title("Crimes by Day of Week")
plt.ylabel("Number of Crimes")
save("03_day_of_week")

# ---------------------------------------------------
# 5. Monthly trend (year-month)
# ---------------------------------------------------
plt.figure(figsize=(12, 5))
monthly = df.groupby(df["date"].dt.to_period("M")).size()
monthly.index = monthly.index.to_timestamp()
plt.plot(monthly.index, monthly.values, marker="o", color="#9467bd")
plt.title("Monthly Crime Trend (2021 - 2026)")
plt.ylabel("Number of Crimes")
save("04_monthly_trend")

# ---------------------------------------------------
# 6. Seasonal pattern
# ---------------------------------------------------
plt.figure(figsize=(7, 5))
season = df["season"].value_counts().reindex(["Winter", "Spring", "Summer", "Fall"])
sns.barplot(x=season.index, y=season.values,
            palette=["#6baed6", "#74c476", "#fd8d3c", "#a1664a"])
plt.title("Crimes by Season")
plt.ylabel("Number of Crimes")
save("05_seasonal")

# ---------------------------------------------------
# 7. Hour x Day heatmap
# ---------------------------------------------------
plt.figure(figsize=(12, 5))
heat = (df.groupby(["day_of_week", "hour"]).size()
        .unstack().reindex(DAY_ORDER))
sns.heatmap(heat, cmap="YlOrRd")
plt.title("Crime Heatmap: Day vs Hour")
plt.xlabel("Hour")
plt.ylabel("")
save("06_hour_day_heatmap")

# ---------------------------------------------------
# 8. Arrest rate by crime type (Top 10 crimes)
# ---------------------------------------------------
plt.figure(figsize=(10, 5))
top10 = df["primary_type"].value_counts().head(10).index
arrest_rate = (df[df["primary_type"].isin(top10)]
               .groupby("primary_type")["arrest"].mean() * 100)
arrest_rate = arrest_rate.sort_values()
sns.barplot(x=arrest_rate.values, y=arrest_rate.index, color="#17becf")
plt.title("Arrest Rate (%) by Crime Type (Top 10)")
plt.xlabel("Arrest Rate %")
plt.ylabel("")
save("07_arrest_rate")

# ---------------------------------------------------
# 9. Domestic incidents by crime type
# ---------------------------------------------------
plt.figure(figsize=(10, 5))
dom = (df.groupby("primary_type")["domestic"].mean() * 100)
dom = dom.sort_values(ascending=False).head(10)
sns.barplot(x=dom.values, y=dom.index, color="#e377c2")
plt.title("Domestic Incident Rate (%) - Top 10 Crime Types")
plt.xlabel("Domestic %")
plt.ylabel("")
save("08_domestic_rate")

# ---------------------------------------------------
# 10. Geographic scatter (100K sample)
# ---------------------------------------------------
plt.figure(figsize=(7, 9))
s = df.sample(100_000, random_state=42)
plt.scatter(s["longitude"], s["latitude"], s=1, alpha=0.3, c="#d62728")
plt.title("Crime Locations - Chicago (100K sample)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
save("09_geo_scatter")

# ---------------------------------------------------
# 11. District-wise crimes
# ---------------------------------------------------
plt.figure(figsize=(12, 5))
dist = df["district"].value_counts().sort_index()
sns.barplot(x=dist.index, y=dist.values, color="#ff7f0e")
plt.title("Crimes by Police District")
plt.xlabel("District")
plt.ylabel("Number of Crimes")
save("10_district_counts")

# ---------------------------------------------------
# 12. Severity by time of day
# ---------------------------------------------------
plt.figure(figsize=(8, 5))
tod = (df.groupby("time_of_day")["crime_severity_score"].mean()
       .reindex(["Morning", "Afternoon", "Evening", "Night"]))
sns.barplot(x=tod.index, y=tod.values, color="#8c564b")
plt.title("Average Crime Severity by Time of Day")
plt.ylabel("Avg Severity Score")
save("11_severity_time_of_day")

# ---------------------------------------------------
# 13. Correlation heatmap
# ---------------------------------------------------
num_cols = ["hour", "day_of_week_num", "month", "is_weekend", "arrest",
            "domestic", "crime_severity_score", "latitude", "longitude",
            "district", "ward", "community_area"]
plt.figure(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlation")
save("12_correlation")

# ---------------------------------------------------
# Key insights
# ---------------------------------------------------
print("\n=== Key Insights ===")
print("Top 3 crimes      :", df["primary_type"].value_counts().head(3).index.tolist())
print("Peak day          :", dow.idxmax())
print("Peak season       :", season.idxmax())
print("Peak month        :", df["month"].value_counts().idxmax())
print("Weekend share     : {:.1f}%".format(df["is_weekend"].mean() * 100))
print("Top 3 districts   :", df["district"].value_counts().head(3).index.tolist())
print("Highest arrest    :", arrest_rate.idxmax(),
      "({:.1f}%)".format(arrest_rate.max()))
print("\nAll charts saved in:", OUT_DIR)
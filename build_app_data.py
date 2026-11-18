import calendar
import json
import os
import shutil

import pandas as pd

os.makedirs("data/app", exist_ok=True)

COLS = ["id", "date", "year", "month", "hour", "day_of_week", "is_weekend",
        "season", "primary_type", "arrest", "domestic", "district",
        "latitude", "longitude", "crime_severity_score"]

df = pd.read_csv("data/processed/crimes_features.csv", usecols=COLS, low_memory=False)
km = pd.read_csv("data/processed/kmeans_labels.csv")
tl = pd.read_csv("data/processed/temporal_labels.csv", usecols=["id", "time_cluster"])
df = df.merge(km, on="id", how="left").merge(tl, on="id", how="left")
print("Missing labels:", df[["kmeans_cluster", "time_cluster"]].isnull().sum().to_dict())


def band(h):
    if 5 <= h < 12:
        return "Morning"
    if 12 <= h < 17:
        return "Afternoon"
    if 17 <= h < 21:
        return "Evening"
    return "Late Night"


tp = pd.read_csv("outputs/temporal/temporal_cluster_profile.csv")
names = list(calendar.month_abbr)
labels = {}
for _, r in tp.iterrows():
    months = sorted(int(m) for m in str(r["top3_months"]).split(","))
    labels[int(r["cluster"])] = f"{band(float(r['avg_hour']))} | " + "/".join(names[m] for m in months)

tp["label"] = tp["cluster"].map(labels)
tp.to_csv("data/app/temporal_profile.csv", index=False)
df["time_cluster_label"] = df["time_cluster"].map(labels)
print("Temporal labels:", labels)

df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d %H:%M")
app_df = df.sample(200_000, random_state=42).reset_index(drop=True)
app_df.to_csv("data/app/crimes_app.csv.gz", index=False, compression="gzip")

shutil.copy("data/processed/dimred_embeddings.csv", "data/app/dimred_embeddings.csv")

try:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    frames = []
    for exp in ["PatrolIQ_Geographic_Clustering", "PatrolIQ_Temporal_Clustering",
                "PatrolIQ_Dimensionality_Reduction"]:
        r = mlflow.search_runs(experiment_names=[exp])
        r["experiment"] = exp
        frames.append(r)
    runs = pd.concat(frames, ignore_index=True)
    keep = (~runs.columns.str.startswith("tags.mlflow.")) | (runs.columns == "tags.mlflow.runName")
    runs = runs.loc[:, keep].drop(columns=["artifact_uri"], errors="ignore")
    runs.to_csv("data/app/mlflow_runs.csv", index=False)
    print("MLflow runs exported:", runs.shape)

    client = MlflowClient()
    reg_model = client.get_registered_model("PatrolIQ_KMeans_Hotspots")
    alias_map = dict(reg_model.aliases)   # {"champion": "1"}
    reg = []
    for v in client.search_model_versions("name='PatrolIQ_KMeans_Hotspots'"):
        reg.append({"name": v.name, "version": v.version,
                    "aliases": [a for a, ver in alias_map.items() if str(ver) == str(v.version)],
                    "tags": dict(v.tags), "run_id": v.run_id})
    with open("data/app/model_registry.json", "w") as f:
        json.dump(reg, f, indent=2)
    print("Registry exported:", reg)
except Exception as e:
    print("MLflow export skip aachu:", type(e).__name__, "-", str(e)[:150])

print("\nFiles in data/app:")
for fn in sorted(os.listdir("data/app")):
    print(f"  {fn}: {os.path.getsize(os.path.join('data/app', fn)) / 1e6:.1f} MB")

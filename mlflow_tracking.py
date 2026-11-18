import os
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, fcluster

SEED = 42
FINAL_K = 9
MODEL_NAME = "PatrolIQ_KMeans_Hotspots"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()


def log_files(paths):
    for p in paths:
        if os.path.exists(p):
            mlflow.log_artifact(p)


def score(Xe, labels, sample=20000):
    """Silhouette + DBI (noise = -1 ignore)"""
    mask = labels != -1
    Xm, lm = Xe[mask], labels[mask]
    if len(set(lm)) < 2:
        return None
    sil = silhouette_score(Xm, lm, sample_size=min(sample, len(Xm)), random_state=SEED)
    dbi = davies_bouldin_score(Xm, lm)
    return float(sil), float(dbi), float((~mask).mean() * 100)


# ===================================================
# EXPERIMENT 1: Geographic Clustering
# ===================================================
mlflow.set_experiment("PatrolIQ_Geographic_Clustering")

df = pd.read_csv(
    "data/processed/crimes_features.csv",
    usecols=["id", "latitude", "longitude"],
)
lat0 = df["latitude"].mean()
df["y_km"] = (df["latitude"] - lat0) * 111.0
df["x_km"] = (df["longitude"] - df["longitude"].mean()) * 111.0 * np.cos(np.radians(lat0))
X = df[["x_km", "y_km"]].values
print("Loaded:", df.shape)

# ---- K-Means (k = 3..10) ----
final_model = None
for k in range(3, 11):
    with mlflow.start_run(run_name=f"KMeans_k{k}"):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        sil, dbi, _ = score(X, km.labels_)

        mlflow.log_params({
            "algorithm": "KMeans",
            "n_clusters": k,
            "n_init": 10,
            "distance_metric": "euclidean (km)",
            "features": "x_km,y_km",
            "n_samples": len(X),
            "seed": SEED,
        })
        mlflow.log_metrics({
            "inertia": float(km.inertia_),
            "silhouette": sil,
            "davies_bouldin": dbi,
        })
        print(f"KMeans k={k}: sil={sil:.4f} dbi={dbi:.4f}")

        if k == FINAL_K:
            mlflow.set_tag("selected", "final")
            log_files([
                "outputs/clustering/kmeans_k_selection.png",
                "outputs/clustering/kmeans_map.png",
                "outputs/clustering/kmeans_cluster_profile.csv",
            ])
            final_model = km
            final_run_id = mlflow.active_run().info.run_id
            mlflow.sklearn.log_model(km, artifact_path="model")

# ---- Model Registry ----
if final_model is not None:
    result = mlflow.register_model(f"runs:/{final_run_id}/model", MODEL_NAME)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(int(v.version) for v in versions)
    client.set_registered_model_alias(MODEL_NAME, "champion", str(latest))
    client.set_model_version_tag(MODEL_NAME, str(latest), "k", str(FINAL_K))
    client.set_model_version_tag(MODEL_NAME, str(latest), "task", "geographic hotspots")
    print(f"Registered {MODEL_NAME} v{latest} with alias 'champion'")

# ---- DBSCAN grid (100K sample) ----
samp = df.sample(100_000, random_state=SEED).reset_index(drop=True)
Xs = samp[["x_km", "y_km"]].values
for eps in [0.3, 0.5, 0.8]:
    for ms in [50, 100]:
        with mlflow.start_run(run_name=f"DBSCAN_eps{eps}_ms{ms}"):
            lab = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1).fit(Xs).labels_
            n_cl = len(set(lab)) - (1 if -1 in lab else 0)
            noise = float((lab == -1).mean() * 100)
            mlflow.log_params({
                "algorithm": "DBSCAN",
                "eps_km": eps,
                "min_samples": ms,
                "distance_metric": "euclidean (km)",
                "n_samples": len(Xs),
            })
            mlflow.log_metrics({"n_clusters": n_cl, "noise_pct": noise})
            res = score(Xs, lab)
            if res is not None:
                mlflow.log_metrics({"silhouette": res[0], "davies_bouldin": res[1]})
            print(f"DBSCAN eps={eps} ms={ms}: clusters={n_cl} noise={noise:.1f}%")
            if eps == 0.8 and ms == 100:
                mlflow.set_tag("selected", "best_dbscan")
                log_files(["outputs/clustering/dbscan_map.png"])

# ---- Hierarchical (5000 sample, Ward) ----
with mlflow.start_run(run_name=f"Hierarchical_ward_k{FINAL_K}"):
    hs = df.sample(5000, random_state=SEED).reset_index(drop=True)
    Xh = hs[["x_km", "y_km"]].values
    Z = linkage(Xh, method="ward")
    hl = fcluster(Z, t=FINAL_K, criterion="maxclust")
    sil, dbi, _ = score(Xh, hl)
    mlflow.log_params({
        "algorithm": "Hierarchical",
        "linkage": "ward",
        "n_clusters": FINAL_K,
        "distance_metric": "euclidean (km)",
        "n_samples": 5000,
    })
    mlflow.log_metrics({"silhouette": sil, "davies_bouldin": dbi})
    log_files(["outputs/clustering/dendrogram.png",
               "outputs/clustering/algorithm_comparison.csv"])
    print(f"Hierarchical: sil={sil:.4f} dbi={dbi:.4f}")

# ===================================================
# EXPERIMENT 2: Temporal Clustering
# ===================================================
mlflow.set_experiment("PatrolIQ_Temporal_Clustering")

t = pd.read_csv("data/processed/crimes_features.csv",
                usecols=["hour", "day_of_week_num", "month"])
t["hour_sin"] = np.sin(2 * np.pi * t["hour"] / 24)
t["hour_cos"] = np.cos(2 * np.pi * t["hour"] / 24)
t["dow_sin"] = np.sin(2 * np.pi * t["day_of_week_num"] / 7)
t["dow_cos"] = np.cos(2 * np.pi * t["day_of_week_num"] / 7)
t["month_sin"] = np.sin(2 * np.pi * (t["month"] - 1) / 12)
t["month_cos"] = np.cos(2 * np.pi * (t["month"] - 1) / 12)
FEATS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
T = t[FEATS].values

for k in range(3, 7):
    with mlflow.start_run(run_name=f"Temporal_KMeans_k{k}"):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(T)
        sil, dbi, _ = score(T, km.labels_)
        mlflow.log_params({
            "algorithm": "KMeans",
            "n_clusters": k,
            "n_init": 10,
            "features": ",".join(FEATS),
            "encoding": "cyclical sin/cos",
            "n_samples": len(T),
        })
        mlflow.log_metrics({
            "inertia": float(km.inertia_),
            "silhouette": sil,
            "davies_bouldin": dbi,
        })
        print(f"Temporal k={k}: sil={sil:.4f} dbi={dbi:.4f}")
        if k == 4:
            mlflow.set_tag("selected", "final")
            log_files([
                "outputs/temporal/temporal_k_selection.png",
                "outputs/temporal/temporal_cluster_profile.csv",
                "outputs/temporal/cluster_hourly_profile.png",
                "outputs/temporal/weekday_vs_weekend.png",
            ])

# ===================================================
# EXPERIMENT 3: Dimensionality Reduction
# ===================================================
mlflow.set_experiment("PatrolIQ_Dimensionality_Reduction")

with open("outputs/dimred/dimred_metrics.json") as f:
    m = json.load(f)

with mlflow.start_run(run_name="PCA"):
    mlflow.log_params({
        "algorithm": "PCA",
        "n_features": m["n_features"],
        "scaler": "StandardScaler",
        "recon_sample_size": m["sample_size"],
    })
    mlflow.log_metrics({
        "explained_var_2_comp": m["var_2_components"],
        "explained_var_3_comp": m["var_3_components"],
        "components_for_70pct": m["components_for_70pct"],
        "components_for_80pct": m["components_for_80pct"],
        **{f"recon_mse_{k}_comp": v for k, v in m["recon_mse"].items()},
    })
    log_files([
        "outputs/dimred/pca_scree.png",
        "outputs/dimred/pca_2d.png",
        "outputs/dimred/pca_feature_importance.png",
        "outputs/dimred/pca_feature_importance.csv",
        "outputs/dimred/pca_loadings.csv",
        "outputs/dimred/pca_2d_interactive.html",
        "outputs/dimred/pca_3d_interactive.html",
    ])

with mlflow.start_run(run_name="tSNE"):
    mlflow.log_params({
        "algorithm": "t-SNE",
        "n_components": 2,
        "perplexity": 30,
        "init": "pca",
        "sample_size": m["sample_size"],
    })
    mlflow.log_metric("kl_divergence", m["tsne_kl_divergence"])
    log_files(["outputs/dimred/tsne_2d.png", "outputs/dimred/tsne_interactive.html"])

if m.get("umap_done"):
    with mlflow.start_run(run_name="UMAP"):
        mlflow.log_params({
            "algorithm": "UMAP",
            "n_components": 2,
            "n_neighbors": 15,
            "min_dist": 0.1,
            "sample_size": m["sample_size"],
        })
        log_files(["outputs/dimred/umap_2d.png"])

# ===================================================
# Summary
# ===================================================
print("\n=== Best run per experiment (by silhouette) ===")
for exp in ["PatrolIQ_Geographic_Clustering", "PatrolIQ_Temporal_Clustering"]:
    runs = mlflow.search_runs(experiment_names=[exp],
                              order_by=["metrics.silhouette DESC"], max_results=1)
    if len(runs):
        r = runs.iloc[0]
        print(f"{exp}: {r['tags.mlflow.runName']} | silhouette={r['metrics.silhouette']:.4f}")

print("\nDone. MLflow UI open panna:")
print("  mlflow ui --backend-store-uri sqlite:///mlflow.db")
print("  Browser: http://127.0.0.1:5000")
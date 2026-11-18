import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

IN_PATH = "data/processed/crimes_features.csv"
OUT_DIR = "outputs/clustering"
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42

# ---------------------------------------------------
# 1. Load + coordinates ah km-ku convert pannu
# ---------------------------------------------------
df = pd.read_csv(
    IN_PATH,
    usecols=["id", "latitude", "longitude", "primary_type", "arrest",
             "crime_severity_score", "district"],
)
print("Loaded:", df.shape)

lat0 = df["latitude"].mean()
df["y_km"] = (df["latitude"] - lat0) * 111.0
df["x_km"] = (df["longitude"] - df["longitude"].mean()) * 111.0 * np.cos(np.radians(lat0))
X = df[["x_km", "y_km"]].values

results = []   # comparison table-ku


def evaluate(name, params, Xe, labels, sample=20000):
    """Silhouette + Davies-Bouldin (noise points -1 ignore pannum)"""
    mask = labels != -1
    Xm, lm = Xe[mask], labels[mask]
    n_clusters = len(set(lm))
    if n_clusters < 2:
        return None
    sil = silhouette_score(Xm, lm, sample_size=min(sample, len(Xm)), random_state=SEED)
    dbi = davies_bouldin_score(Xm, lm)
    noise_pct = (~mask).mean() * 100
    row = {"algorithm": name, "params": params, "clusters": n_clusters,
           "noise_%": round(noise_pct, 1),
           "silhouette": round(sil, 4), "davies_bouldin": round(dbi, 4)}
    results.append(row)
    return row


# ---------------------------------------------------
# 2. K-MEANS: elbow + silhouette (k = 3 to 10)
# ---------------------------------------------------
print("\n=== K-Means: k selection ===")
inertias, sils, dbis = [], [], []
K_RANGE = range(3, 11)
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    inertias.append(km.inertia_)
    s = silhouette_score(X, km.labels_, sample_size=20000, random_state=SEED)
    d = davies_bouldin_score(X, km.labels_)
    sils.append(s)
    dbis.append(d)
    print(f"k={k}: inertia={km.inertia_:,.0f} | silhouette={s:.4f} | DBI={d:.4f}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(list(K_RANGE), inertias, marker="o"); ax[0].set_title("Elbow (Inertia)")
ax[1].plot(list(K_RANGE), sils, marker="o", color="green"); ax[1].set_title("Silhouette (higher better)")
ax[2].plot(list(K_RANGE), dbis, marker="o", color="red"); ax[2].set_title("Davies-Bouldin (lower better)")
for a in ax:
    a.set_xlabel("k")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/kmeans_k_selection.png", dpi=120)
plt.close()

# k=9 : 5-10 hotspot requirement + nalla DBI (0.789) + silhouette 0.411
best_k = 9
print("\nSelected k:", best_k)

km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(X)
df["kmeans_cluster"] = km.labels_
evaluate("K-Means", f"k={best_k}", X, km.labels_)

# ---------------------------------------------------
# 3. DBSCAN (100K sample, parameter grid)
# ---------------------------------------------------
print("\n=== DBSCAN ===")
samp = df.sample(100_000, random_state=SEED).reset_index(drop=True)
Xs = samp[["x_km", "y_km"]].values

best_db, best_db_labels, best_db_sil = None, None, -1
for eps in [0.3, 0.5, 0.8]:
    for ms in [50, 100]:
        lab = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1).fit(Xs).labels_
        n_cl = len(set(lab)) - (1 if -1 in lab else 0)
        noise = (lab == -1).mean() * 100
        print(f"eps={eps} km, min_samples={ms}: clusters={n_cl}, noise={noise:.1f}%")
        if n_cl >= 2 and noise < 60:
            mask = lab != -1
            sil = silhouette_score(Xs[mask], lab[mask],
                                   sample_size=min(20000, mask.sum()), random_state=SEED)
            print(f"    silhouette={sil:.4f}")
            if sil > best_db_sil:
                best_db_sil, best_db, best_db_labels = sil, (eps, ms), lab

if best_db is not None:
    samp["dbscan_cluster"] = best_db_labels
    evaluate("DBSCAN", f"eps={best_db[0]}km, min_samples={best_db[1]}", Xs, best_db_labels)
    print("Best DBSCAN:", best_db)
else:
    print("DBSCAN la valid result illa. eps/min_samples maathi paakkanum.")

# ---------------------------------------------------
# 4. HIERARCHICAL (5000 sample, Ward linkage)
# ---------------------------------------------------
print("\n=== Hierarchical ===")
hs = df.sample(5000, random_state=SEED).reset_index(drop=True)
Xh = hs[["x_km", "y_km"]].values
Z = linkage(Xh, method="ward")

plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode="lastp", p=30, show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram (Ward, 5000 sample, last 30 merges)")
plt.xlabel("Cluster size")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/dendrogram.png", dpi=120)
plt.close()

h_labels = fcluster(Z, t=best_k, criterion="maxclust")
hs["hier_cluster"] = h_labels
evaluate("Hierarchical (Ward)", f"k={best_k}, sample=5000", Xh, h_labels)

# ---------------------------------------------------
# 5. Algorithm comparison
# ---------------------------------------------------
comp = pd.DataFrame(results)
print("\n=== Algorithm Comparison ===")
print(comp.to_string(index=False))
comp.to_csv(f"{OUT_DIR}/algorithm_comparison.csv", index=False)

# ---------------------------------------------------
# 6. K-Means cluster profile (total, dominant crime, arrest rate)
# ---------------------------------------------------
rows = []
for c, g in df.groupby("kmeans_cluster"):
    rows.append({
        "cluster": c,
        "total_crimes": len(g),
        "share_%": round(len(g) / len(df) * 100, 1),
        "center_lat": round(g["latitude"].mean(), 4),
        "center_lon": round(g["longitude"].mean(), 4),
        "dominant_crime": g["primary_type"].value_counts().idxmax(),
        "top3_crimes": ", ".join(g["primary_type"].value_counts().head(3).index),
        "arrest_rate_%": round(g["arrest"].mean() * 100, 1),
        "avg_severity": round(g["crime_severity_score"].mean(), 2),
    })
prof = pd.DataFrame(rows).sort_values("total_crimes", ascending=False)

# Risk level: crime count + severity vachi
prof["risk_score"] = (prof["total_crimes"] / prof["total_crimes"].max()) * 0.7 + \
                     (prof["avg_severity"] / prof["avg_severity"].max()) * 0.3
prof["risk_level"] = pd.qcut(prof["risk_score"].rank(method="first"), 3,
                             labels=["Low", "Medium", "High"])
print("\n=== K-Means Cluster Profile ===")
print(prof.drop(columns=["top3_crimes"]).to_string(index=False))
prof.to_csv(f"{OUT_DIR}/kmeans_cluster_profile.csv", index=False)

# ---------------------------------------------------
# 7. Maps
# ---------------------------------------------------
vis = df.sample(100_000, random_state=SEED)
plt.figure(figsize=(7, 9))
plt.scatter(vis["longitude"], vis["latitude"], c=vis["kmeans_cluster"],
            cmap="tab10", s=1, alpha=0.4)
plt.scatter(prof["center_lon"], prof["center_lat"], c="black", marker="X", s=150)
plt.title(f"K-Means Crime Hotspots (k={best_k})")
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/kmeans_map.png", dpi=120)
plt.close()

if best_db is not None:
    plt.figure(figsize=(7, 9))
    noise = samp[samp["dbscan_cluster"] == -1]
    core = samp[samp["dbscan_cluster"] != -1]
    plt.scatter(noise["longitude"], noise["latitude"], c="lightgrey", s=1, label="Noise")
    plt.scatter(core["longitude"], core["latitude"], c=core["dbscan_cluster"],
                cmap="tab20", s=1)
    plt.title(f"DBSCAN Hotspots (eps={best_db[0]}km, min_samples={best_db[1]})")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/dbscan_map.png", dpi=120)
    plt.close()

# ---------------------------------------------------
# 8. Cluster labels save (Streamlit + MLflow ku)
# ---------------------------------------------------
df[["id", "kmeans_cluster"]].to_csv("data/processed/kmeans_labels.csv", index=False)
print("\nAll outputs saved in:", OUT_DIR)
print("Best algorithm by silhouette:",
      comp.sort_values("silhouette", ascending=False).iloc[0]["algorithm"])
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

IN_PATH = "data/processed/crimes_features.csv"
OUT_DIR = "outputs/dimred"
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42
SAMPLE_N = 10_000   # t-SNE-ku (full data-la romba slow)
sns.set_theme(style="whitegrid")

# ---------------------------------------------------
# 1. Load + feature set (25 features)
# ---------------------------------------------------
df = pd.read_csv(IN_PATH, low_memory=False)
print("Loaded:", df.shape)

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week_num"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week_num"] / 7)
df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

FEATURES = [
    # geographic / administrative
    "latitude", "longitude", "x_coordinate", "y_coordinate",
    "beat", "district", "ward", "community_area",
    # temporal
    "hour", "hour_sin", "hour_cos", "day_of_week_num", "dow_sin", "dow_cos",
    "month", "month_sin", "month_cos", "year", "is_weekend", "season_encoded",
    # crime
    "crime_severity_score", "primary_type_encoded", "location_encoded",
    "arrest", "domestic",
]
print("Number of features:", len(FEATURES))

# Cluster labels merge (plots-la color panna)
km = pd.read_csv("data/processed/kmeans_labels.csv")
df = df.merge(km, on="id", how="left")
tl = pd.read_csv("data/processed/temporal_labels.csv", usecols=["id", "time_cluster"])
df = df.merge(tl, on="id", how="left")

# Crime type groups (top 6 + OTHER)
top6 = df["primary_type"].value_counts().head(6).index
df["crime_group"] = df["primary_type"].where(df["primary_type"].isin(top6), "OTHER")

X = StandardScaler().fit_transform(df[FEATURES].values)

# ---------------------------------------------------
# 2. PCA (full) + explained variance
# ---------------------------------------------------
pca = PCA(random_state=SEED).fit(X)
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)
n70 = int(np.argmax(cum >= 0.70)) + 1
n80 = int(np.argmax(cum >= 0.80)) + 1

print("\n=== PCA Explained Variance ===")
for i in range(10):
    print(f"PC{i+1}: {evr[i]*100:5.2f}%  | cumulative {cum[i]*100:5.2f}%")
print(f"\n2 components : {cum[1]*100:.1f}% variance")
print(f"3 components : {cum[2]*100:.1f}% variance")
print(f"Components needed for 70% variance: {n70}")
print(f"Components needed for 80% variance: {n80}")

# Scree plot
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].bar(range(1, len(evr) + 1), evr * 100, color="#1f77b4")
ax[0].set_title("Scree Plot")
ax[0].set_xlabel("Principal Component"); ax[0].set_ylabel("Explained Variance %")
ax[1].plot(range(1, len(cum) + 1), cum * 100, marker="o", color="#d62728")
ax[1].axhline(70, ls="--", color="grey"); ax[1].axhline(80, ls="--", color="grey")
ax[1].set_title("Cumulative Explained Variance")
ax[1].set_xlabel("Number of Components"); ax[1].set_ylabel("Cumulative %")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/pca_scree.png", dpi=120)
plt.close()

# ---------------------------------------------------
# 3. Feature importance (loadings, variance weighted)
# ---------------------------------------------------
imp = (np.abs(pca.components_[:n70]) * evr[:n70, None]).sum(axis=0)
imp = imp / imp.sum() * 100
imp_df = (pd.DataFrame({"feature": FEATURES, "importance_%": imp})
          .sort_values("importance_%", ascending=False).reset_index(drop=True))
print(f"\n=== Top 10 Important Features (first {n70} PCs) ===")
print(imp_df.head(10).round(2).to_string(index=False))
imp_df.to_csv(f"{OUT_DIR}/pca_feature_importance.csv", index=False)

plt.figure(figsize=(9, 7))
sns.barplot(data=imp_df.head(15), x="importance_%", y="feature", color="#2ca02c")
plt.title("Top 15 Features Driving Variance (PCA loadings)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/pca_feature_importance.png", dpi=120)
plt.close()

# PC1, PC2 loadings
load = pd.DataFrame(pca.components_[:3].T, index=FEATURES, columns=["PC1", "PC2", "PC3"])
load.to_csv(f"{OUT_DIR}/pca_loadings.csv")
print("\nTop |loading| features:")
for pc in ["PC1", "PC2"]:
    print(pc, ":", load[pc].abs().sort_values(ascending=False).head(4).index.tolist())

# ---------------------------------------------------
# 4. Reconstruction error
# ---------------------------------------------------
idx = np.random.RandomState(SEED).choice(len(df), SAMPLE_N, replace=False)
Xs = X[idx]
S = df.iloc[idx].reset_index(drop=True)
Z = pca.transform(Xs)

recon = {}
for k in [2, 3, n70, n80]:
    rec = Z[:, :k] @ pca.components_[:k] + pca.mean_
    recon[k] = float(np.mean((Xs - rec) ** 2))
    print(f"Reconstruction MSE ({k} comps): {recon[k]:.4f}")

# ---------------------------------------------------
# 5. PCA 2D / 3D plots
# ---------------------------------------------------
S["pca1"], S["pca2"], S["pca3"] = Z[:, 0], Z[:, 1], Z[:, 2]

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
sns.scatterplot(data=S, x="pca1", y="pca2", hue="crime_group", s=6, alpha=0.5, ax=ax[0])
ax[0].set_title("PCA 2D - colored by Crime Type")
sns.scatterplot(data=S, x="pca1", y="pca2", hue="time_of_day", s=6, alpha=0.5, ax=ax[1])
ax[1].set_title("PCA 2D - colored by Time of Day")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/pca_2d.png", dpi=120)
plt.close()

hover = ["primary_type", "hour", "district", "season"]
px.scatter(S, x="pca1", y="pca2", color="crime_group", hover_data=hover,
           title="PCA 2D - Crime Patterns", opacity=0.6
           ).write_html(f"{OUT_DIR}/pca_2d_interactive.html")
px.scatter_3d(S, x="pca1", y="pca2", z="pca3", color="crime_group",
              hover_data=hover, title="PCA 3D - Crime Patterns", opacity=0.6
              ).write_html(f"{OUT_DIR}/pca_3d_interactive.html")

# ---------------------------------------------------
# 6. t-SNE (10K sample)
# ---------------------------------------------------
print("\n=== t-SNE running (2-4 minutes) ===")
tsne = TSNE(n_components=2, perplexity=30, init="pca",
            learning_rate="auto", random_state=SEED)
T = tsne.fit_transform(Xs)
S["tsne1"], S["tsne2"] = T[:, 0], T[:, 1]
print("t-SNE KL divergence:", round(float(tsne.kl_divergence_), 4))

fig, ax = plt.subplots(1, 3, figsize=(21, 6))
sns.scatterplot(data=S, x="tsne1", y="tsne2", hue="crime_group", s=6, alpha=0.6, ax=ax[0])
ax[0].set_title("t-SNE - Crime Type")
sns.scatterplot(data=S, x="tsne1", y="tsne2", hue="time_of_day", s=6, alpha=0.6, ax=ax[1])
ax[1].set_title("t-SNE - Time of Day")
sns.scatterplot(data=S, x="tsne1", y="tsne2", hue="kmeans_cluster",
                palette="tab10", s=6, alpha=0.6, ax=ax[2])
ax[2].set_title("t-SNE - Geographic K-Means Cluster")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/tsne_2d.png", dpi=120)
plt.close()

px.scatter(S, x="tsne1", y="tsne2", color="crime_group", hover_data=hover,
           title="t-SNE - Crime Patterns", opacity=0.6
           ).write_html(f"{OUT_DIR}/tsne_interactive.html")

# ---------------------------------------------------
# 7. UMAP (optional - install aagala-na skip aagum)
# ---------------------------------------------------
umap_ok = False
try:
    import umap
    print("\n=== UMAP running ===")
    U = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                  random_state=SEED).fit_transform(Xs)
    S["umap1"], S["umap2"] = U[:, 0], U[:, 1]
    umap_ok = True

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.scatterplot(data=S, x="umap1", y="umap2", hue="crime_group", s=6, alpha=0.6, ax=ax[0])
    ax[0].set_title("UMAP - Crime Type")
    sns.scatterplot(data=S, x="umap1", y="umap2", hue="time_of_day", s=6, alpha=0.6, ax=ax[1])
    ax[1].set_title("UMAP - Time of Day")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/umap_2d.png", dpi=120)
    plt.close()
except Exception as e:
    print("UMAP skip aachu:", type(e).__name__, "-", str(e)[:120])

# ---------------------------------------------------
# 8. Save embeddings + metrics (Streamlit + MLflow ku)
# ---------------------------------------------------
cols = ["id", "primary_type", "crime_group", "hour", "district", "season",
        "time_of_day", "kmeans_cluster", "time_cluster",
        "pca1", "pca2", "pca3", "tsne1", "tsne2"]
if umap_ok:
    cols += ["umap1", "umap2"]
S[cols].to_csv("data/processed/dimred_embeddings.csv", index=False)

metrics = {
    "n_features": len(FEATURES),
    "var_2_components": float(cum[1]),
    "var_3_components": float(cum[2]),
    "components_for_70pct": n70,
    "components_for_80pct": n80,
    "recon_mse": {str(k): v for k, v in recon.items()},
    "tsne_kl_divergence": float(tsne.kl_divergence_),
    "sample_size": SAMPLE_N,
    "umap_done": umap_ok,
}
with open(f"{OUT_DIR}/dimred_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nAll outputs saved in:", OUT_DIR)
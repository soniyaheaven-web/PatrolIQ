import plotly.express as px
import streamlit as st

from utils import exists, load_csv, load_json, show_image

st.set_page_config(page_title="Dimensionality Reduction | PatrolIQ", page_icon="🔬", layout="wide")
st.title("🔬 Dimensionality Reduction")
st.caption("PCA, t-SNE and UMAP on 25 engineered features (10K sample for t-SNE/UMAP)")

m = load_json("outputs", "dimred", "dimred_metrics.json")
k = st.columns(4)
k[0].metric("Features", m["n_features"])
k[1].metric("Variance in 2 PCs", f"{m['var_2_components'] * 100:.1f}%")
k[2].metric("Variance in 3 PCs", f"{m['var_3_components'] * 100:.1f}%")
k[3].metric("PCs for 70% / 80%", f"{m['components_for_70pct']} / {m['components_for_80pct']}")
st.info("2-3 components cover only 31-41% variance here, so 9 components are needed for 70%. "
        "t-SNE and UMAP are used for the 2D visualisation.")

t1, t2, t3 = st.tabs(["PCA analysis", "Interactive embeddings", "3D PCA"])

with t1:
    a, b = st.columns(2)
    with a:
        show_image(("outputs", "dimred", "pca_scree.png"), caption="Scree plot and cumulative variance")
    with b:
        imp = load_csv("outputs", "dimred", "pca_feature_importance.csv")
        top = imp.head(15).sort_values("importance_%")
        st.plotly_chart(px.bar(top, x="importance_%", y="feature", orientation="h",
                               title="Top 15 features (PCA loadings, variance-weighted)"))
    st.write("**Reconstruction MSE:** " +
             ", ".join(f"{k_} comps = {v:.3f}" for k_, v in m["recon_mse"].items()))
    st.caption("primary_type_encoded uses label encoding (arbitrary order), so don't over-interpret "
               "its loading. Latitude/longitude and x/y coordinates carry similar information, "
               "so geography is counted twice.")

emb = load_csv("data", "app", "dimred_embeddings.csv")
for c in ["kmeans_cluster", "time_cluster", "district"]:
    emb[c] = emb[c].astype(str)

with t2:
    methods = {"PCA": ("pca1", "pca2"), "t-SNE": ("tsne1", "tsne2")}
    if "umap1" in emb.columns:
        methods["UMAP"] = ("umap1", "umap2")
    c1, c2, c3 = st.columns(3)
    method = c1.radio("Method", list(methods), horizontal=True)
    color_by = c2.selectbox("Color by", ["crime_group", "time_of_day", "kmeans_cluster",
                                         "time_cluster", "district", "season"])
    groups = sorted(emb["crime_group"].unique())
    sel = c3.multiselect("Crime groups", groups, default=groups)
    d = emb[emb["crime_group"].isin(sel)]
    x, y = methods[method]
    fig = px.scatter(d, x=x, y=y, color=color_by, opacity=0.6, height=650,
                     hover_data=["primary_type", "hour", "district", "season"],
                     title=f"{method} projection, colored by {color_by}")
    fig.update_traces(marker=dict(size=4))
    st.plotly_chart(fig)
    st.caption("PCA is linear. t-SNE/UMAP show local structure. Geographic zones separate "
               "clearly, while crime types overlap a lot because crime type is only one of 25 features.")

with t3:
    sel3 = st.selectbox("Color by (3D)", ["crime_group", "time_of_day", "kmeans_cluster"], key="c3d")
    fig = px.scatter_3d(emb, x="pca1", y="pca2", z="pca3", color=sel3, opacity=0.6, height=700)
    fig.update_traces(marker=dict(size=2.5))
    st.plotly_chart(fig)

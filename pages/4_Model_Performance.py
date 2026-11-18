import pandas as pd
import plotly.express as px
import streamlit as st

from utils import exists, load_csv, load_json

st.set_page_config(page_title="Model Performance | PatrolIQ", page_icon="📊", layout="wide")
st.title("📊 Model Performance (MLflow)")

if not exists("data", "app", "mlflow_runs.csv"):
    st.error("data/app/mlflow_runs.csv illa. Munnadi `python build_app_data.py` run pannunga.")
    st.stop()


def short(c):
    if c == "tags.mlflow.runName":
        return "run_name"
    for pre, rep in (("params.", "p_"), ("metrics.", "m_"), ("tags.", "t_")):
        if c.startswith(pre):
            return rep + c[len(pre):]
    return c


runs = load_csv("data", "app", "mlflow_runs.csv").rename(columns=short)
for c in ["p_n_clusters", "p_eps_km", "p_min_samples"]:
    if c in runs.columns:
        runs[c] = pd.to_numeric(runs[c], errors="coerce")

GEO = "PatrolIQ_Geographic_Clustering"
TMP = "PatrolIQ_Temporal_Clustering"
DIM = "PatrolIQ_Dimensionality_Reduction"
geo, tmp, dim = (runs[runs["experiment"] == e] for e in (GEO, TMP, DIM))

k = st.columns(3)
k[0].metric("Experiments", runs["experiment"].nunique())
k[1].metric("Total runs", len(runs))
k[2].metric("Final geographic model", "K-Means k=9")

t1, t2, t3, t4 = st.tabs(["Geographic", "Temporal", "Dimensionality reduction", "Model registry"])


def show_table(d):
    cols = [c for c in ["run_name", "p_algorithm", "p_n_clusters", "p_eps_km", "p_min_samples",
                        "m_silhouette", "m_davies_bouldin", "m_noise_pct", "m_inertia",
                        "t_selected"] if c in d.columns]
    st.dataframe(d[cols].sort_values("run_name"), hide_index=True)


with t1:
    km = geo[geo["p_algorithm"] == "KMeans"].sort_values("p_n_clusters")
    a, b = st.columns(2)
    fig = px.line(km, x="p_n_clusters", y="m_silhouette", markers=True,
                  title="K-Means silhouette vs k (higher is better)")
    fig.add_hline(y=0.5, line_dash="dash", annotation_text="Target 0.5", annotation_position="bottom right")
    fig.add_vline(x=9, line_dash="dot", annotation_text="Selected k=9", annotation_position="top left")
    a.plotly_chart(fig)
    fig = px.line(km, x="p_n_clusters", y="m_davies_bouldin", markers=True,
                  title="K-Means Davies-Bouldin vs k (lower is better)")
    fig.add_vline(x=9, line_dash="dot", annotation_text="Selected k=9")
    b.plotly_chart(fig)
    st.caption("k=3 has slightly higher silhouette (0.42), but k=9 has the lowest Davies-Bouldin "
               "(0.79) and fits the 5-10 hotspot requirement.")
    show_table(geo)

with t2:
    tk = tmp.sort_values("p_n_clusters")
    fig = px.line(tk, x="p_n_clusters", y="m_silhouette", markers=True,
                  title="Temporal K-Means silhouette vs k")
    st.plotly_chart(fig)
    show_table(tmp)

with t3:
    for _, r in dim.iterrows():
        d = {c: (v.item() if hasattr(v, "item") else v) for c, v in r.items()
             if (c.startswith("p_") or c.startswith("m_")) and pd.notna(v)}
        with st.expander(str(r["run_name"]), expanded=(r["run_name"] == "PCA")):
            st.json(d)

with t4:
    if exists("data", "app", "model_registry.json"):
        reg = load_json("data", "app", "model_registry.json")
        if reg:
            st.dataframe(pd.DataFrame(reg), hide_index=True)
        else:
            st.info("Registry empty.")
    else:
        st.info("model_registry.json illa.")
    st.code("mlflow ui --backend-store-uri sqlite:///mlflow.db", language="bash")
    st.caption("Full MLflow UI locally run pannalam. Idha deployed app-la runs export-ah paakkudhu.")

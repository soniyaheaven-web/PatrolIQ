import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (CHICAGO_CENTER, MAP_STYLE, RISK_COLORS, load_crimes,
                   load_csv, show_image)

st.set_page_config(page_title="Crime Hotspots | PatrolIQ", page_icon="📍", layout="wide")
st.title("📍 Geographic Crime Hotspots")
st.caption("K-Means (k=9) hotspot zones on Chicago crime data")

df = load_crimes()
prof = load_csv("outputs", "clustering", "kmeans_cluster_profile.csv")
risk_map = dict(zip(prof["cluster"], prof["risk_level"]))
df["risk_level"] = df["kmeans_cluster"].map(risk_map)

with st.sidebar:
    st.header("Filters")
    years = sorted(df["year"].unique())
    sel_years = st.multiselect("Year", years, default=years)
    types = df["primary_type"].value_counts().index.tolist()
    sel_types = st.multiselect("Crime type", types, default=[], placeholder="All crime types")
    sel_risk = st.multiselect("Risk level", ["High", "Medium", "Low"],
                              default=["High", "Medium", "Low"])
    max_pts = st.slider("Max points on map", 5000, 50000, 20000, step=5000)

f = df[df["year"].isin(sel_years) & df["risk_level"].isin(sel_risk)]
if sel_types:
    f = f[f["primary_type"].isin(sel_types)]
if f.empty:
    st.warning("Intha filters-ku data illa. Filters-ah maathunga.")
    st.stop()

k = st.columns(4)
k[0].metric("Crimes shown", f"{len(f):,}")
k[1].metric("Zones", f["kmeans_cluster"].nunique())
k[2].metric("Arrest rate", f"{f['arrest'].mean() * 100:.1f}%")
k[3].metric("Avg severity", f"{f['crime_severity_score'].mean():.2f}")

s = f.sample(min(len(f), max_pts), random_state=42)
map_args = dict(lat="latitude", lon="longitude", zoom=9.2, center=CHICAGO_CENTER,
                map_style=MAP_STYLE, height=650)

tab1, tab2, tab3 = st.tabs(["🔥 Heatmap", "🧭 Clusters", "🚦 Risk zones"])

with tab1:
    fig = px.density_map(s, radius=8, color_continuous_scale="YlOrRd", **map_args)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig)

with tab2:
    s2 = s.assign(cluster=s["kmeans_cluster"].astype(str))
    fig = px.scatter_map(s2, color="cluster", opacity=0.6,
                         hover_data=["primary_type", "district", "hour"], **map_args)
    fig.add_trace(go.Scattermap(
        lat=prof["center_lat"], lon=prof["center_lon"], mode="markers+text",
        text=[f"Z{c}" for c in prof["cluster"]], textposition="top right",
        marker=dict(size=14, color="black"), name="Zone centers"))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig)

with tab3:
    fig = px.scatter_map(s, color="risk_level", color_discrete_map=RISK_COLORS,
                         category_orders={"risk_level": ["High", "Medium", "Low"]},
                         opacity=0.55, hover_data=["primary_type", "district"], **map_args)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig)
    st.caption("Risk level = 70% crime volume + 30% average severity, then split into "
               "High / Medium / Low tiers across the 9 zones.")

st.subheader("Zone statistics (current filters)")
stats = (f.groupby("kmeans_cluster")
         .agg(crimes=("id", "size"),
              dominant_crime=("primary_type", lambda x: x.value_counts().idxmax()),
              arrest_rate=("arrest", "mean"),
              avg_severity=("crime_severity_score", "mean"))
         .reset_index())
stats["arrest_rate"] = (stats["arrest_rate"] * 100).round(1)
stats["avg_severity"] = stats["avg_severity"].round(2)
stats["risk_level"] = stats["kmeans_cluster"].map(risk_map)
stats = stats.sort_values("crimes", ascending=False).rename(columns={
    "kmeans_cluster": "zone", "arrest_rate": "arrest_rate_%"})
st.dataframe(stats, hide_index=True)

with st.expander("Full-data zone profile (492,564 records)"):
    st.dataframe(prof[["cluster", "total_crimes", "share_%", "center_lat", "center_lon",
                       "dominant_crime", "top3_crimes", "arrest_rate_%", "avg_severity",
                       "risk_level"]], hide_index=True)

st.divider()
st.subheader("Algorithm comparison")
comp = load_csv("outputs", "clustering", "algorithm_comparison.csv")
st.dataframe(comp, hide_index=True)
fig = px.bar(comp, x="algorithm", y="silhouette", color="algorithm", text="silhouette",
             title="Silhouette score by algorithm (higher is better)")
fig.add_hline(y=0.5, line_dash="dash", annotation_text="Project target 0.5")
st.plotly_chart(fig)
st.info(
    "K-Means (k=9) selected: best silhouette (0.41) and lowest Davies-Bouldin (0.79) among "
    "the 3 algorithms, and 9 zones fit the 5-10 hotspot requirement. The 0.5 target was not "
    "reached because Chicago crimes are spread continuously across the city. DBSCAN's silhouette "
    "is low, but it is useful for detecting dense areas and noise."
)
with st.expander("Elbow / silhouette / DBI by k"):
    show_image(("outputs", "clustering", "kmeans_k_selection.png"))
with st.expander("Hierarchical dendrogram"):
    show_image(("outputs", "clustering", "dendrogram.png"))
with st.expander("DBSCAN map"):
    show_image(("outputs", "clustering", "dbscan_map.png"))

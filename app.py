import streamlit as st

from utils import load_crimes

st.set_page_config(page_title="PatrolIQ", page_icon="🚓", layout="wide")

st.title("🚓 PatrolIQ: Smart Safety Analytics Platform")
st.caption("Unsupervised ML on Chicago crime data to support patrol planning")

df = load_crimes()

c = st.columns(5)
c[0].metric("Records analysed", "492,564")
c[1].metric("Records in this app", f"{len(df):,}")
c[2].metric("Crime types", df["primary_type"].nunique())
c[3].metric("Arrest rate", f"{df['arrest'].mean() * 100:.1f}%")
c[4].metric("Avg severity (1-10)", f"{df['crime_severity_score'].mean():.2f}")
st.caption(
    f"Date range: {df['date'].min():%d %b %Y} to {df['date'].max():%d %b %Y}. "
    "Full analysis 492,564 records-la nadandhadhu. App fast-ah load aaga, "
    "adhula irundhu 200K random subset use aagudhu."
)

st.divider()
left, right = st.columns(2)

with left:
    st.subheader("What you can explore")
    st.markdown(
        """
- **📍 Crime Hotspots:** K-Means zones, heatmap, risk levels, algorithm comparison
- **⏰ Temporal Patterns:** hour, day, month, season trends and time-based clusters
- **🔬 Dimensionality Reduction:** PCA, t-SNE, UMAP interactive plots
- **📊 Model Performance:** MLflow experiments, runs and model registry
- **🚓 Patrol Planner:** "Where should we patrol tonight?" for a chosen day and hour
        """
    )

with right:
    st.subheader("Key findings")
    st.markdown(
        """
- **Geographic:** K-Means with k=9 gave silhouette 0.41 and Davies-Bouldin 0.79. The 0.5 target was not reached because crimes are spread continuously across the city.
- **Temporal:** 4 time-based patterns (afternoon vs late night, warm vs cold months), silhouette 0.15, so they work as profiles, not hard boundaries.
- **Dimensionality:** 9 PCA components are needed for 70% variance. 2D/3D PCA captures 31.6% / 41.5%, so t-SNE and UMAP are used for visualisation.
- **Data note:** the midnight (hour 0) spike is likely a default timestamp for records with unknown time.
        """
    )

st.divider()
st.subheader("Pipeline")
st.markdown(
    "`Chicago Data Portal (2021-2026)` → `500K sample` → `Cleaning + Feature engineering` → "
    "`Geographic + Temporal clustering` → `PCA / t-SNE / UMAP` → `MLflow tracking` → `Streamlit`"
)
st.info("Left sidebar-la irundhu pages-ah select pannunga.")

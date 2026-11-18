import numpy as np
import plotly.express as px
import streamlit as st

from utils import (CHICAGO_CENTER, DAY_ORDER, MAP_STYLE, load_crimes, load_csv)

st.set_page_config(page_title="Patrol Planner | PatrolIQ", page_icon="🚓", layout="wide")
st.title("🚓 Patrol Planner")
st.caption("Where should we patrol? Pick a day and time to rank the hotspot zones.")

df = load_crimes()
prof = load_csv("outputs", "clustering", "kmeans_cluster_profile.csv")

c1, c2, c3, c4 = st.columns(4)
day = c1.selectbox("Day of week", DAY_ORDER, index=4)
hour = c2.slider("Hour (24h)", 0, 23, 22)
window = c3.selectbox("Time window", ["Exact hour", "±1 hour", "±2 hours"], index=1)
season = c4.selectbox("Season", ["Any", "Winter", "Spring", "Summer", "Fall"])

w = {"Exact hour": 0, "±1 hour": 1, "±2 hours": 2}[window]
hours = [(hour + d) % 24 for d in range(-w, w + 1)]

slot = df[(df["day_of_week"] == day) & (df["hour"].isin(hours))]
if season != "Any":
    slot = slot[slot["season"] == season]
if slot.empty:
    st.warning("Intha slot-ku data illa.")
    st.stop()

z = (slot.groupby("kmeans_cluster")
     .agg(crimes=("id", "size"),
          dominant_crime=("primary_type", lambda x: x.value_counts().idxmax()),
          avg_severity=("crime_severity_score", "mean"),
          arrest_rate=("arrest", "mean"))
     .reset_index())
overall = df["kmeans_cluster"].value_counts(normalize=True)
z["share_%"] = z["crimes"] / z["crimes"].sum() * 100
z["usual_share_%"] = z["kmeans_cluster"].map(overall) * 100
z["vs_usual"] = z["share_%"] / z["usual_share_%"]
z["priority_score"] = (z["crimes"] / z["crimes"].max()) * 0.7 + \
                      (z["avg_severity"] / z["avg_severity"].max()) * 0.3
z["arrest_rate"] = z["arrest_rate"] * 100
zones = prof[["cluster", "center_lat", "center_lon", "risk_level"]].rename(
    columns={"cluster": "kmeans_cluster"})
z = z.merge(zones, on="kmeans_cluster").sort_values("priority_score", ascending=False)

st.subheader(f"Recommended zones: {day}, {', '.join(f'{h}:00' for h in hours)}")
for rank, (_, r) in enumerate(z.head(3).iterrows(), 1):
    st.markdown(
        f"**#{rank} Zone {int(r['kmeans_cluster'])}** ({r['risk_level']} risk overall): "
        f"{int(r['crimes'])} crimes in this slot, mostly {r['dominant_crime'].title()}, "
        f"avg severity {r['avg_severity']:.2f}. Center near "
        f"({r['center_lat']:.3f}, {r['center_lon']:.3f})."
    )

fig = px.scatter_map(z, lat="center_lat", lon="center_lon", size="crimes",
                     color="priority_score", color_continuous_scale="YlOrRd",
                     hover_name=z["kmeans_cluster"].map(lambda c: f"Zone {c}"),
                     hover_data={"crimes": True, "dominant_crime": True,
                                 "avg_severity": ":.2f", "center_lat": False,
                                 "center_lon": False},
                     size_max=45, zoom=9.2, center=CHICAGO_CENTER,
                     map_style=MAP_STYLE, height=550)
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig)

show = z[["kmeans_cluster", "crimes", "share_%", "usual_share_%", "vs_usual",
          "dominant_crime", "avg_severity", "arrest_rate", "risk_level", "priority_score"]].round(2)
st.dataframe(show.rename(columns={"kmeans_cluster": "zone", "arrest_rate": "arrest_rate_%"}),
             hide_index=True)
st.caption("Counts come from the 200K app sample. priority_score = 70% crime count + 30% "
           "average severity within this slot. vs_usual above 1 means the zone has more than "
           "its normal share of crimes at this time.")

st.divider()
st.subheader("Which zone is a location in?")
a, b = st.columns(2)
lat = a.number_input("Latitude", value=41.8781, format="%.4f")
lon = b.number_input("Longitude", value=-87.6298, format="%.4f")
dx = (lon - prof["center_lon"]) * np.cos(np.radians(lat))
dy = lat - prof["center_lat"]
near = prof.iloc[int(np.argmin(np.sqrt(dx ** 2 + dy ** 2)))]
st.success(f"Nearest hotspot: **Zone {int(near['cluster'])}** ({near['risk_level']} risk), "
           f"dominant crime {near['dominant_crime'].title()}.")

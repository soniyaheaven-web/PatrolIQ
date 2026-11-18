import plotly.express as px
import streamlit as st

from utils import DAY_ORDER, MONTH_NAMES, load_crimes, load_csv

st.set_page_config(page_title="Temporal Patterns | PatrolIQ", page_icon="⏰", layout="wide")
st.title("⏰ Temporal Crime Patterns")

df = load_crimes()

with st.sidebar:
    st.header("Filters")
    years = sorted(df["year"].unique())
    sel_years = st.multiselect("Year", years, default=years)
    types = df["primary_type"].value_counts().index.tolist()
    sel_types = st.multiselect("Crime type", types, default=[], placeholder="All crime types")
    excl0 = st.checkbox("Exclude hour 0 (midnight)", value=False,
                        help="Hour 0 spike probably default timestamp for unknown-time records.")

f = df[df["year"].isin(sel_years)]
if sel_types:
    f = f[f["primary_type"].isin(sel_types)]
if f.empty:
    st.warning("Intha filters-ku data illa.")
    st.stop()

hourly = f["hour"].value_counts().reindex(range(24), fill_value=0)
hh = hourly.rename_axis("hour").reset_index(name="crimes")
if excl0:
    hh = hh[hh["hour"] != 0]
peak_hour = int(hh.loc[hh["crimes"].idxmax(), "hour"])
dow = f["day_of_week"].value_counts().reindex(DAY_ORDER, fill_value=0)

full = f[f["year"] <= 2025]
mshare = (full["month"].value_counts(normalize=True)
          .reindex(range(1, 13), fill_value=0) * 100) if len(full) else None

k = st.columns(4)
k[0].metric("Peak hour", f"{peak_hour}:00")
k[1].metric("Peak day", dow.idxmax())
k[2].metric("Peak month", MONTH_NAMES[int(mshare.idxmax()) - 1] if mshare is not None else "n/a")
k[3].metric("Weekend share", f"{f['is_weekend'].mean() * 100:.1f}%")

t1, t2, t3, t4, t5 = st.tabs(["Hour & Day", "Heatmap", "Seasonal", "Weekday vs Weekend",
                              "Temporal clusters"])

with t1:
    a, b = st.columns(2)
    hh["note"] = hh["hour"].map(lambda x: "Hour 0 (possible default time)" if x == 0 else "Regular hours")
    fig = px.bar(hh, x="hour", y="crimes", color="note", title="Crimes by hour",
                 color_discrete_map={"Hour 0 (possible default time)": "#d62728",
                                     "Regular hours": "#1f77b4"})
    a.plotly_chart(fig)
    fig = px.bar(x=dow.index, y=dow.values, labels={"x": "Day", "y": "Crimes"},
                 title="Crimes by day of week")
    b.plotly_chart(fig)
    st.subheader("Top time slots")
    slots = f.groupby(["day_of_week", "hour"]).size().rename("crimes").reset_index()
    if excl0:
        slots = slots[slots["hour"] != 0]
    st.dataframe(slots.sort_values("crimes", ascending=False).head(10), hide_index=True)
    sev = f[f["crime_severity_score"] >= 7]["hour"].value_counts().head(5)
    st.write("**Top hours for severe crimes (severity 7+):** " +
             ", ".join(f"{int(h)}:00" for h in sev.index))

with t2:
    pivot = f.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0).reindex(DAY_ORDER)
    if excl0:
        pivot = pivot.drop(columns=0, errors="ignore")
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="YlOrRd",
                    labels=dict(x="Hour", y="Day", color="Crimes"),
                    title="Crime heatmap: day vs hour")
    st.plotly_chart(fig)

with t3:
    a, b = st.columns(2)
    if mshare is not None:
        fig = px.bar(x=MONTH_NAMES, y=mshare.values,
                     labels={"x": "Month", "y": "% of yearly crimes"},
                     title="Monthly share of crimes (2021-2025)")
        a.plotly_chart(fig)
    else:
        a.info("Monthly view-ku 2021-2025 years select pannunga.")
    season = (f["season"].value_counts(normalize=True)
              .reindex(["Winter", "Spring", "Summer", "Fall"], fill_value=0) * 100)
    fig = px.bar(x=season.index, y=season.values, color=season.index,
                 labels={"x": "Season", "y": "% of crimes"}, title="Seasonal share")
    b.plotly_chart(fig)

with t4:
    wk = f.groupby(["is_weekend", "hour"]).size().rename("n").reset_index()
    wk["pct"] = wk["n"] / wk.groupby("is_weekend")["n"].transform("sum") * 100
    wk["type"] = wk["is_weekend"].map({0: "Weekday", 1: "Weekend"})
    if excl0:
        wk = wk[wk["hour"] != 0]
    fig = px.line(wk, x="hour", y="pct", color="type", markers=True,
                  labels={"pct": "% of crimes"}, title="Hourly pattern: weekday vs weekend")
    st.plotly_chart(fig)

with t5:
    tp = load_csv("data", "app", "temporal_profile.csv")
    st.dataframe(tp[["cluster", "label", "share_%", "avg_hour", "top3_hours", "weekend_%",
                     "top3_months", "dominant_crime", "arrest_rate_%", "avg_severity"]],
                 hide_index=True)
    ch = f.groupby(["time_cluster_label", "hour"]).size().rename("n").reset_index()
    ch["pct"] = ch["n"] / ch.groupby("time_cluster_label")["n"].transform("sum") * 100
    st.plotly_chart(px.line(ch, x="hour", y="pct", color="time_cluster_label", markers=True,
                            labels={"pct": "% of cluster crimes", "time_cluster_label": "Cluster"},
                            title="Hourly profile of each temporal cluster"))
    cm = f.groupby(["month", "time_cluster_label"]).size().rename("n").reset_index()
    cm["pct"] = cm["n"] / cm.groupby("month")["n"].transform("sum") * 100
    st.plotly_chart(px.bar(cm, x="month", y="pct", color="time_cluster_label",
                           labels={"pct": "% of month crimes", "time_cluster_label": "Cluster"},
                           title="Temporal cluster mix by month"))
    st.info("Temporal K-Means silhouette is 0.15 (k=4). Hour/day/month are cyclical with no "
            "natural gaps, so treat these clusters as time profiles, not hard boundaries.")

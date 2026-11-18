import json
import os

import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
RISK_COLORS = {"High": "#d62728", "Medium": "#f2c200", "Low": "#2ca02c"}
CHICAGO_CENTER = {"lat": 41.84, "lon": -87.68}
MAP_STYLE = "open-street-map"


def path(*parts):
    return os.path.join(BASE, *parts)


def show_image(parts, caption=None):
    f = path(*parts)
    if os.path.exists(f):
        st.image(f, caption=caption)
    else:
        st.info(f"Image kidaikkala: {'/'.join(parts)}")


@st.cache_data(show_spinner="Loading crime data...")
def load_crimes():
    df = pd.read_csv(path("data", "app", "crimes_app.csv.gz"), parse_dates=["date"])
    df["kmeans_cluster"] = df["kmeans_cluster"].astype(int)
    df["time_cluster"] = df["time_cluster"].astype(int)
    return df


@st.cache_data
def load_csv(*parts):
    return pd.read_csv(path(*parts))


@st.cache_data
def load_json(*parts):
    with open(path(*parts)) as f:
        return json.load(f)


def exists(*parts):
    return os.path.exists(path(*parts))

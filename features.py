import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

IN_PATH = "data/processed/crimes_sample_500k.csv"
OUT_PATH = "data/processed/crimes_features.csv"

# ---------------------------------------------------
# 1. Load
# ---------------------------------------------------
df = pd.read_csv(IN_PATH, low_memory=False)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
print("Loaded:", df.shape)

# ---------------------------------------------------
# 2. Data Cleaning
# ---------------------------------------------------
before = len(df)

# Lat/Long/date missing rows remove (clustering-ku mukkiyam)
df = df.dropna(subset=["latitude", "longitude", "date"])

# Chicago boundary-kulla irukkura points mattum
df = df[
    df["latitude"].between(41.6, 42.1)
    & df["longitude"].between(-87.95, -87.5)
]

# Ward / community_area / district missing - romba kammi, remove
df = df.dropna(subset=["ward", "community_area", "district"])

# Location description missing -> UNKNOWN
df["location_description"] = df["location_description"].fillna("UNKNOWN")

# Text columns standardize
df["primary_type"] = df["primary_type"].str.strip().str.upper()
df["location_description"] = df["location_description"].str.strip().str.upper()

# Integer columns
for c in ["ward", "community_area", "district", "beat"]:
    df[c] = df[c].astype(int)

# Boolean -> 0/1
df["arrest"] = df["arrest"].astype(int)
df["domestic"] = df["domestic"].astype(int)

print(f"Rows removed in cleaning: {before - len(df):,}")
print("After cleaning:", df.shape)

# ---------------------------------------------------
# 3. Temporal Features
# ---------------------------------------------------
df["hour"] = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.day_name()
df["day_of_week_num"] = df["date"].dt.dayofweek   # Monday=0
df["month"] = df["date"].dt.month
df["is_weekend"] = (df["day_of_week_num"] >= 5).astype(int)


def get_season(m):
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Spring"
    if m in (6, 7, 8):
        return "Summer"
    return "Fall"


df["season"] = df["month"].apply(get_season)


def get_time_of_day(h):
    if 5 <= h < 12:
        return "Morning"
    if 12 <= h < 17:
        return "Afternoon"
    if 17 <= h < 21:
        return "Evening"
    return "Night"


df["time_of_day"] = df["hour"].apply(get_time_of_day)

# ---------------------------------------------------
# 4. Crime Severity Score (1 - 10)
# ---------------------------------------------------
severity_map = {
    "HOMICIDE": 10,
    "CRIMINAL SEXUAL ASSAULT": 9,
    "CRIM SEXUAL ASSAULT": 9,
    "KIDNAPPING": 9,
    "HUMAN TRAFFICKING": 9,
    "ROBBERY": 8,
    "ARSON": 8,
    "ASSAULT": 7,
    "BATTERY": 7,
    "WEAPONS VIOLATION": 7,
    "CONCEALED CARRY LICENSE VIOLATION": 5,
    "BURGLARY": 6,
    "MOTOR VEHICLE THEFT": 6,
    "OFFENSE INVOLVING CHILDREN": 6,
    "SEX OFFENSE": 6,
    "STALKING": 6,
    "INTIMIDATION": 5,
    "NARCOTICS": 5,
    "OTHER NARCOTIC VIOLATION": 5,
    "CRIMINAL DAMAGE": 4,
    "THEFT": 4,
    "DECEPTIVE PRACTICE": 3,
    "CRIMINAL TRESPASS": 3,
    "INTERFERENCE WITH PUBLIC OFFICER": 3,
    "PUBLIC PEACE VIOLATION": 2,
    "LIQUOR LAW VIOLATION": 2,
    "GAMBLING": 2,
    "PROSTITUTION": 2,
    "OBSCENITY": 2,
    "PUBLIC INDECENCY": 2,
    "NON-CRIMINAL": 1,
    "OTHER OFFENSE": 2,
}
df["crime_severity_score"] = df["primary_type"].map(severity_map).fillna(2).astype(int)

# ---------------------------------------------------
# 5. Geographic Features
# ---------------------------------------------------
# Coordinate binning (~1 km grid)
df["lat_bin"] = df["latitude"].round(2)
df["lon_bin"] = df["longitude"].round(2)
df["grid_cell"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)

# Normalize coordinates (0-1) for distance-based clustering
scaler = MinMaxScaler()
df[["lat_norm", "lon_norm"]] = scaler.fit_transform(df[["latitude", "longitude"]])

# ---------------------------------------------------
# 6. Categorical Encoding
# ---------------------------------------------------
# Location description: top 20 mattum, matha ellam OTHER
top_locs = df["location_description"].value_counts().nlargest(20).index
df["location_group"] = df["location_description"].where(
    df["location_description"].isin(top_locs), "OTHER"
)

le_type = LabelEncoder()
df["primary_type_encoded"] = le_type.fit_transform(df["primary_type"])

le_loc = LabelEncoder()
df["location_encoded"] = le_loc.fit_transform(df["location_group"])

df["season_encoded"] = df["season"].map(
    {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}
)

# ---------------------------------------------------
# 7. Save + Summary
# ---------------------------------------------------
os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUT_PATH, index=False)

print("\nSaved:", OUT_PATH, df.shape)
print("\nColumns:", df.columns.tolist())
print("\nCrime types:", df["primary_type"].nunique())
print("\nTop 10 crime types:")
print(df["primary_type"].value_counts().head(10))
print("\nSeason counts:")
print(df["season"].value_counts())
print("\nSeverity score distribution:")
print(df["crime_severity_score"].value_counts().sort_index())
print("\nRemaining missing values:", df.isnull().sum().sum())
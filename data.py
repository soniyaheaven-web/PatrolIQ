import pandas as pd
import glob
import os

RAW_PATTERN = "data/raw/crimes_20*.csv"
OUT_PATH = "data/processed/crimes_sample_500k.csv"
START_DATE = "2021-01-01"
TARGET = 500_000
SEED = 42

os.makedirs("data/processed", exist_ok=True)

# 1. Ella year files-ayum load panni merge pannu
files = sorted(glob.glob(RAW_PATTERN))
print("Files found:", files)

frames = []
for f in files:
    d = pd.read_csv(f, low_memory=False)
    print(f"{f}: {d.shape}")
    frames.append(d)

df = pd.concat(frames, ignore_index=True)
print("Combined shape:", df.shape)

# 2. Column names clean pannu (lowercase + underscore): "Primary Type" => "primary_type"
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
)
print(df.columns.tolist())

# 3. Duplicate records remove pannu
before = len(df)
df = df.drop_duplicates(subset="id")
print(f"Duplicates removed: {before - len(df)}")

# 4. Date convert pannu, 2021-ku munnadi records irundha remove pannu
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df = df[df["date"] >= START_DATE]
print("After date filter:", df.shape)

# 5. 2021-2026 la irundhu 500K RANDOM sample
if len(df) > TARGET:
    df = df.sample(n=TARGET, random_state=SEED)

df = df.sort_values("date", ascending=False).reset_index(drop=True)

# 6. Save
df.to_csv(OUT_PATH, index=False)
print("Saved:", df.shape)
print("Date range:", df["date"].min(), "to", df["date"].max())
print("Year-wise count:")
print(df["date"].dt.year.value_counts().sort_index())
print("Missing values (top):")
print(df.isnull().sum().sort_values(ascending=False).head(8))
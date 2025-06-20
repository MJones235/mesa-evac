import pandas as pd
import os

# ---------------- CONFIGURATION ---------------- #
BATCH_PATH = "outputs/batch-20250618194302"
METADATA_FILE = os.path.join(BATCH_PATH, "metadata.csv")
CITY_FILTER = "football"
# ------------------------------------------------ #

# Load metadata and filter for the target config
metadata = pd.read_csv(METADATA_FILE)
df = metadata[
    (metadata["city"] == CITY_FILTER) &
    (metadata["percent_compliant"] == 1.0)
]

print(df)

# Function to compute time to reach X% evacuated
def time_to_threshold(filepath, threshold=0.90):
    try:
        df = pd.read_csv(filepath)
        df = df[df["evacuation_started"] == True].copy()
        if df.empty or df["number_to_evacuate"].max() == 0:
            return None
        max_to_evacuate = df["number_to_evacuate"].max()
        df["evac_ratio"] = df["number_evacuated"] / max_to_evacuate
        hit = df[df["evac_ratio"] >= threshold]
        if not hit.empty:
            return pd.to_timedelta(hit["time_elapsed"].iloc[0]).total_seconds() / 60  # minutes
    except Exception:
        return None
    return None

# Locate and process model.csv files
times = []
for row in df.itertuples():
    model_file = os.path.join(row.output_path, os.path.basename(row.output_path) + ".model.csv")
    if os.path.exists(model_file):
        t = time_to_threshold(model_file)
        if t is not None:
            times.append(t)

# Compute summary stats
mean_time = round(pd.Series(times).mean(), 2)
std_dev = round(pd.Series(times).std(), 2)
count = len(times)

print(mean_time, std_dev, count)
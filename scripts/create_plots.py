import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.wkt import loads
from scipy.ndimage import gaussian_filter1d

# ---------------- CONFIGURATION ---------------- #
# Root path that contains the batch folders
BATCH_PATH = "outputs/batch-20250410173131"
METADATA_FILE = os.path.join(BATCH_PATH, "metadata.csv")

# Behaviour type to plot
BEHAVIOUR_COLUMN = "percent_curious"  # or "percent_familiar", "percent_non_compliant"
BEHAVIOUR_LABEL = "Curiosity"
CITY_FILTER = "football"
# ------------------------------------------------ #

# Load metadata and filter for Monument scenario
metadata = pd.read_csv(METADATA_FILE)
df = metadata[metadata["city"] == CITY_FILTER]


# Define 3 configurations to compare
config_labels = {
    "100% Compliant": (1.0, 0.0),
    f"50% Compliant + {BEHAVIOUR_LABEL}": (0.5, 0.5),
    f"100% {BEHAVIOUR_LABEL}": (0.0, 1.0),
}

# Container for results
evac_times_by_config = {}
evac_counts = {}  # new
total_counts = {}  # new

# Loop over each configuration
for label, (compliant_val, behaviour_val) in config_labels.items():

    subset = df[
        (df["percent_compliant"] == compliant_val)
        & (df[BEHAVIOUR_COLUMN] == behaviour_val)
    ]

    all_evacs = []
    evac_count = 0
    total_count = 0

    for row in subset.itertuples():
        agent_file_name_name = os.path.basename(row.output_path) + ".agent.csv"
        model_file_name_name = os.path.basename(row.output_path) + ".model.csv"
        agent_file = os.path.join(row.output_path, agent_file_name_name)
        model_file = os.path.join(row.output_path, model_file_name_name)

        if not os.path.exists(agent_file):
            continue
        try:
            agent_df = pd.read_csv(agent_file)
            model_df = pd.read_csv(model_file)

            last_row = model_df.iloc[-1]
            total_count += last_row["number_to_evacuate"]
            evac_count += last_row["number_evacuated"]

            evac_df = agent_df[agent_df["evacuated"] == True]

            # Get time of first evacuation per agent
            times = evac_df.groupby("AgentID")["Step"].min().values * 10
            all_evacs.extend(times)
        except Exception as e:
            print(f"Error reading {agent_file}: {e}")
            continue

    evac_times_by_config[label] = all_evacs
    evac_counts[label] = evac_count
    total_counts[label] = total_count

# ---------------- PLOT ---------------- #
plt.figure(figsize=(10, 6))

for label in config_labels:
    times = evac_times_by_config[label]
    if not times:
        continue
    evac_rate = evac_counts[label] / total_counts[label]
    label_with_rate = f"{label} ({evac_rate:.0%} evacuated)"
    sns.kdeplot(times, label=label_with_rate, fill=True, common_norm=False, alpha=0.4)

plt.axvline(2100, color="red", linestyle="--", label="25-minute cutoff")
plt.xlabel("Time to Evacuate (seconds)")
plt.ylabel("Density")
plt.title(f"Evacuation Time Distribution ({BEHAVIOUR_LABEL})")
plt.xlim(0, 2200)  # clip boundaries
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()
"""
#########################################
"""
EXIT_RADIUS = 20

# Optimized version using spatial joins for exit congestion analysis
exit_density_by_config_fast = {label: {} for label in config_labels}

for label, (compliant_val, curious_val) in config_labels.items():
    print(f"label: {label}")
    subset = df[
        (df["percent_compliant"] == compliant_val)
        & (df["percent_curious"] == curious_val)
    ]

    for i, row in enumerate(subset.itertuples(), start=1):
        print(f"→ {label}: run {i}/{len(subset)}")
        agent_file = os.path.join(
            row.output_path, os.path.basename(row.output_path) + ".agent.csv"
        )
        if not os.path.exists(agent_file):
            continue

        try:
            agent_df = pd.read_csv(agent_file)
            agent_df["geometry"] = agent_df["location"].apply(loads)
            gdf = gpd.GeoDataFrame(agent_df, geometry="geometry", crs="EPSG:27700")

            # Identify exit points as first position where each agent was marked as evacuated
            evac_df = gdf[gdf["evacuated"] == True]
            exit_points = evac_df.groupby("AgentID").first()["geometry"].values

            # Create buffered zones around each exit
            exit_buffers = gpd.GeoDataFrame(
                geometry=[pt.buffer(EXIT_RADIUS) for pt in exit_points],
                crs="EPSG:27700",
            )
            exit_buffers["exit_id"] = range(len(exit_buffers))

            # Check congestion near exits over time using spatial joins
            for step in gdf["Step"].unique():
                step_gdf = gdf[gdf["Step"] == step][["AgentID", "geometry"]]
                step_gdf = step_gdf.dropna(subset=["geometry"])
                if step_gdf.empty:
                    continue

                joined = gpd.sjoin(
                    step_gdf, exit_buffers, how="inner", predicate="intersects"
                )
                count = joined["AgentID"].nunique()

                time_sec = step * 10
                if time_sec in exit_density_by_config_fast[label]:
                    exit_density_by_config_fast[label][time_sec] += count
                else:
                    exit_density_by_config_fast[label][time_sec] = count

        except Exception as e:
            print(f"Error processing {agent_file}: {e}")
            continue

# Convert to DataFrame for plotting
exit_density_df_fast = pd.DataFrame(exit_density_by_config_fast).fillna(0).sort_index()
exit_density_df_fast.to_csv(os.path.join(BATCH_PATH, "exit_congestion.csv"))

exit_density_df_fast = pd.read_csv(
    os.path.join(BATCH_PATH, "exit_congestion.csv"), index_col=0
)

# Apply rolling average smoothing with a window of 6 steps (~60 seconds) to the congestion data
smoothed_exit_density_df = exit_density_df_fast.rolling(window=6, center=True).mean()

# Plot smoothed version
plt.figure(figsize=(10, 6))
for label in config_labels:
    if label in smoothed_exit_density_df:
        plt.plot(
            smoothed_exit_density_df.index, smoothed_exit_density_df[label], label=label
        )

plt.xlabel("Time (seconds)")
plt.ylabel("Agents near exits (within 20m, smoothed)")
plt.title("Smoothed Exit Congestion Over Time (St James' Park)")
plt.axvline(1500, color="red", linestyle="--", label="25-minute cutoff")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


##############################


# Compute outflow: number of agents exiting the evacuation zone at each timestep
outflow_by_config = {label: {} for label in config_labels}

for label, (compliant_val, curious_val) in config_labels.items():
    subset = df[
        (df["percent_compliant"] == compliant_val)
        & (df["percent_curious"] == curious_val)
    ]

    for row in subset.itertuples():
        agent_file = os.path.join(
            row.output_path, os.path.basename(row.output_path) + ".agent.csv"
        )
        if not os.path.exists(agent_file):
            continue
        try:
            agent_df = pd.read_csv(agent_file)
            agent_df["Step"] = agent_df["Step"].astype(int)

            # Find agents who are marked as evacuated
            evac_df = agent_df[agent_df["evacuated"] == True]

            # For each agent, find the first timestep where they were marked evacuated
            evac_steps = evac_df.groupby("AgentID")["Step"].min()

            # Count how many agents evacuated at each timestep
            step_counts = evac_steps.value_counts().sort_index()
            for step, count in step_counts.items():
                time_sec = step * 10
                if time_sec in outflow_by_config[label]:
                    outflow_by_config[label][time_sec] += count
                else:
                    outflow_by_config[label][time_sec] = count
        except Exception as e:
            print(f"Error processing {agent_file}: {e}")
            continue

# Convert to DataFrame for plotting
outflow_df = pd.DataFrame(outflow_by_config).fillna(0).sort_index()

# Apply Gaussian smoothing to each series
gaussian_smoothed_df = pd.DataFrame(index=outflow_df.index)
for label in config_labels:
    if label in outflow_df:
        gaussian_smoothed_df[label] = gaussian_filter1d(
            outflow_df[label].values, sigma=4
        )

# Plot Gaussian-smoothed version
plt.figure(figsize=(10, 6))
for label in config_labels:
    if label in gaussian_smoothed_df:
        plt.plot(gaussian_smoothed_df.index, gaussian_smoothed_df[label], label=label)

plt.xlabel("Time (seconds)")
plt.ylabel("Agents exiting zone per timestep (Gaussian-smoothed)")
plt.title("Smoothed Evacuation Zone Outflow Rate (St James' Park)")
# plt.axvline(1500, color="red", linestyle="--", label="25-minute cutoff")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIGURATION ---------------- #
# Root path that contains the batch folders
BATCH_PATH = "outputs/batch-20250328183511"
METADATA_FILE = os.path.join(BATCH_PATH, "metadata.csv")

# Behaviour type to plot
BEHAVIOUR_COLUMN = "percent_curious"  # or "percent_familiar", "percent_non_compliant"
BEHAVIOUR_LABEL = "Curiosity"
CITY_FILTER = "newcastle-md"
# ------------------------------------------------ #

# Load metadata and filter for Monument scenario
metadata = pd.read_csv(METADATA_FILE)
monument_df = metadata[metadata["city"] == CITY_FILTER]

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

    subset = monument_df[
        (monument_df["percent_compliant"] == compliant_val)
        & (monument_df[BEHAVIOUR_COLUMN] == behaviour_val)
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
            df = pd.read_csv(agent_file)
            model_df = pd.read_csv(model_file)

            last_row = model_df.iloc[-1]
            total_count += last_row["number_to_evacuate"]
            evac_count += last_row["number_evacuated"]

            evac_df = df[df["evacuated"] == True]

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

plt.axvline(1500, color="red", linestyle="--", label="25-minute cutoff")
plt.xlabel("Time to Evacuate (seconds)")
plt.ylabel("Density")
plt.title(f"Evacuation Time Distribution ({BEHAVIOUR_LABEL})")
plt.xlim(0, 1600)  # clip boundaries
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()

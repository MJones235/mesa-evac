import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ---------------- CONFIGURATION ---------------- #
BATCH_PATH = "outputs/batch-20250411102536"
METADATA_FILE = os.path.join(BATCH_PATH, "metadata.csv")
BEHAVIOUR_COLUMN = "percent_curious"  # or "percent_familiar", "percent_non_compliant"
BEHAVIOUR_LABEL = "Curiosity"
CITY_FILTER = "football"
TIME_STEP_SECONDS = 10
# ------------------------------------------------ #

# Load metadata and filter for the city
metadata = pd.read_csv(METADATA_FILE)
df = metadata[metadata["city"] == CITY_FILTER]

# Define 3 configurations to compare
config_labels = {
    "100% Compliant": (1.0, 0.0),
    f"50% Compliant + {BEHAVIOUR_LABEL}": (0.5, 0.5),
    f"100% {BEHAVIOUR_LABEL}": (0.0, 1.0),
}

# Compute normalised outflow: % of agents requiring evacuation per timestep
outflow_by_config = {label: {} for label in config_labels}

for label, (compliant_val, behaviour_val) in config_labels.items():
    subset = df[
        (df["percent_compliant"] == compliant_val)
        & (df[BEHAVIOUR_COLUMN] == behaviour_val)
    ]

    for row in subset.itertuples():
        agent_file = os.path.join(row.output_path, os.path.basename(row.output_path) + ".agent.csv")
        model_file = os.path.join(row.output_path, os.path.basename(row.output_path) + ".model.csv")
        if not (os.path.exists(agent_file) and os.path.exists(model_file)):
            continue
        try:
            agent_df = pd.read_csv(agent_file)
            model_df = pd.read_csv(model_file)
            num_required = model_df["number_to_evacuate"].iloc[-1]

            evac_df = agent_df[agent_df["evacuated"] == True]
            evac_steps = evac_df.groupby("AgentID")["Step"].min()
            step_counts = evac_steps.value_counts().sort_index()

            for step, count in step_counts.items():
                time_sec = step * TIME_STEP_SECONDS
                normalised = (count / num_required) * 100  # percent of evacuees
                outflow_by_config[label][time_sec] = outflow_by_config[label].get(time_sec, 0) + normalised
        except Exception as e:
            print(f"Error processing {agent_file}: {e}")
            continue

# Convert to DataFrame for plotting
outflow_df = pd.DataFrame(outflow_by_config).fillna(0).sort_index()

# Apply Gaussian smoothing
gaussian_smoothed_df = pd.DataFrame(index=outflow_df.index)
for label in config_labels:
    if label in outflow_df:
        gaussian_smoothed_df[label] = gaussian_filter1d(outflow_df[label].values, sigma=4)

# Plot
plt.figure(figsize=(6, 4))
for label in config_labels:
    if label in gaussian_smoothed_df:
        plt.plot(gaussian_smoothed_df.index, gaussian_smoothed_df[label], label=label, linewidth=2)

plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Evacuated per Timestep (% of Population)", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title="Behavioural Mix", fontsize=10, title_fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("evacuation_outflow_rate_normalised.png", dpi=300)
plt.show()
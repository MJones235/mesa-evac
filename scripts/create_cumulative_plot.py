import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------- CONFIGURATION ---------------- #
BATCH_PATH = "outputs/batch-20250411103815"
METADATA_FILE = os.path.join(BATCH_PATH, "metadata.csv")
BEHAVIOUR_COLUMN = "percent_curious"  # or "percent_familiar", "percent_non_compliant"
BEHAVIOUR_LABEL = "Curious"
CITY_FILTER = "football"
TIME_STEP_SECONDS = 10
# ------------------------------------------------ #

# Load metadata and filter for the city
metadata = pd.read_csv(METADATA_FILE)
df = metadata[metadata["city"] == CITY_FILTER]

# Define configurations to compare
config_labels = {
    "100% Compliant": (1.0, 0.0),
    f"50% Compliant 50% {BEHAVIOUR_LABEL}": (0.5, 0.5),
    f"100% {BEHAVIOUR_LABEL}": (0.0, 1.0),
}

# Prepare cumulative time series storage
all_runs = []

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

            cumulative = 0
            time_series = {}
            for step in range(0, max(evac_steps.max(), 211)):
                time_sec = step * TIME_STEP_SECONDS
                count = step_counts.get(step, 0)
                cumulative += (count / num_required) * 100  # percent of evacuees
                time_series[time_sec] = cumulative

            ts_df = pd.DataFrame(list(time_series.items()), columns=["time", "cumulative"])
            ts_df["Configuration"] = label
            all_runs.append(ts_df)

        except Exception as e:
            print(f"Error processing {agent_file}: {e}")
            continue

# Combine and summarise all runs
combined_df = pd.concat(all_runs)

# Group by time and configuration
summary_df = (
    combined_df.groupby(["Configuration", "time"])["cumulative"]
    .agg([
        ("mean", "mean"),
        ("lower", lambda x: x.quantile(0.025)),
        ("upper", lambda x: x.quantile(0.975))
    ])
    .reset_index()
)

# Plot
plt.figure(figsize=(6, 5.5))
for label in config_labels:
    subset = summary_df[summary_df["Configuration"] == label]
    plt.plot(subset["time"], subset["mean"], label=label, linewidth=2)
    plt.fill_between(subset["time"], subset["lower"], subset["upper"], alpha=0.2)

# Add custom legend entry for CI band
ci_patch = Patch(facecolor='grey', alpha=0.2, label="95% Confidence Interval")
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(ci_patch)
labels.append("95% Confidence Interval")

plt.legend(handles=handles, labels=labels, title="Behavioural Mix", fontsize=12, title_fontsize=14, loc="lower right")

plt.xlabel("Time (seconds)", fontsize=16)
plt.ylabel("Cumulative Evacuated (% of Population)", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("cumulative_evacuation_with_ci.png", dpi=300)
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

# === CONFIG: Metadata paths per behaviour type ===
metadata_files = {
    "Curiosity": "outputs/batch-20250411103815/metadata.csv",
    "Social Attachment": "outputs/batch-20250411044617/metadata.csv",
    "Non-Compliance": "outputs/batch-20250411044617/metadata.csv",
}

def get_summary_stats(metadata_path):
    df = pd.read_csv(metadata_path)
    summary = []

    for _, row in df.iterrows():
        model_file = os.path.join(row.output_path, os.path.basename(row.output_path) + ".model.csv")

        try:
            model_df = pd.read_csv(model_file)
            last_row = model_df.iloc[150]  # assumes 150 = 25 minutes
            num_evacuated = last_row["number_evacuated"]
            num_required = last_row["number_to_evacuate"]

            prop_compliant = row.percent_compliant

            if prop_compliant is not None and num_required > 0:
                summary.append({
                    "prop_compliant": float(prop_compliant) * 100,  # convert to percentage
                    "evacuation_rate": 100 * num_evacuated / num_required,  # also percentage
                })
        except Exception as e:
            print(f"Skipping {model_file}: {e}")
            continue

    return pd.DataFrame(summary)

# === Aggregate across behaviour types ===
all_data = []
for label, metadata_file in metadata_files.items():
    df = get_summary_stats(metadata_file)
    df["Behaviour"] = label
    all_data.append(df)

summary_df = pd.concat(all_data, ignore_index=True)

# === Group and summarise ===
plot_data = (
    summary_df.groupby(["Behaviour", "prop_compliant"])
    .agg(
        mean_evac=("evacuation_rate", "mean"),
        lower=("evacuation_rate", lambda x: x.quantile(0.025)),
        upper=("evacuation_rate", lambda x: x.quantile(0.975)),
    )
    .reset_index()
)

# === Plot with improved styling ===
plt.figure(figsize=(6, 5.5))
for behaviour in plot_data["Behaviour"].unique():
    subset = plot_data[plot_data["Behaviour"] == behaviour]
    plt.plot(subset["prop_compliant"], subset["mean_evac"], label=behaviour, linewidth=2)
    plt.fill_between(subset["prop_compliant"], subset["lower"], subset["upper"], alpha=0.2)


# Add custom legend entry for CI band
ci_patch = Patch(facecolor='grey', alpha=0.2, label="95% Confidence Interval")
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(ci_patch)
labels.append("95% Confidence Interval")

plt.legend(handles=handles, labels=labels, title="Contrasting Behaviour", fontsize=13, title_fontsize=14)


plt.xlabel("Compliant Behaviour (% of Population)", fontsize=16)
plt.ylabel("Evacuated Within 25 Mins (% of Population)", fontsize=16)
plt.ylim(0, 105)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("evacuation_summary_plot_updated.png", dpi=300)
plt.show()
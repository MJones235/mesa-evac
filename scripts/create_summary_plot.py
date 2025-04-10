import pandas as pd
import matplotlib.pyplot as plt
import os

# CONFIG: List of metadata files for each behaviour type
metadata_files = {
    "Curiosity": "outputs/batch-20250328203520/metadata.csv",
    "Social Attachment": "outputs/batch-20240807094242/metadata.csv",
    "Non-Compliance": "outputs/batch-20240806231644/metadata.csv",
}


def get_summary_stats(metadata_path):
    df = pd.read_csv(metadata_path)
    summary = []

    for _, row in df.iterrows():
        model_file_name_name = os.path.basename(row.output_path) + ".model.csv"
        model_file = os.path.join(row.output_path, model_file_name_name)

        try:
            model_df = pd.read_csv(model_file)
            last_row = model_df.iloc[-1]
            num_evacuated = last_row["number_evacuated"]
            num_required = last_row["number_to_evacuate"]

            # Extract proportion compliant from the string representation of agent behaviour
            prop_compliant = row.percent_compliant

            if prop_compliant is not None and num_required > 0:
                summary.append(
                    {
                        "prop_compliant": float(prop_compliant),
                        "evacuation_rate": num_evacuated / num_required,
                    }
                )

        except Exception as e:
            # print(f"Skipping {folder_name}: {e}")
            pass

    return pd.DataFrame(summary)


# Aggregate and label data
all_data = []

for label, metadata_file in metadata_files.items():
    df = get_summary_stats(metadata_file)
    df["Behaviour"] = label
    all_data.append(df)

summary_df = pd.concat(all_data, ignore_index=True)

# Group and aggregate
plot_data = (
    summary_df.groupby(["Behaviour", "prop_compliant"])
    .agg(
        mean_evac=("evacuation_rate", "mean"),
        lower=("evacuation_rate", lambda x: x.quantile(0.025)),
        upper=("evacuation_rate", lambda x: x.quantile(0.975)),
    )
    .reset_index()
)

# Plot
plt.figure(figsize=(10, 6))
for behaviour in plot_data["Behaviour"].unique():
    subset = plot_data[plot_data["Behaviour"] == behaviour]
    plt.plot(subset["prop_compliant"], subset["mean_evac"], label=behaviour)
    plt.fill_between(
        subset["prop_compliant"], subset["lower"], subset["upper"], alpha=0.2
    )

plt.xlabel("Proportion Compliant")
plt.ylabel("Proportion Evacuated (within 25 minutes)")
plt.title("Evacuation Success by Compliance Level and Behavioural Mix")
plt.ylim(0, 1.05)
plt.legend(title="Contrasting Behaviour")
plt.grid(True)
plt.tight_layout()
plt.savefig("evacuation_summary_plot.png", dpi=300)
plt.show()

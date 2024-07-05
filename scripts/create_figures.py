import osmnx as ox
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scripts.load_data_from_file import load_data_from_file


def plot_environment(output_path: str) -> None:
    (_, _, graph, _, _, building_df) = load_data_from_file(output_path)

    _, ax = ox.plot_graph(graph, show=False, node_size=0, edge_linewidth=0.5)

    building_df.plot(ax=ax, color="#8f8f8f")

    plt.savefig(output_path + "-environment" + ".png")


def plot_number_agents_against_evacuation_zone_size(batch_path: str) -> None:
    metadata = pd.read_csv(batch_path + "/metadata.csv", header=0)
    data = {
        "evacuation_zone_radius": [],
        "number_to_evacuate": [],
        "number_evacuated": [],
    }
    for row in metadata.itertuples():
        model_df = pd.read_csv(
            batch_path
            + f"/radius-{row.evacuation_zone_radius}-run-{row.n}"
            + f"/radius-{row.evacuation_zone_radius}-run-{row.n}.model.csv"
        )
        final_row = model_df.iloc[-1]
        data["evacuation_zone_radius"].append(row.evacuation_zone_radius)
        data["number_to_evacuate"].append(final_row.number_to_evacuate)
        data["number_evacuated"].append(final_row.number_evacuated)

    df = pd.DataFrame.from_dict(data)

    sns.set_theme(font_scale=1.2, style="whitegrid")

    g = sns.relplot(
        data=df.melt("evacuation_zone_radius", var_name="variable", value_name="count"),
        x="evacuation_zone_radius",
        y="count",
        hue="variable",
        kind="line",
    )
    g.set(xlabel="Evacuation zone radius (m)")
    g.set(ylabel="Number of people")
    g.legend.set_title("")
    for t, l in zip(
        g.legend.texts, ["Requiring evacuation", "Evacuated after 10 mins"]
    ):
        t.set_text(l)
    sns.move_legend(g, "upper center", frameon=True)
    g.figure.tight_layout()
    plt.savefig(batch_path + "/agents_against_evacuation_zone_radius.png")

import osmnx as ox
import matplotlib.pyplot as plt

from scripts.load_data_from_file import load_data_from_file


def plot_environment(output_path: str) -> None:
    (agent_df, _, graph, _, _, _, building_df) = load_data_from_file(output_path)

    _, ax = ox.plot_graph(graph, show=False, node_size=0, edge_linewidth=0.5)

    building_df.plot(ax=ax, color="#8f8f8f")

    plt.savefig(output_path + "-environment" + ".png")

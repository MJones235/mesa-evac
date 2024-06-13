import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
from matplotlib import animation, pyplot


def create_video(output_path: str) -> None:
    (agent_df, model_df, graph, nodes, evacuation_zone, exits) = load_data_from_file(
        output_path
    )

    writer = animation.writers["ffmpeg"]
    metadata = dict(title="MesaEvac Simulation")
    writer = writer(fps=5, metadata=metadata)

    f, ax = ox.plot_graph(graph, show=False, node_size=0, edge_linewidth=0.5)
    evacuation_zone.plot(ax=ax, alpha=0.5)
    ax.scatter(exits.geometry.x, exits.geometry.y, color="#0f9900", s=10)

    pyplot.show()


def load_data_from_file(output_path: str) -> None:
    agent_df = pd.read_csv(output_path + ".agent.csv", index_col="Step")
    model_df = pd.read_csv(output_path + ".model.csv")
    graph = nx.read_gml(output_path + ".gml")
    nodes, _ = ox.convert.graph_to_gdfs(graph)
    gpkg = output_path + ".gpkg"
    evacuation_zone = gpd.read_file(gpkg, layer="evacuation_zone")
    exits = gpd.read_file(gpkg, layer="exits")

    return agent_df, model_df, graph, nodes, evacuation_zone, exits

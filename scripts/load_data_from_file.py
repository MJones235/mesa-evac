import pandas as pd
import geopandas as gpd
from shapely import wkt
import networkx as nx
import osmnx as ox


def load_data_from_file(output_path: str) -> None:
    agent_df = pd.read_csv(output_path + ".agent.csv", index_col="Step")
    agent_df["location"] = agent_df["location"].apply(wkt.loads)
    model_df = pd.read_csv(output_path + ".model.csv")
    graph = nx.read_gml(output_path + ".gml")
    nodes, _ = ox.convert.graph_to_gdfs(graph)
    gpkg = output_path + ".gpkg"
    evacuation_zone = gpd.read_file(gpkg, layer="evacuation_zone")
    buildings = gpd.read_file(gpkg, layer="buildings")

    return agent_df, model_df, graph, nodes, evacuation_zone, buildings

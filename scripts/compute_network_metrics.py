import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
from shapely import Point
import matplotlib.pyplot as plt

# Define study areas
locations = {
    "Monument": Point(424860, 564443),
    "St_James_Park": Point(424192, 564602)
}

buffers = {name: point.buffer(500) for name, point in locations.items()}

buffers_wgs = {
    name: gpd.GeoSeries(buffer, crs="EPSG:27700").to_crs(epsg=4326).iloc[0]
    for name, buffer in buffers.items()
}

graphs = {}
for name, polygon_wgs in buffers_wgs.items():
    print(f"Processing {name}...")
    # Download the network
    G = ox.graph_from_polygon(polygon_wgs, network_type="walk")

    # Simplify and project
    G_proj = ox.project_graph(G)
    graphs[name] = G_proj


def compute_network_metrics(G):
    G_undirected = G.to_undirected()
    nodes = len(G_undirected.nodes)
    edges = len(G_undirected.edges)
    avg_degree = sum(dict(G_undirected.degree()).values()) / nodes
    avg_path_length = nx.average_shortest_path_length(G_undirected, weight='length')
    diameter = nx.diameter(G_undirected)
    betweenness = nx.betweenness_centrality(G_undirected)
    closeness = nx.closeness_centrality(G_undirected)
    avg_betweenness = sum(betweenness.values()) / nodes
    avg_closeness = sum(closeness.values()) / nodes

    plt.hist(betweenness.values(), bins=50)
    plt.title("Betweenness Centrality Distribution")
    plt.xlabel("Betweenness")
    plt.ylabel("Number of Nodes")
    plt.show()

    return {
        "Nodes": nodes,
        "Edges": edges,
        "Avg Degree": avg_degree,
        "Avg Path Length": avg_path_length,
        "Diameter": diameter,
        "Avg Betweenness": avg_betweenness,
        "Avg Closeness": avg_closeness
    }

metrics = {name: compute_network_metrics(G) for name, G in graphs.items()}
df_metrics = pd.DataFrame(metrics).T
print(df_metrics)

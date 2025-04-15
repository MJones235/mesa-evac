import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.wkt import loads
from shapely.geometry import Point
from scipy.spatial import cKDTree
import numpy as np

# ---------------- CONFIGURATION ---------------- #
agent_file = "outputs/newcastle-md/20250412234447/20250412234447.agent.csv"
gpkg_file = "outputs/newcastle-md/20250412234447/20250412234447.gpkg"
gml_file = "outputs/newcastle-md/20250412234447/20250412234447.gml"
crs_epsg = "EPSG:27700"
# ------------------------------------------------ #

# Load data
agent_df = pd.read_csv(agent_file)
agent_df["geometry"] = agent_df["location"].apply(loads)
agent_gdf = gpd.GeoDataFrame(agent_df, geometry="geometry", crs=crs_epsg)
agent_start = agent_gdf[agent_gdf["Step"] == 0]


# Load evacuation zone
evac_zone = gpd.read_file(gpkg_file, layer="evacuation_zone", crs=crs_epsg)
evac_polygon = evac_zone.unary_union
agent_start = agent_start[agent_start.geometry.within(evac_polygon)]


# Load road network and simplify
G = nx.read_gml(gml_file)
G = nx.MultiDiGraph(G)
G = ox.simplify_graph(G)
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
nodes = nodes.to_crs(crs_epsg)
edges = edges.to_crs(crs_epsg)

# Identify exits as intersections of road edges with evacuation zone boundary
exit_points = edges.unary_union.intersection(evac_polygon.boundary)

# If the intersection is a MultiPoint or LineString, flatten it to a list of Points
if exit_points.geom_type == "Point":
    exit_points = [exit_points]
elif hasattr(exit_points, "geoms"):
    exit_points = [pt for pt in exit_points.geoms if pt.geom_type == "Point"]

# Build KDTree for road network nodes
node_coords = np.array(list(zip(nodes.geometry.x, nodes.geometry.y)))
kd_tree = cKDTree(node_coords)
node_ids = list(nodes.index)

# Map exit points to nearest road network node indices
exit_node_indices = []
for pt in exit_points:
    dist, idx = kd_tree.query([pt.x, pt.y])
    exit_node_indices.append(node_ids[idx])

# Function to find nearest node to a given point
def get_nearest_node_idx(point):
    dist, idx = kd_tree.query([point.x, point.y])
    return node_ids[idx]

# Compute shortest path lengths for each agent to nearest exit node
path_lengths = []

for row in agent_start.itertuples():
    try:
        start_node = get_nearest_node_idx(row.geometry)
        lengths = {
            target: nx.shortest_path_length(G, start_node, target, weight="length")
            for target in exit_node_indices
            if nx.has_path(G, start_node, target)
        }
        if lengths:
            print(min(lengths.values()))
            path_lengths.append(min(lengths.values()))
    except Exception:
        continue

# Compute statistics
mean_distance = np.mean(path_lengths)
median_distance = np.median(path_lengths)
count = len(path_lengths)

print(mean_distance, median_distance, count)

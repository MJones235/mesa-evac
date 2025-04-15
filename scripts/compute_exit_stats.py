import pandas as pd
import geopandas as gpd
from shapely.wkt import loads
from shapely.geometry import Point
from scipy.spatial import cKDTree
import numpy as np
import networkx as nx
import osmnx as ox

# ---------------- CONFIGURATION ---------------- #
agent_file = "outputs/newcastle-md/20250412234447/20250412234447.agent.csv"
gpkg_file = "outputs/newcastle-md/20250412234447/20250412234447.gpkg"
gml_file = "outputs/newcastle-md/20250412234447/20250412234447.gml"

crs_epsg = "EPSG:27700"
# ------------------------------------------------ #

# Load agent data
agent_df = pd.read_csv(agent_file)
agent_df["geometry"] = agent_df["location"].apply(loads)
agent_gdf = gpd.GeoDataFrame(agent_df, geometry="geometry", crs=crs_epsg)


# Identify the first timestep where each agent becomes evacuated
evac_events = (
    agent_gdf.sort_values(["AgentID", "Step"])
    .groupby("AgentID")
    .apply(lambda df: df[df["evacuated"] == True].head(1))
    .reset_index(drop=True)
)

# Load evacuation zone and road edges
evac_zone = gpd.read_file(gpkg_file, layer="evacuation_zone", crs=crs_epsg)
# Load road network and simplify
G = nx.read_gml(gml_file)
G = nx.MultiDiGraph(G)  # ensure correct type
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
nodes = nodes.to_crs(crs_epsg)
edges = edges.to_crs(crs_epsg)

# Find intersections of evacuation zone boundary with roads = exit points
exit_points = edges.unary_union.intersection(evac_zone.geometry[0].boundary)

# Flatten into a list of Points
if exit_points.geom_type == "Point":
    exit_points = [exit_points]
elif hasattr(exit_points, "geoms"):
    exit_points = [pt for pt in exit_points.geoms if pt.geom_type == "Point"]
else:
    exit_points = []

# Convert exit points to GeoDataFrame
exits_gdf = gpd.GeoDataFrame(geometry=exit_points, crs=crs_epsg)
exits_gdf["exit_id"] = range(len(exits_gdf))

# Build KDTree for snapping evacuated agents to nearest exit
exit_coords = np.array([[pt.x, pt.y] for pt in exits_gdf.geometry])
kd_tree = cKDTree(exit_coords)

# Map each evacuated agent to nearest exit
agent_coords = np.array([[pt.x, pt.y] for pt in evac_events.geometry])
dists, idxs = kd_tree.query(agent_coords)
evac_events["exit_id"] = idxs

# Count usage
exit_counts = evac_events["exit_id"].value_counts().sort_values(ascending=False)
exit_counts = exit_counts.rename_axis("exit_id").reset_index(name="count")
exit_counts["percentage"] = (exit_counts["count"] / exit_counts["count"].sum() * 100).round(2)

print(exit_counts)
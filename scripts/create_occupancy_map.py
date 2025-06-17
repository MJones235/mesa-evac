import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.wkt import loads
import osmnx as ox
import networkx as nx
from shapely.geometry import Point
from shapely.ops import nearest_points
from scipy.spatial import cKDTree
from shapely.strtree import STRtree
import numpy as np

# ---------------- CONFIGURATION ---------------- #
scenarios = {
    "Monument": {
        "agent_file": "outputs/newcastle-md/20250412234447/20250412234447.agent.csv",
        "gpkg_file": "outputs/newcastle-md/20250412234447/20250412234447.gpkg",
        "gml_file": "outputs/newcastle-md/20250412234447/20250412234447.gml"
    },
    "St James' Park": {
        "agent_file": "outputs/batch-20250411234245/agent_behaviour-{<Behaviour.NON_COMPLIANT: 2>: 0, <Behaviour.COMPLIANT: 1>: 1.0, <Behaviour.CURIOUS: 3>: 0.0, <Behaviour.FAMILIAR: 4>: 0}-run-49/agent_behaviour-{<Behaviour.NON_COMPLIANT: 2>: 0, <Behaviour.COMPLIANT: 1>: 1.0, <Behaviour.CURIOUS: 3>: 0.0, <Behaviour.FAMILIAR: 4>: 0}-run-49.agent.csv",
        "gpkg_file": "outputs/batch-20250411234245/agent_behaviour-{<Behaviour.NON_COMPLIANT: 2>: 0, <Behaviour.COMPLIANT: 1>: 1.0, <Behaviour.CURIOUS: 3>: 0.0, <Behaviour.FAMILIAR: 4>: 0}-run-49/agent_behaviour-{<Behaviour.NON_COMPLIANT: 2>: 0, <Behaviour.COMPLIANT: 1>: 1.0, <Behaviour.CURIOUS: 3>: 0.0, <Behaviour.FAMILIAR: 4>: 0}-run-49.gpkg",
        "gml_file": "outputs/batch-20250411234245/agent_behaviour-{<Behaviour.NON_COMPLIANT: 2>: 0, <Behaviour.COMPLIANT: 1>: 1.0, <Behaviour.CURIOUS: 3>: 0.0, <Behaviour.FAMILIAR: 4>: 0}-run-49/agent_behaviour-{<Behaviour.NON_COMPLIANT: 2>: 0, <Behaviour.COMPLIANT: 1>: 1.0, <Behaviour.CURIOUS: 3>: 0.0, <Behaviour.FAMILIAR: 4>: 0}-run-49.gml"
    }
}
time_steps = [0, 48, 96]  # e.g. 0 and 800s if 10s timestep
# ------------------------------------------------ #

def snap_points_to_lines(points_gdf, lines_gdf):
    snapped_ids = []

    for pt in points_gdf.geometry:
        # Compute distances to all road segments
        distances = lines_gdf.geometry.distance(pt)
        segment_index = distances.idxmin()  # index of the closest segment
        snapped_ids.append(segment_index)

    return pd.Series(snapped_ids, index=points_gdf.index)



# --- Compute global max values for consistent scaling ---
all_building_counts = []
all_road_densities = []

for scenario, files in scenarios.items():
    agent_df = pd.read_csv(files["agent_file"])
    agent_df["geometry"] = agent_df["location"].apply(loads)
    agent_gdf = gpd.GeoDataFrame(agent_df, geometry="geometry", crs="EPSG:27700")

    buildings = gpd.read_file(files["gpkg_file"], layer="buildings", crs="EPSG:27700")
    G = nx.read_gml(files["gml_file"])
    G = nx.MultiDiGraph(G)
    G = ox.simplify_graph(G)
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

    for t in time_steps:
        at_time = agent_gdf[agent_df["Step"] == t].copy()
        if at_time.empty:
            continue

        # Building counts
        parked = at_time[at_time["status"] == "parked"]
        join = gpd.sjoin(parked, buildings, how="left", predicate="within")
        building_counts = join.groupby("index_right").size()
        all_building_counts.extend(building_counts.values)

        # Road densities
        moving = at_time[at_time["status"] != "parked"]
        if not moving.empty:
            snapped_ids = snap_points_to_lines(moving, edges)
            road_counts = snapped_ids.value_counts()
            edges["count"] = road_counts.reindex(edges.index).fillna(0)
        else:
            edges["count"] = 0
        edges["density"] = (edges["count"] / edges.geometry.length).replace(0, 1e-3)
        all_road_densities.extend(edges["density"].fillna(0).values)

vmax_building_global = pd.Series(all_building_counts).quantile(0.98)
vmax_road_global = pd.Series(all_road_densities).quantile(0.98)


# Create plot grid
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 12))
axes = axes.flatten()
panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
panel_idx = 0

for scenario, files in scenarios.items():
    # Load agent data
    agent_df = pd.read_csv(files["agent_file"])
    agent_df["geometry"] = agent_df["location"].apply(loads)
    agent_gdf = gpd.GeoDataFrame(agent_df, geometry="geometry", crs="EPSG:27700")

    # Load buildings and evac zone
    buildings = gpd.read_file(files["gpkg_file"], layer="buildings", crs="EPSG:27700")
    evac_zone = gpd.read_file(files["gpkg_file"], layer="evacuation_zone", crs="EPSG:27700")
    bounds = evac_zone.total_bounds  # xmin, ymin, xmax, ymax

    # Load road network from GML
    G = nx.read_gml(files["gml_file"])
    G = nx.MultiDiGraph(G)  # convert to directed multigraph
    G = ox.simplify_graph(G)
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

    for t in time_steps:
        ax = axes[panel_idx]
        panel_idx += 1
        time_label = f"t = {t * 10} s"

        # Filter agents at time step
        at_time = agent_gdf[agent_gdf["Step"] == t].copy()
        if at_time.empty:
            continue

        # --- Building Occupancy ---
        parked = at_time[at_time["status"] == "parked"]
        join = gpd.sjoin(parked, buildings, how="left", predicate="within")
        building_counts = join.groupby("index_right").size()
        buildings["count"] = building_counts.reindex(buildings.index).fillna(0)
        max_val = buildings["count"].max()
        if max_val > vmax_building_global * 10:
            outlier_idx = buildings["count"].idxmax()
            outlier_val = buildings.loc[outlier_idx, "count"]
            buildings_plot = buildings.copy()
            buildings_plot.loc[outlier_idx, "count"] = np.nan
        else:
            outlier_idx = None
            buildings_plot = buildings

        # --- Road Occupancy ---
        moving = at_time[at_time["status"] != "parked"]
        moving = moving[moving.geometry.type == "Point"]

        if not moving.empty:
            snapped_ids = snap_points_to_lines(moving, edges)
            road_counts = snapped_ids.value_counts()
            edges["count"] = road_counts.reindex(edges.index).fillna(0)

        else:
            edges["count"] = 0


        # Plot base layers
        vmax_building = vmax_building_global
        buildings_plot.plot(
            ax=ax,
            column="count",
            cmap="Blues",
            linewidth=0.1,
            edgecolor="white",
            vmin=0,
            vmax=vmax_building,
            legend=False
        )

        # --- If there's an outlier, plot and annotate separately ---
        if outlier_idx is not None:
            buildings.loc[[outlier_idx]].plot(
                ax=ax,
                color="black",
                edgecolor="white",
                linewidth=0.1,
                zorder=5,
            )

            ax.text(0.5, -0.05, f"● Stadium: {int(outlier_val)} agents",
                    transform=ax.transAxes,
                    ha='center', va='top',
                    fontsize=16,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black"))


        edges["density"] = (edges["count"] / edges.geometry.length).replace(0, 1e-3)

        vmax_roads = vmax_road_global

        edges.plot(
            ax=ax,
            column="density",
            cmap="OrRd",
            linewidth=2,
            vmin=0,
            vmax=vmax_roads,
            legend=False
        )
        
        evac_zone.plot(
            ax=ax,
            edgecolor="black",
            linestyle="--",
            linewidth=1.5,
            facecolor="none",
            zorder=10  # Make sure it's on top
        )

        ax.set_title(f"{scenario}, {time_label}", fontsize=16)
        ax.set_axis_off()
        ax.set_xlim(bounds[0] - 400, bounds[2] + 400)
        ax.set_ylim(bounds[1] - 400, bounds[3] + 400)

        ax.text(0.01, 0.95, panel_labels[panel_idx - 1], transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left', backgroundcolor="white")

        """
        try:
            ctx.add_basemap(ax, crs="EPSG:27700", source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception as e:
            print(f"Error adding basemap for {scenario} at {time_label}: {e}")
        """

plt.tight_layout(rect=[0, 0.12, 1, 1])

# --- Shared colourbars ---
import matplotlib as mpl

norm_building = mpl.colors.Normalize(vmin=0, vmax=vmax_building_global)
norm_road = mpl.colors.LogNorm(vmin=1e-3, vmax=vmax_road_global)

sm_building = plt.cm.ScalarMappable(cmap="Blues", norm=norm_building)
sm_building._A = []
sm_road = plt.cm.ScalarMappable(cmap="OrRd", norm=norm_road)
sm_road._A = []

cbar_ax1 = fig.add_axes([0.2, 0.08, 0.3, 0.02])
cbar_ax2 = fig.add_axes([0.55, 0.08, 0.3, 0.02])

cb1 = fig.colorbar(sm_building, cax=cbar_ax1, orientation='horizontal')
cb1.set_label("Building Occupancy (No. of Agents)", fontsize=16)
cb1.ax.tick_params(labelsize=14)
cb1.ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
cb1.ax.xaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())

cb2 = fig.colorbar(sm_road, cax=cbar_ax2, orientation='horizontal')
cb2.set_label("Road Density (Agents per Metre Length)", fontsize=16)
cb2.ax.tick_params(labelsize=14)
#cb2.ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x:g}"))

plt.savefig("road_building_congestion_snapshot.png", dpi=300)
plt.show()
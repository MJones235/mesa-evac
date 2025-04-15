import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.wkt import loads
import os
import contextily as ctx

# ---------------- CONFIGURATION ---------------- #
agent_files = {
    "monument": {
        "agent": "/home/michael/Downloads/agent_behaviour-{<Behaviour.NON_COMPLIANT: 2>: 0, <Behaviour.COMPLIANT: 1>: 1.0, <Behaviour.CURIOUS: 3>: 0, <Behaviour.FAMILIAR: 4>: 0.0}-run-49.agent.csv",
        "gpkg": "/home/michael/Downloads/agent_behaviour-{<Behaviour.NON_COMPLIANT: 2>: 0, <Behaviour.COMPLIANT: 1>: 1.0, <Behaviour.CURIOUS: 3>: 0, <Behaviour.FAMILIAR: 4>: 0.0}-run-49.gpkg"
    },
    "football": {
        "agent": "outputs/batch-20250411102536/agent_behaviour-{<Behaviour.NON_COMPLIANT: 2>: 0, <Behaviour.COMPLIANT: 1>: 1.0, <Behaviour.CURIOUS: 3>: 0.0, <Behaviour.FAMILIAR: 4>: 0}-run-1/agent_behaviour-{<Behaviour.NON_COMPLIANT: 2>: 0, <Behaviour.COMPLIANT: 1>: 1.0, <Behaviour.CURIOUS: 3>: 0.0, <Behaviour.FAMILIAR: 4>: 0}-run-1.agent.csv",
        "gpkg": "outputs/batch-20250411102536/agent_behaviour-{<Behaviour.NON_COMPLIANT: 2>: 0, <Behaviour.COMPLIANT: 1>: 1.0, <Behaviour.CURIOUS: 3>: 0.0, <Behaviour.FAMILIAR: 4>: 0}-run-1/agent_behaviour-{<Behaviour.NON_COMPLIANT: 2>: 0, <Behaviour.COMPLIANT: 1>: 1.0, <Behaviour.CURIOUS: 3>: 0.0, <Behaviour.FAMILIAR: 4>: 0}-run-1.gpkg"
    }
}

timepoints = [0, 90]  # timestep indices (0 and 1000 seconds)
time_step_seconds = 10
colormap = "viridis"
levels = 100
bw_adjust = 0.2
thresh = 0.05
# ------------------------------------------------ #

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
axes = axes.flatten()
panel_idx = 0
panel_labels = ['(a)', '(b)', '(c)', '(d)']

# Collect bounds to align spatial extents later
all_bounds = []

# First pass: collect all agent positions and bounds
data_cache = {}

for city, files in agent_files.items():
    agent_file = files["agent"]
    gpkg_file = files["gpkg"]

    if not (os.path.exists(agent_file) and os.path.exists(gpkg_file)):
        print(f"Missing files for {city}")
        continue

    df = pd.read_csv(agent_file)
    df["geometry"] = df["location"].apply(loads)
    gdf_agents = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:27700")
    evac_zone = gpd.read_file(gpkg_file, layer="evacuation_zone", crs="EPSG:27700")
    bounds = evac_zone.total_bounds  # xmin, ymin, xmax, ymax

    for time_step in timepoints:
        ax = axes[panel_idx]
        time_label = f"{time_step * time_step_seconds} s"
        gdf_time = gdf_agents[(gdf_agents["Step"] == time_step) & (gdf_agents.geometry.type == "Point")]

        if not gdf_time.empty:
            x = gdf_time.geometry.x
            y = gdf_time.geometry.y
            sns.kdeplot(
                x=x,
                y=y,
                fill=True,
                ax=ax,
                cmap=colormap,
                bw_adjust=bw_adjust,
                levels=levels,
                thresh=thresh,
                alpha=0.8
            )

        evac_zone.plot(ax=ax, edgecolor="yellow", linewidth=1.5, facecolor="none")
        ax.set_xlim(bounds[0] - 400, bounds[2] + 400)
        ax.set_ylim(bounds[1] - 400, bounds[3] + 400)
        ax.set_title(f"{city}, t = {time_label}", fontsize=11)
        ax.set_axis_off()
        ax.text(0.01, 0.95, panel_labels[panel_idx], transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left', backgroundcolor="white")

        try:
            ctx.add_basemap(ax, crs="EPSG:27700", source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception as e:
            print(f"Error adding basemap for {city}, t={time_label}: {e}")

        panel_idx += 1

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("density_heatmaps_compliant_final.png", dpi=300)
plt.show()

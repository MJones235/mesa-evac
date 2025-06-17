import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import contextily as ctx
from shapely.wkt import loads
from matplotlib.patches import Patch
from matplotlib_scalebar.scalebar import ScaleBar

# === INPUT PATHS ===
#agent_file = "outputs/newcastle-md/20250412234447/20250412234447.agent.csv"
#gpkg_file = "outputs/newcastle-md/20250412234447/20250412234447.gpkg"

agent_file = "outputs/football/20250410185730/20250410185730.agent.csv"
gpkg_file = "outputs/football/20250613211514/20250613211514.gpkg"


output_image = "initial_map_st_james.png"  # Output image file

# === COLOUR SCHEME ===
agent_color = "green"
bomb_color = "yellow"
zone_fill = "blue"
zone_edge = "navy"


# === Load agent positions ===
df_agents = pd.read_csv(agent_file)


df_agents["geometry"] = df_agents["location"].apply(loads)
gdf_agents = gpd.GeoDataFrame(df_agents, geometry="geometry", crs="EPSG:27700")
gdf_step0 = gdf_agents[gdf_agents["Step"] == 0]

# === Load evacuation zone from .gpkg ===
evac_gdf = gpd.read_file(gpkg_file, layer="evacuation_zone")

bomb_location = evac_gdf.geometry.unary_union.centroid
bomb_gdf = gpd.GeoDataFrame(geometry=[bomb_location], crs=evac_gdf.crs)

# === Plot ===
fig, ax = plt.subplots(figsize=(8, 8))
evac_gdf.plot(ax=ax, color=zone_fill, alpha=0.1, edgecolor=zone_edge, linewidth=2, label="Evacuation zone")
gdf_step0.plot(
    ax=ax,
    markersize=10,
    color=agent_color,
    alpha=0.8,
    edgecolor="black",
    linewidth=0.2,
    label="Agents (initial positions)"
)

bomb_gdf.plot(
    ax=ax,
    color=bomb_color, 
    marker="*",
    markersize=500,
    edgecolor="black",
    linewidth=1,
    label="Bomb"
)


# Add basemap
ctx.add_basemap(ax, crs=gdf_step0.crs.to_string(), source=ctx.providers.CartoDB.Positron, zoom=17)

# Style
ax.set_axis_off()


legend_elements = [
    Patch(facecolor=zone_fill, edgecolor=zone_edge, alpha=0.1, label='Evacuation zone'),
    plt.Line2D([], [], marker='o', color='w', label='Agents (initial positions)',
               markerfacecolor=agent_color, markersize=10, alpha=0.8, markeredgecolor='black', linewidth=0.5),
    plt.Line2D([], [], marker='*', color='w', label='Bomb',
               markerfacecolor=bomb_color, markersize=18, markeredgecolor='black', linewidth=1),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=20)

buffered_bounds = evac_gdf.total_bounds
ax.set_xlim(buffered_bounds[0] - 400, buffered_bounds[2] + 400)
ax.set_ylim(buffered_bounds[1] - 400, buffered_bounds[3] + 400)

scalebar = ScaleBar(1, units="m", location='lower right', box_alpha=0.5, font_properties={'size': 16})
ax.add_artist(scalebar)

# Get plot bounds
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()

# Position near top-left
arrow_x = x_max - 0.05 * (x_max - x_min)
arrow_y = y_max - 0.05 * (y_max - y_min)

# Draw north arrow
ax.annotate(
    'N',
    xy=(arrow_x, arrow_y),
    xytext=(arrow_x, arrow_y - 150),  # arrow length
    arrowprops=dict(facecolor='black', width=5, headwidth=15),
    ha='center',
    va='center',
    fontsize=20,
    fontweight='bold'
)

# Save and show
plt.tight_layout()
plt.savefig(output_image, dpi=300, bbox_inches="tight", facecolor='white')
plt.show()

print(f"Map saved to: {output_image}")

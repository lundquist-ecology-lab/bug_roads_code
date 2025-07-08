import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

# Load the layers
buildings = gpd.read_file('new_york_city_buildings_26918.gpkg')
borough_boundaries = gpd.read_file('borough_boundaries_26918.gpkg')
parks_shapes = gpd.read_file('open_and_park_single_26918.gpkg')

# Reproject to EPSG:4326 (degrees)
buildings = buildings.to_crs(epsg=4326)
borough_boundaries = borough_boundaries.to_crs(epsg=4326)
parks_shapes = parks_shapes.to_crs(epsg=4326)

# Debugging output to ensure data is valid
print("Buildings CRS:", buildings.crs)
print("Borough Boundaries CRS:", borough_boundaries.crs)
print("Parks CRS:", parks_shapes.crs)
print("Buildings extent:", buildings.total_bounds)
print("Borough boundaries extent:", borough_boundaries.total_bounds)

# Reproject data to EPSG:3857 (meters)
buildings_metric = buildings.to_crs(epsg=3857)
borough_boundaries_metric = borough_boundaries.to_crs(epsg=3857)
parks_shapes_metric = parks_shapes.to_crs(epsg=3857)

# Plot the layers in the metric projection
fig, ax = plt.subplots(figsize=(15, 15))

# Plot borough boundaries
borough_boundaries_metric.plot(ax=ax, color='none', edgecolor='black', linewidth=2)

# Plot parks
parks_shapes_metric.plot(ax=ax, color='green', alpha=1, edgecolor='black', linewidth=0.5)

# Plot buildings
buildings_metric.plot(ax=ax, color='lightgrey', edgecolor='grey', linewidth=0.3, alpha=0.8)

# Add a scale bar (in meters)
scalebar = ScaleBar(1, location='lower left', units="m", dimension="si-length")  # 1 unit = 1 meter
ax.add_artist(scalebar)

# Add legend
borough_patch = mpatches.Patch(edgecolor='black', facecolor='none', label='Borough Boundary')
park_patch = mpatches.Patch(edgecolor='black', facecolor='green', label='Managed properties')
building_patch = mpatches.Patch(edgecolor='black', facecolor='lightgrey', linewidth=0.5, label='Buildings')
ax.legend(handles=[borough_patch, park_patch, building_patch], loc='upper right')

# Save the plot
plt.savefig('nyc_map_metric.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

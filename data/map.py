import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, FuncFormatter

# Define a formatter function for full number formatting
def full_number_formatter(x, pos):
    return f'{int(x):,}'  # Format with commas as thousands separators

# Load the layers
buildings = gpd.read_file('new_york_city_buildings_26918.gpkg')
borough_boundaries = gpd.read_file('borough_boundaries_26918.gpkg')
parks_shapes = gpd.read_file('open_and_park_single_26918.gpkg')

# Debugging output to ensure data is valid
print("Buildings CRS:", buildings.crs)
print("Borough Boundaries CRS:", borough_boundaries.crs)
print("Parks CRS:", parks_shapes.crs)
print("Buildings extent:", buildings.total_bounds)
print("Borough boundaries extent:", borough_boundaries.total_bounds)

# Plot the layers
fig, ax = plt.subplots(figsize=(15, 15))

# Plot borough boundaries with black outlines
borough_boundaries.plot(ax=ax, color='none', edgecolor='black', linewidth=2)

# Plot parks in green
parks_shapes.plot(ax=ax, color='green', alpha=1, edgecolor='black', linewidth=0.5)

# Plot buildings with light grey fill and black edges
buildings.plot(ax=ax, color='lightgrey', edgecolor='grey', linewidth=0.3, alpha=0.8)

# Add a scale bar
scalebar = ScaleBar(1, location='lower left')  # 1 unit = 1 meter (CRS in meters)
ax.add_artist(scalebar)

# Add a legend
borough_patch = mpatches.Patch(edgecolor='black', facecolor='none', label='Borough Boundary')
park_patch = mpatches.Patch(edgecolor='black', facecolor='green', label='Managed properties')
building_patch = mpatches.Patch(edgecolor='black', facecolor='lightgrey', linewidth=0.5, label='Buildings')
ax.legend(handles=[borough_patch, park_patch, building_patch], loc='upper right')

# Set the axis to use integer ticks (avoid decimals for better readability)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(FuncFormatter(full_number_formatter))

ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(FuncFormatter(full_number_formatter))

# Save the plot as a PNG
plt.savefig('nyc_map_no_reprojection.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

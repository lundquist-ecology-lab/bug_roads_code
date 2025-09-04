import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# Read the data files
paths = gpd.read_file('../dijkstra/results.gpkg')
boroughs = gpd.read_file('../data/borough_boundaries_26918.gpkg')

# Convert both to Web Mercator for basemap compatibility
paths = paths.to_crs(epsg=26981)
boroughs = boroughs.to_crs(epsg=26981)

# Get the total bounds from the boroughs
total_bounds = boroughs.total_bounds

# Define distance thresholds (modified to requested distances)
thresholds = [100, 250, 500, 1000]
# All line colors changed to black
colors = ['black', 'black', 'black', 'black']

# Create a figure for each threshold (individual graphs without borders or titles)
for i, threshold in enumerate(thresholds):
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='white')
    ax.set_facecolor('white')
    
    # Plot borough boundaries with white fill and black outlines
    boroughs.plot(ax=ax, color='white', edgecolor='black', linewidth=0.5)
    
    # Create mask for current threshold using the existing length column
    mask = paths['path_length'] <= threshold
    
    # Adjust line width and alpha based on threshold
    if threshold <= 100:
        linewidth = 2.0  # Thicker lines for short paths
        alpha = 0.9      # More opaque for better visibility
    elif threshold <= 250:
        linewidth = 1.5
        alpha = 0.8
    else:
        linewidth = 1.0
        alpha = 0.7
    
    # Plot the filtered paths with black color
    paths[mask].plot(ax=ax, color=colors[i], linewidth=linewidth, alpha=alpha, 
                    label=f'Paths ≤ {threshold}m')
    
    # Set the extent to the borough bounds
    ax.set_xlim(total_bounds[0], total_bounds[2])
    ax.set_ylim(total_bounds[1], total_bounds[3])
    
    # Remove axis labels and spines
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    
    # Add scale bar in bottom right
    scale_bar_length = 2000  # 2km in meters
    scale_bar_x = total_bounds[2] - (total_bounds[2] - total_bounds[0]) * 0.15  # Bottom right
    scale_bar_y = total_bounds[1] + (total_bounds[3] - total_bounds[1]) * 0.1
    
    ax.plot([scale_bar_x, scale_bar_x + scale_bar_length], 
            [scale_bar_y, scale_bar_y], 
            'k-', linewidth=2)
    ax.text(scale_bar_x + scale_bar_length/2, scale_bar_y - 2000,
            '2 km', ha='center', va='top')
    
    # Add path count
    path_count = len(paths[mask])
    ax.text(total_bounds[0] + (total_bounds[2] - total_bounds[0]) * 0.1,
            total_bounds[3] - (total_bounds[3] - total_bounds[1]) * 0.05,
            f'Number of paths: {path_count}',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7))
    
    # Save each map with a unique filename
    plt.savefig(f'../maps/nyc_paths_map_{threshold}m.png', dpi=300, bbox_inches='tight')
    plt.close()

print("Created maps have been saved as:")
for threshold in thresholds:
    print(f"../maps/nyc_paths_map_{threshold}m.png")

# Create a four-panel figure
fig, axes = plt.subplots(2, 2, figsize=(20, 20), facecolor='white')
axes = axes.flatten()

# Add spacing between subplots to make borders more visible
plt.subplots_adjust(hspace=0.1, wspace=0.1)

# Panel labels
panel_labels = ['A', 'B', 'C', 'D']

for i, threshold in enumerate(thresholds):
    ax = axes[i]
    
    # Plot borough boundaries with white fill and black outlines
    boroughs.plot(ax=ax, color='white', edgecolor='black', linewidth=0.5)
    
    # Create mask for current threshold using the existing length column
    mask = paths['path_length'] <= threshold
    
    # Adjust line width and alpha based on threshold
    if threshold <= 100:
        linewidth = 1.5  # Slightly thinner for multi-panel
        alpha = 0.9
    elif threshold <= 250:
        linewidth = 1.2
        alpha = 0.8
    else:
        linewidth = 1.0
        alpha = 0.7
    
    # Plot the filtered paths with black color
    paths[mask].plot(ax=ax, color='black', linewidth=linewidth, alpha=alpha)
    
    # Set the extent to the borough bounds
    ax.set_xlim(total_bounds[0], total_bounds[2])
    ax.set_ylim(total_bounds[1], total_bounds[3])
    
    # Add panel label without box
    ax.text(0.02, 0.98, panel_labels[i], transform=ax.transAxes, 
            fontsize=16, fontweight='bold', va='top', ha='left')
    
    # Remove axis labels
    ax.set_axis_off()
    
    # Add scale bar (2 km) in bottom right for each panel
    scale_bar_length = 2000  # 2km in meters
    scale_bar_x = total_bounds[2] - (total_bounds[2] - total_bounds[0]) * 0.1  # Bottom right
    scale_bar_y = total_bounds[1] + (total_bounds[3] - total_bounds[1]) * 0.05
    
    ax.plot([scale_bar_x, scale_bar_x + scale_bar_length], 
            [scale_bar_y, scale_bar_y], 
            'k-', linewidth=2)
    ax.text(scale_bar_x + scale_bar_length/2, scale_bar_y + 1000,
            '2 km', ha='center', va='bottom', fontsize=10)
    
    # Add path count in top right
    path_count = len(paths[mask])
    ax.text(0.98, 0.98, f'n = {path_count}', transform=ax.transAxes,
            fontsize=10, ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig('../maps/nyc_paths_four_panel.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nFour-panel figure saved as: ../maps/nyc_paths_four_panel.png")

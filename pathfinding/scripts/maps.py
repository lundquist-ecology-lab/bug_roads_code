import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# Read the data files
paths = gpd.read_file('../outputs/results.gpkg')
boroughs = gpd.read_file('../../data/borough_boundaries_26918.gpkg')

# Convert both to Web Mercator for basemap compatibility
paths = paths.to_crs(epsg=26981)
boroughs = boroughs.to_crs(epsg=26981)

# Get the total bounds from the boroughs
total_bounds = boroughs.total_bounds

# Define distance thresholds and corresponding colors
thresholds = [25, 50, 100, 500, 1000]
colors = ['#ff0000', '#ff6600', '#ffd700', '#00ff00', '#0000ff']
titles = ['Paths ≤ 25m', 'Paths ≤ 50m', 'Paths ≤ 100m', 'Paths ≤ 500m', 'Paths ≤ 1000m']

# Create a figure for each threshold
for i, threshold in enumerate(thresholds):
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='white')
    ax.set_facecolor('white')
    
    # Plot borough boundaries first with light gray fill
    boroughs.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.5, alpha=0.3)
    
    # Create mask for current threshold using the existing length column
    mask = paths['path_length'] <= threshold
    
    # Adjust line width and alpha based on threshold
    if threshold <= 50:
        linewidth = 2.0  # Thicker lines for short paths
        alpha = 0.9      # More opaque for better visibility
    elif threshold <= 100:
        linewidth = 1.5
        alpha = 0.8
    else:
        linewidth = 1.0
        alpha = 0.7
    
    # Plot the filtered paths with enhanced visibility
    paths[mask].plot(ax=ax, color=colors[i], linewidth=linewidth, alpha=alpha, 
                    label=f'Paths ≤ {threshold}m')
    
    # Set the extent to the borough bounds
    ax.set_xlim(total_bounds[0], total_bounds[2])
    ax.set_ylim(total_bounds[1], total_bounds[3])
    
    # Customize the map
    ax.set_title(titles[i], fontsize=16, pad=20)
    ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.1, 1))
    
    # Remove axis labels
    ax.set_axis_off()
    
    # Add scale bar
    scale_bar_length = 2000  # 2km in meters
    scale_bar_x = total_bounds[0] + (total_bounds[2] - total_bounds[0]) * 0.1
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
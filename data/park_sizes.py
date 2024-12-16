import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.cm import viridis
from matplotlib.colors import to_hex
import pandas as pd

# Load the GeoPackage files
file_path = "open_and_park_single_26918.gpkg"
borough_boundaries_path = "borough_boundaries_26918.gpkg"
gdf = gpd.read_file(file_path)
boroughs = gpd.read_file(borough_boundaries_path)

# Ensure CRS matches between datasets
gdf = gdf.to_crs(boroughs.crs)

# Perform spatial join to assign boroughs to each shape
gdf = gpd.sjoin(gdf, boroughs, how="left", predicate="intersects")

# Calculate the area of each shape in square meters
gdf['area_m2'] = gdf.geometry.area

# Apply log10 transformation
gdf['log_area_m2'] = np.log10(gdf['area_m2'])

# Add a 'citywide' category for overall comparison
gdf['boro_name'] = gdf['boro_name'].fillna('Citywide')

# Define the order for the boroughs and citywide comparison
borough_order = ["Citywide", "Bronx", "Manhattan", "Brooklyn", "Queens", "Staten Island"]

# Generate a more diverse Viridis color palette
colors = [to_hex(viridis(i / (len(borough_order) - 1))) for i in range(len(borough_order))]

# Examine smallest parks (showing top 10)
smallest_parks = gdf.sort_values('area_m2').head(10)[['area_m2', 'boro_name']].copy()
smallest_parks['area_m2_readable'] = smallest_parks['area_m2'].apply(lambda x: f"{x:.2f}")
print("\nSmallest parks:")
print(smallest_parks)

# Let's set a reasonable minimum threshold (e.g., 10 square meters)
min_area_threshold = 0.01  # adjust this value as needed
filtered_gdf = gdf[gdf['area_m2'] >= min_area_threshold].copy()

# Calculate what percentage of data points we're removing
total_count = len(gdf)
filtered_count = len(filtered_gdf)
removed_count = total_count - filtered_count
removed_percentage = (removed_count / total_count) * 100

print(f"\nRemoved {removed_count} shapes ({removed_percentage:.2f}% of total) smaller than {min_area_threshold} m²")

# Create the modified plot with filtered data
plt.figure(figsize=(12, 8))
sns.boxplot(x="boro_name", y="log_area_m2", data=filtered_gdf, 
            palette=colors, order=borough_order)

plt.xlabel("Borough", fontsize=10)
plt.ylabel("Log₁₀(Area) m²", fontsize=10)
plt.xticks(rotation=45)

# Get current axis
ax = plt.gca()
yticks = ax.get_yticks()
ax.set_yticks(yticks)
ax.set_yticklabels([f'{tick:.1f}' for tick in yticks])

# plt.title(f"Park Size Distribution by Borough (excluding areas < {min_area_threshold} m²)")
plt.tight_layout()

# Save the modified plot
plt.savefig("box_plot_log_shape_sizes_borough_citywide_filtered.png", dpi=300)
plt.close()

# Calculate the minimum and maximum park sizes
min_area = gdf['area_m2'].min()
max_area = gdf['area_m2'].max()

# Calculate the log10 of min and max areas
log_min_area = np.log10(min_area)
log_max_area = np.log10(max_area)

# Print the results
print(f"Minimum park area (m²): {min_area}")
print(f"Maximum park area (m²): {max_area}")
print(f"Log₁₀(Minimum park area): {log_min_area}")
print(f"Log₁₀(Maximum park area): {log_max_area}")

# Save the summary statistics to a file for record-keeping
summary_stats_path = "summary_statistics_parks.txt"
with open(summary_stats_path, 'w') as file:
    file.write(f"Minimum park area (m²): {min_area}\n")
    file.write(f"Maximum park area (m²): {max_area}\n")
    file.write(f"Log₁₀(Minimum park area): {log_min_area}\n")
    file.write(f"Log₁₀(Maximum park area): {log_max_area}\n")

print(f"Summary statistics saved to {summary_stats_path}")

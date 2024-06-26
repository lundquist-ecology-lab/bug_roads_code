import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
from tqdm import tqdm
import time
from itertools import combinations

# Load the GeoPackage files with the correct CRS
parks_gdf = gpd.read_file(r"park_centroids.gpkg")
parks_gdf = parks_gdf.to_crs(epsg=2263)  # Ensure CRS is EPSG:2263

buildings_gdf = gpd.read_file(r"new_york_city_buildings.gpkg")
buildings_gdf = buildings_gdf.to_crs(epsg=2263)  # Ensure CRS is EPSG:2263

# Ensure the necessary columns exist
if 'heightroof' not in buildings_gdf.columns:
    raise ValueError("The 'heightroof' column is not found in the GeoPackage file.")

# Convert heightroof from feet to meters
buildings_gdf['heightroof_m'] = buildings_gdf['heightroof'] * 0.3048

# Function to ensure we get a point geometry
def get_point_geometry(geom):
    if geom.geom_type == 'Point':
        return geom
    elif geom.geom_type in ['Polygon', 'MultiPolygon']:
        return geom.centroid
    else:
        raise ValueError(f"Unhandled geometry type: {geom.geom_type}")

# Prepare the CSV file
output_file_no_intersections = '../csv/paths_without_building_intersections.csv'

with open(output_file_no_intersections, 'w') as f:
    f.write('Park1_ID,Park1_name311,Park1_borough,Park2_ID,Park2_name311,Park2_borough,Distance_ft,Distance_m,Count_Intersected_Buildings\n')

# Start the timer
start_time = time.time()

# Calculate distances and check intersections with a progress bar
total_iterations = len(parks_gdf) * (len(parks_gdf) - 1) // 2  # Total number of iterations
progress_bar = tqdm(total=total_iterations, desc="Processing parks")

for (i, park1), (j, park2) in combinations(parks_gdf.iterrows(), 2):
    point1 = get_point_geometry(park1.geometry)
    point2 = get_point_geometry(park2.geometry)
    
    # Calculate the distance between the parks in feet
    distance_ft = point1.distance(point2)
    distance_m = distance_ft * 0.3048  # Convert feet to meters
    
    if distance_ft <= 3300:
        line = LineString([point1, point2])
        intersected_buildings = buildings_gdf[buildings_gdf.geometry.intersects(line)]
        
        if intersected_buildings.empty:
            result_no_intersections = {
                'Park1_ID': i,
                'Park1_name311': park1['name311'],
                'Park1_borough': park1['borough'],
                'Park2_ID': j,
                'Park2_name311': park2['name311'],
                'Park2_borough': park2['borough'],
                'Distance_ft': distance_ft,
                'Distance_m': distance_m,
                'Count_Intersected_Buildings': 0
            }

            # Append the result to the CSV file for paths without intersections
            with open(output_file_no_intersections, 'a') as f:
                f.write(','.join(map(str, result_no_intersections.values())) + '\n')

    # Update the progress bar
    progress_bar.update(1)

# Close the progress bar
progress_bar.close()

# End the timer
end_time = time.time()
elapsed_time = end_time - start_time

print(f"CSV file '{output_file_no_intersections}' has been created.")
print(f"Elapsed time: {elapsed_time:.2f} seconds.")

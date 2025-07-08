import geopandas as gpd
import numpy as np

# Load the GeoPackage file
file_path = "open_space_single_filtered_26918.gpkg"
gdf = gpd.read_file(file_path)

# Ensure the GeoDataFrame is in the correct CRS (projected for area calculations)
if not gdf.crs.is_projected:
    print("The GeoDataFrame is not in a projected CRS. Reprojecting to EPSG:26918...")
    gdf = gdf.to_crs(epsg=26918)

# Calculate the area in m²
gdf["area_m2"] = gdf.geometry.area

# Calculate total area in m² and convert to km²
total_area_m2 = gdf["area_m2"].sum()
total_area_km2 = total_area_m2 / 1_000_000

# Calculate average and standard error of the areas
average_area = gdf["area_m2"].mean()
se_area = gdf["area_m2"].std(ddof=1) / np.sqrt(len(gdf))

# Print the results
print(f"Total Area (m²): {total_area_m2}")
print(f"Total Area (km²): {total_area_km2}")
print(f"Average Area (m²): {average_area}")
print(f"Standard Error (m²): {se_area}")
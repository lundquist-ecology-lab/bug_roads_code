#%%
import geopandas as gpd

# Load the GeoPackage file
gdf = gpd.read_file("nyc_parks_shapes_fixed.gpkg")

# Ensure the columns are properly named and exist
if 'name311' not in gdf.columns or 'borough' not in gdf.columns:
    raise ValueError("The required columns 'name311' and 'borough' are not in the GeoDataFrame.")

# Calculate the total number of unique name311
total_unique_name311 = gdf['name311'].nunique()

# Calculate the number of unique name311 per borough
unique_name311_per_borough = gdf.groupby('borough')['name311'].nunique().reset_index()

# Display the results
print(f"Total unique name311: {total_unique_name311}")
print("Unique name311 per borough:")
print(unique_name311_per_borough)

# Save results to CSV if needed
unique_name311_per_borough.to_csv('../csv/unique_name311_per_borough.csv', index=False)

# %%

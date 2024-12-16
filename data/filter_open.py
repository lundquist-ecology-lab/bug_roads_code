import geopandas as gpd

def remove_cemetery_features(input_gpkg, output_gpkg):
    """
    Remove features containing 'cemetery' in the name field from a GeoPackage using GeoPandas
    
    Parameters:
    input_gpkg (str): Path to input GeoPackage file
    output_gpkg (str): Path to output GeoPackage file
    """
    # Read the GeoPackage
    gdf = gpd.read_file(input_gpkg)
    
    # Filter out features with 'cemetery' in the name (case insensitive)
    filtered_gdf = gdf[~gdf['name'].str.lower().str.contains('cemetery', na=False)]
    
    # Reset index before saving
    filtered_gdf = filtered_gdf.reset_index(drop=True)
    
    # Save to new GeoPackage
    filtered_gdf.to_file(output_gpkg, driver='GPKG', index=False)
    
    print(f"Original feature count: {len(gdf)}")
    print(f"Filtered feature count: {len(filtered_gdf)}")
    print(f"Filtered layer saved to {output_gpkg}")

# Example usage
input_gpkg = "../../data/open_space_single_26918.gpkg"
output_gpkg = "../../data/open_space_single_filtered_26918.gpkg"

remove_cemetery_features(input_gpkg, output_gpkg)
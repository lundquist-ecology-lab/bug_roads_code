import geopandas as gpd

# Read the GeoPackage file
gdf = gpd.read_file('nyc_with_water_connections.gpkg')

# Ensure the data is in EPSG:26918 (NAD83 / UTM zone 18N)
if gdf.crs != 'EPSG:26918':
    gdf = gdf.to_crs('EPSG:26918')

# Calculate area in square meters
area_m2 = gdf.geometry.area.sum()

# Convert to square kilometers
area_km2 = area_m2 / 1_000_000

print(f"Total water area: {area_km2:.2f} square kilometers")
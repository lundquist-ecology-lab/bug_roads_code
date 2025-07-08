import geopandas as gpd
from shapely.ops import unary_union
import numpy as np

def create_buffered_boundary():
    # Read NYC boroughs
    print("Reading borough boundaries...")
    boroughs = gpd.read_file('https://data.cityofnewyork.us/api/geospatial/tqmj-j8zm?method=export&format=GeoJSON')
    
    # Ensure proper projection
    boroughs = boroughs.to_crs('EPSG:26918')  # NAD83 / UTM zone 18N (meters)
    
    # Create different buffers for different connections
    print("Creating enhanced water-inclusive boundary...")
    
    # Create larger buffer around waterfront areas
    water_boundary = unary_union(boroughs.geometry).buffer(2000)  # 2km buffer
    
    # Create extra buffer for key water crossings
    manhattan = boroughs[boroughs['boro_name'] == 'Manhattan'].geometry.iloc[0]
    brooklyn = boroughs[boroughs['boro_name'] == 'Brooklyn'].geometry.iloc[0]
    queens = boroughs[boroughs['boro_name'] == 'Queens'].geometry.iloc[0]
    
    # Create specific connections for key areas
    manhattan_brooklyn = unary_union([manhattan, brooklyn]).buffer(2500)  # Extra buffer for East River
    brooklyn_queens = unary_union([brooklyn, queens]).buffer(2500)  # Extra buffer for their connection
    
    # Combine all buffers
    final_boundary = unary_union([water_boundary, manhattan_brooklyn, brooklyn_queens])
    
    # Create the boundary GeoDataFrame
    boundary_gdf = gpd.GeoDataFrame(
        geometry=[final_boundary],
        crs='EPSG:26918'
    )
    
    # Save individual borough buffers for visualization
    borough_buffers = gpd.GeoDataFrame(
        geometry=[
            manhattan_brooklyn,
            brooklyn_queens,
            water_boundary
        ],
        data={
            'buffer_type': ['Manhattan-Brooklyn', 'Brooklyn-Queens', 'General Water Buffer']
        },
        crs='EPSG:26918'
    )
    
    # Save to GPKG
    print("Saving boundary files...")
    boundary_gdf.to_file('../../data/nyc_with_water_connections.gpkg', driver='GPKG')
    borough_buffers.to_file('/../../data/water_connection_buffers.gpkg', driver='GPKG')
    
    print("""
Two files created:
1. 'nyc_with_water_connections.gpkg' - The final boundary to use for clipping
2. 'water_connection_buffers.gpkg' - Individual buffers for inspection

Please check in QGIS to verify:
1. The East River connection between Manhattan and Brooklyn is wide enough
2. The connection between Brooklyn and Queens is sufficient
3. All other water connections look appropriate
4. The extent of water coverage is suitable for your needs
    """)

if __name__ == "__main__":
    create_buffered_boundary()
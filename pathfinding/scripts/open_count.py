import geopandas as gpd
import pandas as pd

def count_path_open_space_intersections(path_file, open_space_file, park_file, borough_file):
    """
    Count intersections between paths and open spaces, broken down by borough.
    
    Parameters:
    path_file (str): Path to the GPKG file containing paths
    open_space_file (str): Path to the GPKG file containing open spaces
    park_file (str): Path to the GPKG file containing park data with unique_ids
    borough_file (str): Path to the GPKG file containing borough boundaries
    
    Returns:
    GeoDataFrame: Original paths data with additional intersection count columns
    """
    # Read the spatial data
    paths = gpd.read_file(path_file)
    open_spaces = gpd.read_file(open_space_file)
    parks = gpd.read_file(park_file)
    boroughs = gpd.read_file(borough_file)
    
    # Ensure all datasets are in the same CRS
    target_crs = paths.crs
    open_spaces = open_spaces.to_crs(target_crs)
    parks = parks.to_crs(target_crs)
    boroughs = boroughs.to_crs(target_crs)
    
    # First, associate open spaces with boroughs through parks
    # Spatial join between parks and boroughs
    parks_with_borough = gpd.sjoin(
        parks, 
        boroughs[['boro_name', 'geometry']], 
        how='left', 
        predicate='intersects'
    )
    
    # Join open spaces with parks to get borough information
    open_spaces_with_borough = open_spaces.merge(
        parks_with_borough[['unique_id', 'boro_name']], 
        on='unique_id', 
        how='left'
    )
    
    # Initialize columns for total and borough-specific counts
    paths['total_intersections'] = 0
    for borough in boroughs['boro_name'].unique():
        paths[f'intersections_{borough}'] = 0
    
    # Count intersections for each path
    for idx, path in paths.iterrows():
        # Find all open spaces that intersect with this path
        intersecting_spaces = open_spaces_with_borough[
            open_spaces_with_borough.geometry.intersects(path.geometry)
        ]
        
        # Update total count
        paths.loc[idx, 'total_intersections'] = len(intersecting_spaces)
        
        # Update borough-specific counts
        borough_counts = intersecting_spaces['boro_name'].value_counts()
        for borough, count in borough_counts.items():
            if pd.notna(borough):  # Only count if borough is not NaN
                paths.loc[idx, f'intersections_{borough}'] = count
    
    return paths

# Example usage
if __name__ == "__main__":
    # File paths
    path_file = "../outputs/results_unique.gpkg"
    open_space_file = "../../data/open_space_single_filtered_26918.gpkg"
    park_file = "../../data/open_and_park_single_26918.gpkg"
    borough_file = "../../data/borough_boundaries_26918.gpkg"
    
    # Process the data
    result = count_path_open_space_intersections(
        path_file,
        open_space_file,
        park_file,
        borough_file
    )
    
    # Print summary statistics
    print(f"Total number of paths: {len(result)}")
    print(f"Total open space intersections: {result['total_intersections'].sum()}")
    print("\nIntersections by borough:")
    borough_columns = [col for col in result.columns if col.startswith('intersections_')]
    for col in borough_columns:
        borough = col.replace('intersections_', '')
        print(f"{borough}: {result[col].sum()}")
    
    # Save as CSV, keeping all columns except geometry
    result_df = result.drop(columns=['geometry'])
    result_df.to_csv("../outputs/paths_with_open_space_counts.csv", index=False)
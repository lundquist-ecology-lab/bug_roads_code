import geopandas as gpd
import pandas as pd

def count_path_tree_intersections(path_file, tree_file, park_file, borough_file, buffer_radius=5):
    """
    Count the number of times each path intersects with street trees, broken down by borough.
    
    Parameters:
    path_file (str): Path to the GPKG file containing paths
    tree_file (str): Path to the GPKG file containing tree points
    park_file (str): Path to the GPKG file containing park data with unique_ids
    borough_file (str): Path to the GPKG file containing borough boundaries
    buffer_radius (float): Radius in meters to create buffer around trees
    
    Returns:
    GeoDataFrame: Original paths data with additional intersection count columns
    """
    # Read the spatial data
    paths = gpd.read_file(path_file)
    trees = gpd.read_file(tree_file)
    parks = gpd.read_file(park_file)
    boroughs = gpd.read_file(borough_file)
    
    # Ensure all datasets are in the same CRS
    target_crs = paths.crs
    trees = trees.to_crs(target_crs)
    parks = parks.to_crs(target_crs)
    boroughs = boroughs.to_crs(target_crs)
    
    # Create buffers around trees
    tree_buffers = trees.geometry.buffer(buffer_radius)
    trees_with_buffers = gpd.GeoDataFrame(
        geometry=tree_buffers,
        data=trees,  # Keep original tree data
        crs=trees.crs
    )
    
    # Spatial join between parks and boroughs to get borough information
    parks_with_borough = gpd.sjoin(
        parks, 
        boroughs[['boro_name', 'geometry']], 
        how='left', 
        predicate='intersects'
    )
    
    # Spatial join between trees and parks to get borough information
    trees_with_borough = gpd.sjoin(
        trees_with_buffers,
        parks_with_borough[['unique_id', 'boro_name', 'geometry']],
        how='left',
        predicate='intersects'
    )
    
    # Initialize columns for total and borough-specific counts
    paths['total_trees'] = 0
    for borough in boroughs['boro_name'].unique():
        paths[f'trees_{borough}'] = 0
    
    # Count intersections for each path
    for idx, path in paths.iterrows():
        # Find all tree buffers that intersect with this path
        intersecting_trees = trees_with_borough[
            trees_with_borough.geometry.intersects(path.geometry)
        ]
        
        # Update total count
        paths.loc[idx, 'total_trees'] = len(intersecting_trees)
        
        # Update borough-specific counts
        borough_counts = intersecting_trees['boro_name'].value_counts()
        for borough, count in borough_counts.items():
            if pd.notna(borough):  # Only count if borough is not NaN
                paths.loc[idx, f'trees_{borough}'] = count
    
    return paths

# Example usage
if __name__ == "__main__":
    # Replace with your file paths
    path_file = "../outputs/results.gpkg"
    tree_file = "../../data/street_trees_26918.gpkg"
    park_file = "../../data/open_and_park_single_26918.gpkg"
    borough_file = "../../data/borough_boundaries_26918.gpkg"
    
    # Process the data
    result = count_path_tree_intersections(
        path_file,
        tree_file,
        park_file,
        borough_file,
        buffer_radius=18
    )
    
    # Print summary statistics
    print(f"Total number of paths: {len(result)}")
    print(f"Total tree intersections: {result['total_trees'].sum()}")
    print(f"Average intersections per path: {result['total_trees'].mean():.2f}")
    print(f"Maximum intersections for a single path: {result['total_trees'].max()}")
    
    print("\nTree intersections by borough:")
    borough_columns = [col for col in result.columns if col.startswith('trees_')]
    for col in borough_columns:
        borough = col.replace('trees_', '')
        print(f"{borough}: {result[col].sum()}")
    
    # Save as CSV, keeping all columns except geometry
    result_df = result.drop(columns=['geometry'])
    result_df.to_csv("../output/paths_with_tree_counts.csv", index=False)
import geopandas as gpd
import pandas as pd
import numpy as np

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

def calculate_statistics(data, column):
    """
    Calculate mean and standard error for a given column.
    
    Parameters:
    data (Series): The data column
    column (str): Column name for labeling
    
    Returns:
    dict: Dictionary with mean and SE values
    """
    mean_val = data.mean()
    se_val = data.std() / np.sqrt(len(data)) if len(data) > 0 else 0
    return {
        'mean': mean_val,
        'se': se_val,
        'n': len(data)
    }

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
    
    # Calculate overall statistics
    total_stats = calculate_statistics(result['total_trees'], 'total_trees')
    
    # Print summary statistics with SE
    print(f"Total number of paths: {len(result)}")
    print(f"Total tree intersections: {result['total_trees'].sum()}")
    print(f"Mean intersections per path: {total_stats['mean']:.2f} ± {total_stats['se']:.2f} (SE)")
    print(f"Standard deviation: {result['total_trees'].std():.2f}")
    print(f"Maximum intersections for a single path: {result['total_trees'].max()}")
    print(f"Minimum intersections for a single path: {result['total_trees'].min()}")
    
    print("\nTree intersections by borough (with statistics):")
    borough_columns = [col for col in result.columns if col.startswith('trees_')]
    
    # Create summary table
    borough_summary = []
    
    for col in borough_columns:
        borough = col.replace('trees_', '')
        borough_data = result[col]
        stats = calculate_statistics(borough_data, col)
        
        total_count = borough_data.sum()
        print(f"{borough}:")
        print(f"  Total: {total_count}")
        print(f"  Mean ± SE: {stats['mean']:.2f} ± {stats['se']:.2f}")
        print(f"  Standard deviation: {borough_data.std():.2f}")
        
        borough_summary.append({
            'borough': borough,
            'total_intersections': total_count,
            'mean_per_path': stats['mean'],
            'se_per_path': stats['se'],
            'std_per_path': borough_data.std(),
            'n_paths': stats['n']
        })
    
    # Create and save borough summary DataFrame
    borough_summary_df = pd.DataFrame(borough_summary)
    borough_summary_df.to_csv("../outputs/borough_tree_intersection_summary.csv", index=False)
    
    # Print overall summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Metric':<25} {'Value':<15} {'SE':<10}")
    print("-"*50)
    print(f"{'Overall Mean':<25} {total_stats['mean']:<15.2f} {total_stats['se']:<10.2f}")
    
    for _, row in borough_summary_df.iterrows():
        print(f"{row['borough'] + ' Mean':<25} {row['mean_per_path']:<15.2f} {row['se_per_path']:<10.2f}")
    
    # Save detailed results as CSV, keeping all columns except geometry
    result_df = result.drop(columns=['geometry'])
    result_df.to_csv("../outputs/paths_with_tree_counts.csv", index=False)
    
    # Save summary statistics
    summary_stats = {
        'overall': {
            'total_paths': len(result),
            'total_tree_intersections': int(result['total_trees'].sum()),
            'mean_intersections_per_path': total_stats['mean'],
            'se_intersections_per_path': total_stats['se'],
            'std_intersections_per_path': result['total_trees'].std(),
            'max_intersections': int(result['total_trees'].max()),
            'min_intersections': int(result['total_trees'].min())
        },
        'by_borough': borough_summary
    }
    
    # Save as JSON for easy programmatic access
    import json
    with open("../outputs/tree_intersection_statistics.json", 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    print(f"\nFiles saved:")
    print(f"- Detailed results: ../outputs/paths_with_tree_counts.csv")
    print(f"- Borough summary: ../outputs/borough_tree_intersection_summary.csv") 
    print(f"- Statistics JSON: ../outputs/tree_intersection_statistics.json")

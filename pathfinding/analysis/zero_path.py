import geopandas as gpd
import pandas as pd
from pathlib import Path

def assign_boroughs(paths_gdf, parks_gdf, boroughs_gdf):
    """Assign borough information to paths based on park locations."""
    # Create spatial indices for faster operations
    boroughs_sindex = boroughs_gdf.sindex
    
    print("Assigning boroughs to paths...")
    # For each path, determine which borough contains the majority of the path
    def determine_primary_borough(path):
        possible_matches_idx = list(boroughs_sindex.intersection(path.geometry.bounds))
        if not possible_matches_idx:
            return None
            
        max_length = 0
        primary_borough = None
        
        for b_idx in possible_matches_idx:
            borough = boroughs_gdf.iloc[b_idx]
            intersection = path.geometry.intersection(borough.geometry)
            if not intersection.is_empty:
                length = intersection.length
                if length > max_length:
                    max_length = length
                    primary_borough = borough['boro_name']
        
        return primary_borough
    
    paths_gdf['primary_borough'] = paths_gdf.apply(lambda row: determine_primary_borough(row), axis=1)
    return paths_gdf

def analyze_zero_building_crossings(paths_gdf, parks_gdf):
    """Analyze zero building path crossings with parks."""
    print("Analyzing paths...")
    # Print the total number of paths and zero building paths
    total_paths = len(paths_gdf)
    total_zero_paths = paths_gdf['Is_Zero_Building_Path'].sum()
    print(f"Total paths: {total_paths}")
    print(f"Zero building paths: {total_zero_paths}")
    
    # Create spatial index for parks
    parks_sindex = parks_gdf.sindex
    
    # Initialize storage for crossing data
    crossing_data = []
    
    # Track unique parks crossed by each path
    path_park_counts = {}
    
    # Get only the zero building paths
    zero_building_paths = paths_gdf[paths_gdf['Is_Zero_Building_Path']]
    
    print(f"Processing {len(zero_building_paths)} zero building paths...")
    for idx, path in zero_building_paths.iterrows():
        # Find potential park intersections
        possible_matches_idx = list(parks_sindex.intersection(path.geometry.bounds))
        possible_matches = parks_gdf.iloc[possible_matches_idx]
        
        # Find actual intersections
        intersecting_parks = possible_matches[possible_matches.intersects(path.geometry)]
        if not intersecting_parks.empty:
            path_park_counts[idx] = len(intersecting_parks)
            
            for _, park in intersecting_parks.iterrows():
                intersection = path.geometry.intersection(park.geometry)
                if not intersection.is_empty:
                    crossing_data.append({
                        'path_id': idx,
                        'intersection_length_m': intersection.length,
                        'primary_borough': path['primary_borough']
                    })
    
    # Create DataFrame from results
    crossings_df = pd.DataFrame(crossing_data)
    
    # Calculate overall statistics
    overall_stats = {
        'total_paths': total_paths,
        'zero_building_paths': total_zero_paths,
        'unique_paths_with_crossings': len(path_park_counts),
        'parks_crossed': {
            'min': min(path_park_counts.values()) if path_park_counts else 0,
            'max': max(path_park_counts.values()) if path_park_counts else 0,
            'mean': sum(path_park_counts.values()) / len(path_park_counts) if path_park_counts else 0
        },
        'crossing_lengths': {
            'min': crossings_df['intersection_length_m'].min() if not crossings_df.empty else 0,
            'max': crossings_df['intersection_length_m'].max() if not crossings_df.empty else 0,
            'mean': crossings_df['intersection_length_m'].mean() if not crossings_df.empty else 0
        }
    }
    
    # Calculate borough-specific statistics
    borough_stats = {}
    for borough in paths_gdf['primary_borough'].dropna().unique():
        borough_paths = paths_gdf[paths_gdf['primary_borough'] == borough]
        borough_zero_paths = zero_building_paths[zero_building_paths['primary_borough'] == borough]
        borough_crossings = crossings_df[crossings_df['primary_borough'] == borough]
        
        # Get park counts for borough paths only
        borough_park_counts = {k: v for k, v in path_park_counts.items() 
                             if k in borough_zero_paths.index}
        
        borough_stats[borough] = {
            'total_paths': len(borough_paths),
            'zero_building_paths': len(borough_zero_paths),
            'unique_paths_with_crossings': len(set(borough_crossings['path_id'])),
            'parks_crossed': {
                'min': min(borough_park_counts.values()) if borough_park_counts else 0,
                'max': max(borough_park_counts.values()) if borough_park_counts else 0,
                'mean': sum(borough_park_counts.values()) / len(borough_park_counts) if borough_park_counts else 0
            },
            'crossing_lengths': {
                'min': borough_crossings['intersection_length_m'].min() if not borough_crossings.empty else 0,
                'max': borough_crossings['intersection_length_m'].max() if not borough_crossings.empty else 0,
                'mean': borough_crossings['intersection_length_m'].mean() if not borough_crossings.empty else 0
            }
        }
    
    return overall_stats, borough_stats

def save_results(overall_stats, borough_stats, output_path):
    """Save analysis results to a file."""
    with open(output_path, 'w') as f:
        # Write overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("==================\n")
        f.write(f"Zero Building Paths: {overall_stats['zero_building_paths']} out of {overall_stats['total_paths']}\n")
        f.write(f"Unique Paths Crossing Parks: {overall_stats['unique_paths_with_crossings']}\n")
        f.write(f"Parks Crossed per Path: min={overall_stats['parks_crossed']['min']:.0f}, ")
        f.write(f"max={overall_stats['parks_crossed']['max']:.0f}, ")
        f.write(f"mean={overall_stats['parks_crossed']['mean']:.1f}\n")
        f.write(f"Crossing Lengths (m): min={overall_stats['crossing_lengths']['min']:.1f}, ")
        f.write(f"max={overall_stats['crossing_lengths']['max']:.1f}, ")
        f.write(f"mean={overall_stats['crossing_lengths']['mean']:.1f}\n")
        
        # Write borough statistics
        f.write("\nBOROUGH STATISTICS\n")
        f.write("==================\n")
        for borough, stats in borough_stats.items():
            f.write(f"\n{borough}\n")
            f.write(f"Zero Building Paths: {stats['zero_building_paths']} out of {stats['total_paths']}\n")
            f.write(f"Unique Paths Crossing Parks: {stats['unique_paths_with_crossings']}\n")
            f.write(f"Parks Crossed per Path: min={stats['parks_crossed']['min']:.0f}, ")
            f.write(f"max={stats['parks_crossed']['max']:.0f}, ")
            f.write(f"mean={stats['parks_crossed']['mean']:.1f}\n")
            f.write(f"Crossing Lengths (m): min={stats['crossing_lengths']['min']:.1f}, ")
            f.write(f"max={stats['crossing_lengths']['max']:.1f}, ")
            f.write(f"mean={stats['crossing_lengths']['mean']:.1f}\n")

def main():
    base_dir = Path(__file__).parent
    gpkg_path = base_dir.parent / "output" / "park_straight_line_paths_visualization.gpkg"
    parks_gpkg_path = base_dir.parent.parent / "data" / "open_and_park_single_26918.gpkg"
    borough_gpkg_path = base_dir.parent.parent / "data" / "borough_boundaries_26918.gpkg"
    output_path = base_dir.parent / "output" / "zero_building_park_crossing_analysis.txt"
    
    # Load data
    print("Loading data...")
    paths_gdf = gpd.read_file(gpkg_path, layer='paths')
    parks_gdf = gpd.read_file(parks_gpkg_path)
    boroughs_gdf = gpd.read_file(borough_gpkg_path)
    
    # Assign boroughs to paths
    paths_gdf = assign_boroughs(paths_gdf, parks_gdf, boroughs_gdf)
    
    # Run analysis
    print("Running analysis...")
    overall_stats, borough_stats = analyze_zero_building_crossings(paths_gdf, parks_gdf)
    
    # Save results to file
    print("Saving results...")
    save_results(overall_stats, borough_stats, output_path)
    print(f"Analysis complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
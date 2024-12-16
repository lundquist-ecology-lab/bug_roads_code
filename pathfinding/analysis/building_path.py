import geopandas as gpd
import pandas as pd
from pathlib import Path

def assign_boroughs(paths_gdf, parks_gdf, boroughs_gdf):
    """Assign borough information to paths based on park locations."""
    # Create spatial indices for faster operations
    boroughs_sindex = boroughs_gdf.sindex
    
    print("Assigning boroughs to paths...")
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

def analyze_building_path_crossings(paths_gdf, parks_gdf):
    """Analyze park crossings for paths that intersect with buildings."""
    print("Analyzing paths...")
    # Print the total number of paths and building-intersecting paths
    total_paths = len(paths_gdf)
    total_building_paths = (~paths_gdf['Is_Zero_Building_Path']).sum()
    print(f"Total paths: {total_paths}")
    print(f"Building-intersecting paths: {total_building_paths}")
    
    # Create spatial index for parks
    parks_sindex = parks_gdf.sindex
    
    # Initialize storage for crossing data
    crossing_data = []
    
    # Track unique parks crossed by each path
    path_park_counts = {}
    
    # Get only the building-intersecting paths
    building_paths = paths_gdf[~paths_gdf['Is_Zero_Building_Path']]
    
    print(f"Processing {len(building_paths)} building-intersecting paths...")
    for idx, path in building_paths.iterrows():
        # Find potential park intersections
        possible_matches_idx = list(parks_sindex.intersection(path.geometry.bounds))
        possible_matches = parks_gdf.iloc[possible_matches_idx]
        
        # Find actual intersections
        intersecting_parks = possible_matches[possible_matches.intersects(path.geometry)]
        if not intersecting_parks.empty:
            path_park_counts[idx] = len(intersecting_parks)
            
            for park_idx, park in intersecting_parks.iterrows():
                intersection = path.geometry.intersection(park.geometry)
                if not intersection.is_empty:
                    crossing_data.append({
                        'path_id': idx,
                        'park_id': park_idx,  # Using park index instead of name
                        'intersection_length_m': intersection.length,
                        'primary_borough': path['primary_borough'],
                        'total_path_length': path.geometry.length,
                        'park_portion': intersection.length / path.geometry.length
                    })
    
    # Create DataFrame from results
    crossings_df = pd.DataFrame(crossing_data)
    
    # Calculate overall statistics
    overall_stats = {
        'total_paths': total_paths,
        'building_intersecting_paths': total_building_paths,
        'paths_with_park_crossings': len(path_park_counts),
        'parks_crossed': {
            'min': min(path_park_counts.values()) if path_park_counts else 0,
            'max': max(path_park_counts.values()) if path_park_counts else 0,
            'mean': sum(path_park_counts.values()) / len(path_park_counts) if path_park_counts else 0
        },
        'crossing_lengths': {
            'min': crossings_df['intersection_length_m'].min() if not crossings_df.empty else 0,
            'max': crossings_df['intersection_length_m'].max() if not crossings_df.empty else 0,
            'mean': crossings_df['intersection_length_m'].mean() if not crossings_df.empty else 0
        },
        'park_portion': {
            'min': crossings_df['park_portion'].min() if not crossings_df.empty else 0,
            'max': crossings_df['park_portion'].max() if not crossings_df.empty else 0,
            'mean': crossings_df['park_portion'].mean() if not crossings_df.empty else 0
        }
    }
    
    # Calculate borough-specific statistics
    borough_stats = {}
    for borough in paths_gdf['primary_borough'].dropna().unique():
        borough_paths = paths_gdf[paths_gdf['primary_borough'] == borough]
        borough_building_paths = building_paths[building_paths['primary_borough'] == borough]
        borough_crossings = crossings_df[crossings_df['primary_borough'] == borough]
        
        borough_park_counts = {k: v for k, v in path_park_counts.items() 
                             if k in borough_building_paths.index}
        
        borough_stats[borough] = {
            'total_paths': len(borough_paths),
            'building_intersecting_paths': len(borough_building_paths),
            'paths_with_park_crossings': len(set(borough_crossings['path_id'])),
            'parks_crossed': {
                'min': min(borough_park_counts.values()) if borough_park_counts else 0,
                'max': max(borough_park_counts.values()) if borough_park_counts else 0,
                'mean': sum(borough_park_counts.values()) / len(borough_park_counts) if borough_park_counts else 0
            },
            'crossing_lengths': {
                'min': borough_crossings['intersection_length_m'].min() if not borough_crossings.empty else 0,
                'max': borough_crossings['intersection_length_m'].max() if not borough_crossings.empty else 0,
                'mean': borough_crossings['intersection_length_m'].mean() if not borough_crossings.empty else 0
            },
            'park_portion': {
                'min': borough_crossings['park_portion'].min() if not borough_crossings.empty else 0,
                'max': borough_crossings['park_portion'].max() if not borough_crossings.empty else 0,
                'mean': borough_crossings['park_portion'].mean() if not borough_crossings.empty else 0
            }
        }
    
    # Additional analysis: Most frequently crossed parks by ID
    if not crossings_df.empty:
        park_frequency = crossings_df['park_id'].value_counts().head(10)
        overall_stats['most_crossed_parks'] = park_frequency.to_dict()
    
    return overall_stats, borough_stats, crossings_df

def save_results(overall_stats, borough_stats, crossings_df, output_path):
    """Save analysis results to files."""
    # Save main statistics
    with open(output_path, 'w') as f:
        f.write("OVERALL STATISTICS\n")
        f.write("==================\n")
        f.write(f"Building-Intersecting Paths: {overall_stats['building_intersecting_paths']} out of {overall_stats['total_paths']}\n")
        f.write(f"Paths Crossing Parks: {overall_stats['paths_with_park_crossings']}\n")
        f.write(f"Parks Crossed per Path: min={overall_stats['parks_crossed']['min']:.0f}, ")
        f.write(f"max={overall_stats['parks_crossed']['max']:.0f}, ")
        f.write(f"mean={overall_stats['parks_crossed']['mean']:.1f}\n")
        f.write(f"Crossing Lengths (m): min={overall_stats['crossing_lengths']['min']:.1f}, ")
        f.write(f"max={overall_stats['crossing_lengths']['max']:.1f}, ")
        f.write(f"mean={overall_stats['crossing_lengths']['mean']:.1f}\n")
        f.write(f"Portion of Path in Parks: min={overall_stats['park_portion']['min']:.2%}, ")
        f.write(f"max={overall_stats['park_portion']['max']:.2%}, ")
        f.write(f"mean={overall_stats['park_portion']['mean']:.2%}\n")
        
        if 'most_crossed_parks' in overall_stats:
            f.write("\nMost Frequently Crossed Parks (by ID):\n")
            for park_id, count in overall_stats['most_crossed_parks'].items():
                f.write(f"Park ID {park_id}: {count} crossings\n")
        
        f.write("\nBOROUGH STATISTICS\n")
        f.write("==================\n")
        for borough, stats in borough_stats.items():
            f.write(f"\n{borough}\n")
            f.write(f"Building-Intersecting Paths: {stats['building_intersecting_paths']} out of {stats['total_paths']}\n")
            f.write(f"Paths Crossing Parks: {stats['paths_with_park_crossings']}\n")
            f.write(f"Parks Crossed per Path: min={stats['parks_crossed']['min']:.0f}, ")
            f.write(f"max={stats['parks_crossed']['max']:.0f}, ")
            f.write(f"mean={stats['parks_crossed']['mean']:.1f}\n")
            f.write(f"Crossing Lengths (m): min={stats['crossing_lengths']['min']:.1f}, ")
            f.write(f"max={stats['crossing_lengths']['max']:.1f}, ")
            f.write(f"mean={stats['crossing_lengths']['mean']:.1f}\n")
            f.write(f"Portion of Path in Parks: min={stats['park_portion']['min']:.2%}, ")
            f.write(f"max={stats['park_portion']['max']:.2%}, ")
            f.write(f"mean={stats['park_portion']['mean']:.2%}\n")
    
    # Save detailed crossings data to CSV
    csv_path = output_path.parent / "building_path_park_crossings_details.csv"
    crossings_df.to_csv(csv_path, index=False)

def main():
    base_dir = Path(__file__).parent
    gpkg_path = base_dir.parent / "output" / "park_straight_line_paths_visualization.gpkg"
    parks_gpkg_path = base_dir.parent.parent / "data" / "open_and_park_single_26918.gpkg"
    borough_gpkg_path = base_dir.parent.parent / "data" / "borough_boundaries_26918.gpkg"
    output_path = base_dir.parent / "output" / "building_path_park_crossing_analysis.txt"
    
    print("Loading data...")
    paths_gdf = gpd.read_file(gpkg_path, layer='paths')
    parks_gdf = gpd.read_file(parks_gpkg_path)
    boroughs_gdf = gpd.read_file(borough_gpkg_path)
    
    # Print available columns in parks_gdf
    print("\nAvailable columns in parks dataset:")
    print(parks_gdf.columns)
    
    paths_gdf = assign_boroughs(paths_gdf, parks_gdf, boroughs_gdf)
    
    print("Running analysis...")
    overall_stats, borough_stats, crossings_df = analyze_building_path_crossings(paths_gdf, parks_gdf)
    
    print("Saving results...")
    save_results(overall_stats, borough_stats, crossings_df, output_path)
    print(f"Analysis complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
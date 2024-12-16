import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from shapely.geometry import LineString, Point
import math

def calculate_bearing(point1, point2):
    """Calculate the bearing between two points"""
    x1, y1 = point1
    x2, y2 = point2
    angle = math.degrees(math.atan2(x2 - x1, y2 - y1))
    return angle

def count_direction_changes(linestring, threshold=30):
    """Count significant direction changes in a path
    threshold: minimum angle change to count as a turn (degrees)"""
    coords = list(linestring.coords)
    if len(coords) < 3:
        return 0
    
    turns = 0
    prev_bearing = calculate_bearing(coords[0], coords[1])
    
    for i in range(1, len(coords)-1):
        current_bearing = calculate_bearing(coords[i], coords[i+1])
        angle_diff = abs(((current_bearing - prev_bearing + 180) % 360) - 180)
        
        if angle_diff > threshold:
            turns += 1
        
        prev_bearing = current_bearing
    
    return turns

def calculate_sinuosity(linestring):
    """Calculate path sinuosity (ratio of path length to straight-line distance)"""
    if linestring.length == 0:
        return 0
        
    start_point = Point(linestring.coords[0])
    end_point = Point(linestring.coords[-1])
    straight_distance = start_point.distance(end_point)
    
    if straight_distance == 0:
        return 1
        
    return linestring.length / straight_distance

def format_stats(stats, title):
    """Format statistics into a readable text block"""
    output = f"\n{title}\n{'=' * len(title)}\n"
    output += f"Number of Paths: {stats['count']:,.0f}\n"
    output += f"Mean: {stats['mean']:,.4f}\n"
    output += f"Standard Error: {stats['se']:,.6f}\n"
    output += f"Median: {stats['median']:,.4f}\n"
    output += f"Standard Deviation: {stats['std']:,.4f}\n"
    output += f"Minimum: {stats['min']:,.4f}\n"
    output += f"Maximum: {stats['max']:,.4f}\n"
    output += f"25th Percentile: {stats['q25']:,.4f}\n"
    output += f"75th Percentile: {stats['q75']:,.4f}\n"
    return output

def get_summary_stats(data):
    return pd.Series({
        'count': len(data),
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
        'se': data.std() / np.sqrt(len(data)),
        'min': data.min(),
        'max': data.max(),
        'q25': data.quantile(0.25),
        'q75': data.quantile(0.75)
    })

def analyze_path_complexity():
    # Configure logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('../log/complexity_analysis.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    try:
        # Load the path results
        logger.info("Loading path results...")
        paths_gdf = gpd.read_file('../output/results_unique.gpkg')
        
        # Filter paths to <= 1000m
        logger.info("Filtering paths to <= 1000m...")
        paths_gdf = paths_gdf[paths_gdf['path_length'] <= 1000]
        logger.info(f"Number of paths <= 1000m: {len(paths_gdf)}")
        
        # Load borough boundaries and parks
        logger.info("Loading borough boundaries...")
        boroughs_gdf = gpd.read_file('../../data/borough_boundaries_26918.gpkg')
        
        logger.info("Loading parks data...")
        parks_gdf = gpd.read_file('../../data/open_and_park_single_26918.gpkg')
        
        # Create park ID to borough mapping
        logger.info("Assigning parks to boroughs...")
        park_centroids = parks_gdf.copy()
        park_centroids['geometry'] = park_centroids['geometry'].centroid
        parks_with_borough = gpd.sjoin(park_centroids, boroughs_gdf[['boro_name', 'geometry']], 
                                     how='left', predicate='within')
        park_borough_dict = parks_with_borough.set_index('unique_id')['boro_name'].to_dict()
        
        # Add borough information to paths
        paths_gdf['park1_borough'] = paths_gdf['park1_unique_id'].map(park_borough_dict)
        paths_gdf['park2_borough'] = paths_gdf['park2_unique_id'].map(park_borough_dict)
        
        # Calculate complexity metrics
        logger.info("Calculating complexity metrics...")
        paths_gdf['num_segments'] = paths_gdf['geometry'].apply(lambda x: len(list(x.coords)) - 1)
        paths_gdf['avg_segment_length'] = paths_gdf['path_length'] / paths_gdf['num_segments']
        paths_gdf['num_turns'] = paths_gdf['geometry'].apply(count_direction_changes)
        paths_gdf['sinuosity'] = paths_gdf['geometry'].apply(calculate_sinuosity)
        
        def get_segment_lengths_cv(linestring):
            coords = list(linestring.coords)
            if len(coords) < 2:
                return 0
            segments = [Point(coords[i]).distance(Point(coords[i+1])) 
                       for i in range(len(coords)-1)]
            return np.std(segments) / np.mean(segments) if np.mean(segments) > 0 else 0
            
        paths_gdf['segment_length_cv'] = paths_gdf['geometry'].apply(get_segment_lengths_cv)

        # Create output directory
        output_dir = Path('../output')
        output_dir.mkdir(exist_ok=True)

        # Generate timestamp for the analysis
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Define metrics
        metrics = {
            'Number of Segments': 'num_segments',
            'Average Segment Length (m)': 'avg_segment_length',
            'Number of Turns': 'num_turns',
            'Path Sinuosity': 'sinuosity',
            'Segment Length Coefficient of Variation': 'segment_length_cv'
        }

        # Write results
        with open(output_dir / 'path_complexity_analysis_1000m.txt', 'w', encoding='utf-8') as f:
            f.write(f"Path Complexity Analysis Results (Paths <= 1000m)\n")
            f.write(f"Generated on: {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            
            # Citywide statistics
            f.write("CITYWIDE STATISTICS\n")
            f.write("==================\n")
            for metric_name, column in metrics.items():
                stats = get_summary_stats(paths_gdf[column])
                f.write(format_stats(stats, metric_name))
            
            # Within-borough statistics
            f.write("\nWITHIN-BOROUGH STATISTICS\n")
            f.write("========================\n")
            within_borough_paths = paths_gdf[paths_gdf['park1_borough'] == paths_gdf['park2_borough']]
            
            unique_boroughs = [b for b in paths_gdf['park1_borough'].unique() if pd.notna(b)]
            unique_boroughs.sort()
            
            for borough in unique_boroughs:
                f.write(f"\n{borough}\n")
                f.write("-" * len(borough) + "\n")
                borough_data = within_borough_paths[within_borough_paths['park1_borough'] == borough]
                
                if len(borough_data) > 0:
                    for metric_name, column in metrics.items():
                        stats = get_summary_stats(borough_data[column])
                        f.write(format_stats(stats, metric_name))
            
            # Between-borough statistics
            f.write("\nBETWEEN-BOROUGH STATISTICS\n")
            f.write("=========================\n")
            between_borough_paths = paths_gdf[paths_gdf['park1_borough'] != paths_gdf['park2_borough']]
            
            for i, boro1 in enumerate(unique_boroughs):
                for boro2 in unique_boroughs[i+1:]:
                    between_data = between_borough_paths[
                        ((between_borough_paths['park1_borough'] == boro1) & 
                         (between_borough_paths['park2_borough'] == boro2)) |
                        ((between_borough_paths['park1_borough'] == boro2) & 
                         (between_borough_paths['park2_borough'] == boro1))
                    ]
                    
                    if len(between_data) > 0:
                        f.write(f"\n{boro1} <-> {boro2}\n")
                        f.write("-" * len(f"{boro1} <-> {boro2}") + "\n")
                        for metric_name, column in metrics.items():
                            stats = get_summary_stats(between_data[column])
                            f.write(format_stats(stats, metric_name))

        logger.info("Analysis completed successfully. Results written to path_complexity_analysis_1000m.txt")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    analyze_path_complexity()
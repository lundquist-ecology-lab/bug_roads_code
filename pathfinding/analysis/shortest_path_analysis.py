import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def format_stats(stats, title):
    """Format statistics into a readable text block"""
    output = f"\n{title}\n{'=' * len(title)}\n"
    output += f"Number of Paths: {stats['count']:,.0f}\n"
    output += f"Mean Distance: {stats['mean']:,.2f} meters\n"
    output += f"Standard Error: {stats['se']:,.2f} meters\n"
    output += f"Median Distance: {stats['median']:,.2f} meters\n"
    output += f"Standard Deviation: {stats['std']:,.2f} meters\n"
    output += f"Minimum Distance: {stats['min']:,.2f} meters\n"
    output += f"Maximum Distance: {stats['max']:,.2f} meters\n"
    output += f"25th Percentile: {stats['q25']:,.2f} meters\n"
    output += f"75th Percentile: {stats['q75']:,.2f} meters\n"
    return output

def analyze_path_distances():
    # Configure logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('../log/path_analysis.log', encoding='utf-8'),
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
        
        # Load borough boundaries
        logger.info("Loading borough boundaries...")
        boroughs_gdf = gpd.read_file('../../data/borough_boundaries_26918.gpkg')
        
        # Load parks to get their centroids for borough assignment
        logger.info("Loading parks data...")
        parks_gdf = gpd.read_file('../../data/open_and_park_single_26918.gpkg')
        
        # Create a dictionary to map park IDs to boroughs
        logger.info("Assigning parks to boroughs...")
        park_centroids = parks_gdf.copy()
        park_centroids['geometry'] = park_centroids['geometry'].centroid
        
        # Spatial join to get borough for each park
        parks_with_borough = gpd.sjoin(park_centroids, boroughs_gdf[['boro_name', 'geometry']], 
                                     how='left', predicate='within')
        
        # Create park ID to borough mapping
        park_borough_dict = parks_with_borough.set_index('unique_id')['boro_name'].to_dict()
        
        # Add borough information to paths
        paths_gdf['park1_borough'] = paths_gdf['park1_unique_id'].map(park_borough_dict)
        paths_gdf['park2_borough'] = paths_gdf['park2_unique_id'].map(park_borough_dict)
        
        # Function to calculate summary statistics
        def get_summary_stats(data):
            return pd.Series({
                'count': len(data),
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'se': data.std() / np.sqrt(len(data)),  # Standard Error
                'min': data.min(),
                'max': data.max(),
                'q25': data.quantile(0.25),
                'q75': data.quantile(0.75)
            })

        # Create output directory
        output_dir = Path('../output')
        output_dir.mkdir(exist_ok=True)

        # Generate timestamp for the analysis
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Open main output file with UTF-8 encoding
        with open(output_dir / 'path_distance_analysis_1000m.txt', 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"Path Distance Analysis Results (Paths <= 1000m)\n")
            f.write(f"Generated on: {timestamp}\n")
            f.write("=" * 50 + "\n\n")

            # Calculate and write citywide statistics
            logger.info("Calculating citywide statistics...")
            citywide_stats = get_summary_stats(paths_gdf['path_length'])
            f.write(format_stats(citywide_stats, "Citywide Statistics"))

            # Calculate and write within-borough statistics
            logger.info("Calculating within-borough statistics...")
            f.write("\nWithin-Borough Statistics\n")
            f.write("=" * 23 + "\n")
            within_borough_paths = paths_gdf[paths_gdf['park1_borough'] == paths_gdf['park2_borough']]
            
            # Get unique boroughs, excluding NaN values
            unique_boroughs = [b for b in paths_gdf['park1_borough'].unique() if pd.notna(b)]
            unique_boroughs.sort()  # Sort the valid borough names
            
            for borough in unique_boroughs:
                borough_data = within_borough_paths[within_borough_paths['park1_borough'] == borough]
                if len(borough_data) > 0:
                    borough_stats = get_summary_stats(borough_data['path_length'])
                    f.write(format_stats(borough_stats, f"{borough}"))

            # Calculate and write between-borough statistics
            logger.info("Calculating between-borough statistics...")
            f.write("\nBetween-Borough Statistics\n")
            f.write("=" * 25 + "\n")
            between_borough_paths = paths_gdf[paths_gdf['park1_borough'] != paths_gdf['park2_borough']]
            
            for i, boro1 in enumerate(unique_boroughs):
                for boro2 in unique_boroughs[i+1:]:  # This ensures we only get each pair once
                    between_data = between_borough_paths[
                        ((between_borough_paths['park1_borough'] == boro1) & 
                         (between_borough_paths['park2_borough'] == boro2)) |
                        ((between_borough_paths['park1_borough'] == boro2) & 
                         (between_borough_paths['park2_borough'] == boro1))
                    ]
                    if len(between_data) > 0:
                        between_stats = get_summary_stats(between_data['path_length'])
                        f.write(format_stats(between_stats, f"{boro1} <-> {boro2}"))

        logger.info("Analysis completed successfully. Results written to path_distance_analysis_1000m.txt")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    analyze_path_distances()
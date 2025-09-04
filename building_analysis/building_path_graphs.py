import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy.stats import sem
import geopandas as gpd
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)

def check_environment():
    """Check if required packages are installed with correct versions."""
    try:
        import geopandas
        import pandas
        import numpy
        print(f"geopandas version: {geopandas.__version__}")
        print(f"pandas version: {pandas.__version__}")
        print(f"numpy version: {numpy.__version__}")
        return True
    except ImportError as e:
        print(f"Missing required package: {str(e)}")
        return False

def load_and_filter_paths(results_gpkg_path, max_distance=1000):
    """Load paths from results.gpkg and filter by Euclidean distance."""
    logging.info("Loading path results...")
    paths_gdf = gpd.read_file(results_gpkg_path, layer="results_point_id_fixed")
    
    logging.info(f"Total paths before filtering: {len(paths_gdf)}")
    
    # Filter paths by Euclidean distance
    logging.info(f"Filtering paths to <= {max_distance}m...")
    paths_gdf = paths_gdf[paths_gdf["euclidean_distance"] <= max_distance]
    
    logging.info(f"Number of paths after filtering: {len(paths_gdf)}")
    
    # Rename columns to match expected format
    paths_gdf = paths_gdf.rename(columns={
        'park1_point_id': 'Park1_point_id',
        'park2_point_id': 'Park2_point_id'
    })
    
    return paths_gdf

def process_intersection_data(intersections_gpkg_path, filtered_paths_gdf):
    """Load and process intersection data from GeoPackage, filtering by valid paths."""
    # Read intersections from the correct layer - "intersections"
    try:
        intersections_gdf = gpd.read_file(intersections_gpkg_path, layer='intersections')
        print("Successfully read from 'intersections' layer")
    except Exception as e:
        print(f"Could not read 'intersections' layer, trying main layer: {e}")
        intersections_gdf = gpd.read_file(intersections_gpkg_path)
    
    # Print available columns to debug
    print("Available columns in intersections data:", intersections_gdf.columns.tolist())
    print("Sample of intersections data:")
    print(intersections_gdf.head())
    
    # Filter intersections to only include those for valid paths
    valid_path_tuples = set(zip(filtered_paths_gdf['Park1_point_id'], 
                               filtered_paths_gdf['Park2_point_id']))
    
    # Filter intersections
    intersections_gdf = intersections_gdf[
        intersections_gdf.apply(lambda row: (row['Park1_point_id'], row['Park2_point_id']) in valid_path_tuples, axis=1)
    ]
    
    logging.info(f"Intersections after filtering: {len(intersections_gdf)}")
    
    # Group intersections by path identifiers
    if 'Park1_point_id' in intersections_gdf.columns and 'Park2_point_id' in intersections_gdf.columns:
        path_groups = intersections_gdf.groupby(['Park1_point_id', 'Park2_point_id'])
    else:
        raise KeyError("Expected 'Park1_point_id' and 'Park2_point_id' in intersections data but they were not found.")
    
    # Create a dictionary to store building heights for each path
    path_heights = defaultdict(list)
    
    # Process each path's intersections
    for (park1_id, park2_id), group in path_groups:
        # Get unique buildings and their heights
        if 'building_id' in group.columns:
            unique_buildings = group.drop_duplicates('building_id')
        else:
            # If no building_id, use all rows (assuming each row is a unique building)
            unique_buildings = group
            
        if 'height' in group.columns:
            heights = unique_buildings['height'].tolist()
            print(f"Found height column with sample values: {heights[:5] if heights else 'empty'}")
        elif 'height_m' in group.columns:
            heights = unique_buildings['height_m'].tolist()
            print(f"Found height_m column with sample values: {heights[:5] if heights else 'empty'}")
        elif 'roof_height' in group.columns:
            heights = unique_buildings['roof_height'].tolist()
            print(f"Found roof_height column with sample values: {heights[:5] if heights else 'empty'}")
        else:
            print("Warning: No height column found, using default value of 10m")
            heights = [10.0] * len(unique_buildings)
            
        path_heights[(park1_id, park2_id)] = heights
    
    # Add heights to filtered paths dataframe
    filtered_paths_gdf['Intersected_heightroofs_m'] = filtered_paths_gdf.apply(
        lambda row: path_heights.get((row['Park1_point_id'], row['Park2_point_id']), []),
        axis=1
    )
    
    # Add count of intersected buildings if not already present
    if 'Count_Intersected_Buildings' not in filtered_paths_gdf.columns:
        filtered_paths_gdf['Count_Intersected_Buildings'] = filtered_paths_gdf['Intersected_heightroofs_m'].apply(len)
    
    return filtered_paths_gdf

def add_borough_info(paths_gdf, parks_gpkg_path, borough_gpkg_path):
    """Add borough information to paths GeoDataFrame based on spatial location."""
    try:
        parks_gdf = gpd.read_file(parks_gpkg_path)
        boroughs_gdf = gpd.read_file(borough_gpkg_path)
        
        # Print borough information to debug
        print("Available borough columns:", boroughs_gdf.columns.tolist())
        print("Borough values:", boroughs_gdf['boro_name'].unique() if 'boro_name' in boroughs_gdf.columns else "No 'boro_name' column found")
        
    except Exception as e:
        print(f"Error reading input files: {str(e)}")
        raise
    
    # Create dictionary to store park_id to borough mappings
    park_borough_dict = {}
    
    # Spatial join for Park1 unique IDs
    park1_ids = paths_gdf['Park1_point_id'].unique()
    park1_locations = parks_gdf[parks_gdf['point_id'].isin(park1_ids)]
    park1_boroughs = gpd.sjoin(park1_locations, boroughs_gdf, how='left', predicate='intersects')
    
    # Determine the correct borough name column
    borough_col = next((col for col in park1_boroughs.columns if 'boro' in col.lower()), None)
    if borough_col is None:
        raise ValueError("Could not find borough column in spatial join result")
        
    print(f"\nUsing column '{borough_col}' for borough names")
    
    # Store Park1 borough mappings
    for _, row in park1_boroughs.iterrows():
        park_borough_dict[row['point_id']] = row[borough_col]
    
    # Spatial join for Park2 unique IDs
    park2_ids = paths_gdf['Park2_point_id'].unique()
    park2_locations = parks_gdf[parks_gdf['point_id'].isin(park2_ids)]
    park2_boroughs = gpd.sjoin(park2_locations, boroughs_gdf, how='left', predicate='intersects')
    
    # Store Park2 borough mappings
    for _, row in park2_boroughs.iterrows():
        park_borough_dict[row['point_id']] = row[borough_col]
    
    # Add borough columns to original dataframe
    paths_gdf['Park1_borough'] = paths_gdf['Park1_point_id'].map(park_borough_dict)
    paths_gdf['Park2_borough'] = paths_gdf['Park2_point_id'].map(park_borough_dict)
    
    return paths_gdf

def clean_and_process_data(gdf):
    """Clean and process the GeoDataFrame for analysis."""
    # Create a copy of the dataframe
    gdf = gdf.copy()
    
    # Print initial state using both metrics
    print("\nBefore processing:")
    no_buildings_count = (gdf['Count_Intersected_Buildings'] == 0).sum()
    no_heights_count = gdf['Intersected_heightroofs_m'].apply(lambda x: len(x) == 0).sum()
    print(f"Paths with no buildings (Count column): {no_buildings_count}")
    print(f"Paths with empty height lists: {no_heights_count}")
    
    # Check for inconsistencies
    inconsistent_paths = gdf[
        (gdf['Count_Intersected_Buildings'] > 0) & 
        (gdf['Intersected_heightroofs_m'].apply(len) == 0)
    ]
    if len(inconsistent_paths) > 0:
        print(f"\nFound {len(inconsistent_paths)} paths with Count > 0 but no height data")
    
    # Process height data
    max_length = max(len(heights) for heights in gdf['Intersected_heightroofs_m']) if len(gdf) > 0 else 0
    
    if max_length > 0:
        # Pad lists with NaNs, but only for paths that actually have buildings
        gdf['Intersected_heightroofs_m'] = gdf.apply(
            lambda row: (row['Intersected_heightroofs_m'] + [np.nan] * (max_length - len(row['Intersected_heightroofs_m'])))
            if row['Count_Intersected_Buildings'] > 0 and len(row['Intersected_heightroofs_m']) > 0
            else [], axis=1
        )
    
    # Print final state
    print("\nAfter processing:")
    final_no_buildings = gdf['Intersected_heightroofs_m'].apply(
        lambda x: len(x) == 0 or all(np.isnan(h) for h in x)
    ).sum()
    print(f"Paths with no buildings: {final_no_buildings}")
    
    return gdf

def plot_summary_roof_heights_and_heatmap(ax, df, title, tick_interval):
    """Create summary plots for roof heights, including zero-building paths."""
    if df.empty:
        print(f"No data available for {title}")
        return

    # Calculate number of buildings for ALL paths (including zero)
    num_buildings = df['Intersected_heightroofs_m'].apply(
        lambda x: len([h for h in x if not np.isnan(h)])
    ).values
    max_num_buildings = num_buildings.max() if len(num_buildings) > 0 else 0
    building_counts = np.bincount(num_buildings.astype(int), minlength=max_num_buildings + 1)

    # Filter out rows with no buildings for roof height statistics
    df_with_buildings = df[num_buildings > 0].copy()
    if df_with_buildings.empty:
        print(f"No paths with buildings available for {title}")
        # Still draw heatmap with zero column if possible
        if len(building_counts) > 0:
            sns.heatmap([building_counts], cmap="Blues",
                        ax=ax[1], cbar=True, xticklabels=True, yticklabels=False,
                        cbar_kws={'orientation': 'vertical'}, annot_kws={"size": 12})
            ax[1].set_xlabel('Number of buildings in path', fontsize=16, color='black')
            ax[1].set_yticks([])
            tick_positions = range(0, len(building_counts), tick_interval)
            ax[1].set_xticks(tick_positions)
            ax[1].set_xticklabels([str(pos) for pos in tick_positions],
                                  rotation=0, fontsize=12, color='black')
        return

    # Build height matrix
    height_matrix = np.array(df_with_buildings['Intersected_heightroofs_m'].tolist())

    # Calculate stats across buildings
    mean_heights, min_heights, max_heights = [], [], []
    for col in range(height_matrix.shape[1]):
        column_data = height_matrix[:, col]
        valid_data = column_data[~np.isnan(column_data)]
        if len(valid_data) > 0:
            mean_heights.append(np.mean(valid_data))
            min_heights.append(np.min(valid_data))
            max_heights.append(np.max(valid_data))
        else:
            break

    mean_heights, min_heights, max_heights = map(np.array, [mean_heights, min_heights, max_heights])

    # Plot mean heights with index starting at 0
    if len(mean_heights) > 0:
        x_values = np.arange(0, len(mean_heights))
        ax[0].plot(x_values, mean_heights,
                   label='Mean building height', color='black', linestyle='-')
        ax[0].fill_between(x_values, min_heights, max_heights,
                           color='skyblue', alpha=0.5, label='Min-Max building height')

    ax[0].set_title(title, loc='left', fontsize=20, color='black')
    ax[0].set_xlabel('Building index (0 = first)', fontsize=16, color='black')
    ax[0].set_ylabel('Roof Height (m)', fontsize=16, color='black')
    ax[0].legend(fontsize=14)
    if len(mean_heights) > 0:
        ax[0].set_xlim(0, len(mean_heights) - 1)
        ax[0].set_xticks(range(0, len(mean_heights), tick_interval))
    ax[0].tick_params(axis='x', labelsize=12, colors='black')
    ax[0].tick_params(axis='y', labelsize=12, colors='black')

    # Heatmap including zero-building paths
    if len(building_counts) > 0:
        sns.heatmap([building_counts], cmap="Blues",
                    ax=ax[1], cbar=True, xticklabels=True, yticklabels=False,
                    cbar_kws={'orientation': 'vertical'}, annot_kws={"size": 12})
        ax[1].set_xlabel('Number of buildings in path', fontsize=16, color='black')
        ax[1].set_yticks([])
        tick_positions = range(0, len(building_counts), tick_interval)
        ax[1].set_xticks(tick_positions)
        ax[1].set_xticklabels([str(pos) for pos in tick_positions],
                              rotation=0, fontsize=12, color='black')


def create_visualizations(df):
    """Create all visualizations."""
    # Update borough mapping in traditional NYC order
    borough_order = ['X', 'M', 'B', 'Q', 'R']  # Bronx, Manhattan, Brooklyn, Queens, Staten Island
    borough_names = ['Bronx', 'Manhattan', 'Brooklyn', 'Queens', 'Staten Island']
    num_boroughs = len(borough_order)
    
    # Create figure
    fig = plt.figure(figsize=(18, 6 * (1 + num_boroughs)))
    gs = gridspec.GridSpec(2 + num_boroughs, 2, 
                          height_ratios=[4, 2] + [2] * num_boroughs,
                          width_ratios=[3, 1])
    
    # Plot city-wide data
    ax_all_parks_mean = plt.subplot(gs[0, :])
    ax_all_parks_heatmap = plt.subplot(gs[1, :])
    plot_summary_roof_heights_and_heatmap(
        [ax_all_parks_mean, ax_all_parks_heatmap], df, 'New York City', tick_interval=10
    )
    
    # Plot borough-specific data
    for i, (borough_code, borough_name) in enumerate(zip(borough_order, borough_names)):
        ax_borough_mean = plt.subplot(gs[i + 2, 0])
        ax_borough_heatmap = plt.subplot(gs[i + 2, 1])
        borough_df = df[df['Park1_borough'] == borough_code].copy()
        
        if not borough_df.empty:
            plot_summary_roof_heights_and_heatmap(
                [ax_borough_mean, ax_borough_heatmap], borough_df, borough_name, tick_interval=8
            )
        else:
            print(f"No valid roof height data for borough: {borough_name}")
    
    plt.tight_layout()
    return fig

def calculate_all_statistics(df):
    """Calculate building height, distance, and building intersection statistics."""
    # Calculate building height statistics - excluding zeros and NaN values
    all_heights = []
    for heights in df['Intersected_heightroofs_m']:
        valid_heights = [h for h in heights if not np.isnan(h) and h > 0]  # Changed to exclude zeros
        all_heights.extend(valid_heights)
    
    if all_heights:
        height_stats = {
            'max_height': np.max(all_heights),
            'min_height': np.min(all_heights),  # This will now be the minimum non-zero height
            'mean_height': np.mean(all_heights),
            'se_height': sem(all_heights)
        }
    else:
        height_stats = {
            'max_height': np.nan,
            'min_height': np.nan,
            'mean_height': np.nan,
            'se_height': np.nan
        }
    
    # Calculate building intersection statistics
    building_counts = [len([h for h in heights if not np.isnan(h)]) 
                      for heights in df['Intersected_heightroofs_m']]
    if building_counts:
        intersection_stats = {
            'max_buildings': np.max(building_counts),
            'min_buildings': np.min(building_counts),
            'mean_buildings': np.mean(building_counts),
            'se_buildings': sem(building_counts) if len(building_counts) > 1 else np.nan
        }
    else:
        intersection_stats = {
            'max_buildings': 0,
            'min_buildings': 0,
            'mean_buildings': 0,
            'se_buildings': np.nan
        }
    
    # Calculate distance statistics using euclidean_distance column
    if 'euclidean_distance' in df.columns:
        distances = df['euclidean_distance'].dropna().values
    else:
        print("Warning: No euclidean_distance column found")
        distances = np.array([])
        
    if len(distances) > 0:
        distance_stats = {
            'mean_distance': np.mean(distances),
            'se_distance': sem(distances) if len(distances) > 1 else np.nan,
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'count': len(distances)
        }
    else:
        distance_stats = {
            'mean_distance': np.nan,
            'se_distance': np.nan,
            'min_distance': np.nan,
            'max_distance': np.nan,
            'count': 0
        }
    
    # Add count of paths with non-zero heights
    height_stats['paths_with_heights'] = len([heights for heights in df['Intersected_heightroofs_m'] 
                                            if any(h > 0 and not np.isnan(h) for h in heights)])
    
    # Combine all statistics
    return {**height_stats, **distance_stats, **intersection_stats}

def calculate_borough_statistics(df):
    """Calculate statistics for each borough."""
    borough_stats = {}
    for borough in df['Park1_borough'].unique():
        borough_df = df[df['Park1_borough'] == borough]
        if not borough_df.empty and isinstance(borough, str):  # Ensure borough is valid
            borough_stats[borough] = calculate_all_statistics(borough_df)
    return borough_stats

def save_summary_statistics(df, stats, output_path):
    """Save summary statistics to a text file."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        # City-wide statistics
        f.write("New York City Summary (Euclidean Distances <= 1000m)\n")
        f.write("=" * 53 + "\n")
        
        # Basic path statistics
        total_paths = len(df)
        zero_building_paths = df['Intersected_heightroofs_m'].apply(
            lambda x: len([h for h in x if not np.isnan(h)]) == 0
        ).sum()
        f.write(f"Total paths: {total_paths}\n")
        f.write(f"Paths with zero buildings: {zero_building_paths}\n")
        f.write(f"Paths with buildings: {total_paths - zero_building_paths}\n")
        f.write(f"Paths with non-zero height buildings: {stats['paths_with_heights']}\n\n")
        
        # Building intersection statistics
        f.write("Building Intersection Statistics\n")
        f.write("-----------------------------\n")
        f.write(f"Maximum buildings intersected: {stats['max_buildings']}\n")
        f.write(f"Minimum buildings intersected: {stats['min_buildings']}\n")
        f.write(f"Mean buildings intersected: {stats['mean_buildings']:.2f}\n")
        f.write(f"Standard error of intersections: {stats['se_buildings']:.2f}\n\n")
        
        # Building height statistics
        f.write("Building Height Statistics (excluding zero heights)\n")
        f.write("--------------------------------------------\n")
        if not np.isnan(stats['max_height']):
            f.write(f"Maximum building height: {stats['max_height']:.2f} m\n")
            f.write(f"Minimum building height: {stats['min_height']:.2f} m\n")
            f.write(f"Mean building height: {stats['mean_height']:.2f} m\n")
            f.write(f"Standard error of height: {stats['se_height']:.2f} m\n")
        else:
            f.write("No valid building height data available\n")
        f.write("\n")
        
        # Distance statistics
        f.write("Distance Statistics\n")
        f.write("-----------------\n")
        if not np.isnan(stats['mean_distance']):
            f.write(f"Mean distance between parks: {stats['mean_distance']:.2f} m\n")
            f.write(f"Standard error of distance: {stats['se_distance']:.2f} m\n")
            f.write(f"Minimum distance: {stats['min_distance']:.2f} m\n")
            f.write(f"Maximum distance: {stats['max_distance']:.2f} m\n")
        else:
            f.write("Distance data not available\n")
        f.write(f"Number of paths: {stats['count']}\n\n")

        # Borough-specific statistics
        borough_mapping = {
            'X': 'Bronx',
            'M': 'Manhattan',
            'B': 'Brooklyn',
            'Q': 'Queens',
            'R': 'Staten Island'
        }
        
        borough_stats = calculate_borough_statistics(df)
        
        for borough_code, borough_name in borough_mapping.items():
            if borough_code in borough_stats:
                stats = borough_stats[borough_code]
                f.write(f"{borough_name} Summary\n")
                f.write("="* len(f"{borough_name} Summary") + "\n")
                
                # Basic path statistics
                borough_df = df[df['Park1_borough'] == borough_code]
                total_borough_paths = len(borough_df)
                zero_building_borough_paths = borough_df['Intersected_heightroofs_m'].apply(
                    lambda x: len([h for h in x if not np.isnan(h)]) == 0
                ).sum()
                f.write(f"Total paths: {total_borough_paths}\n")
                f.write(f"Paths with zero buildings: {zero_building_borough_paths}\n")
                f.write(f"Paths with buildings: {total_borough_paths - zero_building_borough_paths}\n")
                f.write(f"Paths with non-zero height buildings: {stats['paths_with_heights']}\n\n")
                
                # Building intersection statistics
                f.write("Building Intersection Statistics:\n")
                f.write(f"Maximum buildings intersected: {stats['max_buildings']}\n")
                f.write(f"Minimum buildings intersected: {stats['min_buildings']}\n")
                f.write(f"Mean buildings intersected: {stats['mean_buildings']:.2f}\n")
                f.write(f"Standard error of intersections: {stats['se_buildings']:.2f}\n\n")
                
                # Building height statistics
                if not np.isnan(stats['max_height']):
                    f.write("Building Height Statistics (excluding zero heights):\n")
                    f.write(f"Maximum building height: {stats['max_height']:.2f} m\n")
                    f.write(f"Minimum building height: {stats['min_height']:.2f} m\n")
                    f.write(f"Mean building height: {stats['mean_height']:.2f} m\n")
                    f.write(f"Standard error of height: {stats['se_height']:.2f} m\n\n")
                else:
                    f.write("No valid building height data available\n\n")
                
                # Distance statistics
                f.write("Distance Statistics:\n")
                if not np.isnan(stats['mean_distance']):
                    f.write(f"Mean distance between parks: {stats['mean_distance']:.2f} m\n")
                    f.write(f"Standard error of distance: {stats['se_distance']:.2f} m\n")
                    f.write(f"Minimum distance: {stats['min_distance']:.2f} m\n")
                    f.write(f"Maximum distance: {stats['max_distance']:.2f} m\n")
                else:
                    f.write("Distance data not available\n")
                f.write(f"Number of paths: {stats['count']}\n\n")

def main(results_gpkg_path, intersections_gpkg_path, parks_gpkg_path, borough_gpkg_path, output_path, figure_path=None):
    """Main function to run the entire analysis."""
    # Check environment first
    if not check_environment():
        raise EnvironmentError("Required packages are missing")
    
    # Load and filter paths from results.gpkg
    print("Loading and filtering paths...")
    filtered_paths_gdf = load_and_filter_paths(results_gpkg_path, max_distance=1000)
    
    # Process intersection data for filtered paths only
    print("Processing intersection data...")
    paths_gdf = process_intersection_data(intersections_gpkg_path, filtered_paths_gdf)
    
    # Add borough information
    print("Adding borough information...")
    paths_gdf = add_borough_info(paths_gdf, parks_gpkg_path, borough_gpkg_path)
    
    # Clean and process data
    paths_gdf = clean_and_process_data(paths_gdf)
    
    # Create visualizations
    fig = create_visualizations(paths_gdf)
    
    # Calculate all statistics
    print("\nCalculating statistics...")
    stats = calculate_all_statistics(paths_gdf)
    
    # Print city-wide statistics
    print(f"\nCity-wide Stats (Euclidean distances <= 1000m):")
    print(f"Total paths: {len(paths_gdf)}")
    print(f"Building Intersections:")
    print(f"  Maximum buildings intersected: {stats['max_buildings']}")
    print(f"  Mean buildings intersected: {stats['mean_buildings']:.2f}")
    print(f"  Standard error of intersections: {stats['se_buildings']:.2f}")
    
    if not np.isnan(stats['max_height']):
        print(f"\nBuilding Heights:")
        print(f"  Maximum height: {stats['max_height']:.2f} m")
        print(f"  Mean height: {stats['mean_height']:.2f} m")
        print(f"  Standard error of height: {stats['se_height']:.2f} m")
    
    if not np.isnan(stats['mean_distance']):
        print(f"\nDistances:")
        print(f"  Mean distance between parks: {stats['mean_distance']:.2f} m")
        print(f"  Standard error of distance: {stats['se_distance']:.2f} m")
        print(f"  Number of paths: {stats['count']}")
    
    # Calculate and print borough statistics
    borough_stats = calculate_borough_statistics(paths_gdf)
    print("\nBorough-specific Stats:")
    borough_names = {'X': 'Bronx', 'M': 'Manhattan', 'B': 'Brooklyn', 
                    'Q': 'Queens', 'R': 'Staten Island'}
    
    for borough_code, b_stats in borough_stats.items():
        if borough_code in borough_names:
            print(f"\n{borough_names[borough_code]}:")
            print(f"Building Intersections:")
            print(f"  Maximum buildings: {b_stats['max_buildings']}")
            print(f"  Mean buildings: {b_stats['mean_buildings']:.2f}")
            
            if not np.isnan(b_stats['max_height']):
                print(f"Building Heights:")
                print(f"  Mean height: {b_stats['mean_height']:.2f} m")
                print(f"  Standard error: {b_stats['se_height']:.2f} m")
                
            if not np.isnan(b_stats['mean_distance']):
                print(f"Distances:")
                print(f"  Mean distance: {b_stats['mean_distance']:.2f} m")
                print(f"  Standard error: {b_stats['se_distance']:.2f} m")
                print(f"  Number of paths: {b_stats['count']}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save processed data and outputs
    paths_gdf.to_csv(output_path, index=False)
    print(f"\nData saved to: {output_path}")
    
    if figure_path:
        try:
            os.makedirs(os.path.dirname(figure_path), exist_ok=True)
            fig.savefig(figure_path, bbox_inches='tight', dpi=300)
            print(f"Figure saved to: {figure_path}")
        except Exception as e:
            print(f"Error saving figure: {str(e)}")
    
    # Save summary statistics
    summary_dir = os.path.dirname(output_path)
    summary_path = os.path.join(summary_dir, 'building_summary_statistics_1000m.txt')
    save_summary_statistics(paths_gdf, stats, summary_path)
    print(f"Summary statistics saved to: {summary_path}")
    
    return paths_gdf, fig, stats

if __name__ == "__main__":
    # File paths - updated to use the correct files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_gpkg_path = os.path.join(base_dir, "../outputs/dijkstra_paths.gpkg")  # Main results file
    intersections_gpkg_path = os.path.join(base_dir, "../outputs/euclidean_path_building_analysis.gpkg")
    parks_gpkg_path = os.path.join(base_dir, "../data/open_and_park_single_26918.gpkg")
    borough_gpkg_path = os.path.join(base_dir, "../data/borough_boundaries_26918.gpkg")
    output_path = os.path.join(base_dir, "../outputs/parks_with_boroughs_and_analysis_1000m.csv")
    figure_path = os.path.join(base_dir, "../figures/building_heights_analysis_1000m.png")
    
    # Run analysis
    df, fig, stats = main(results_gpkg_path, intersections_gpkg_path, parks_gpkg_path, borough_gpkg_path, output_path, figure_path)

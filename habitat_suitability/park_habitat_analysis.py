import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob

# Create a directory to save the graphics
output_dir = '../figures'
os.makedirs(output_dir, exist_ok=True)

# Define boroughs and cover types for the new output format
boroughs = ['staten_island', 'manhattan', 'bronx', 'brooklyn', 'queens']
cover_types = ['high_vegetation', 'medium_vegetation', 'low_vegetation', 'building', 'ground', 'water']

# Define the base pattern for filenames - updated for new output location
base_pattern = 'lidar/{borough}_{cover_type}_*.csv'

all_data = []

# Load and combine all boroughs and cover types
for borough in boroughs:
    for cover_type in cover_types:
        pattern = base_pattern.format(borough=borough, cover_type=cover_type)
        matching_files = glob.glob(pattern)

        if not matching_files:
            print(f"No files found for {borough} - {cover_type}")
            continue

        file_path = matching_files[0]
        print(f"Processing file: {file_path}")

        try:
            df = pd.read_csv(file_path)
            
            # Get the area column for this cover type (new format has category-specific columns)
            area_col = f'{cover_type}_total_area_m2'
            pixel_col = f'{cover_type}_pixel_count'
            
            if area_col in df.columns:
                # Create a standardized dataframe
                park_df = pd.DataFrame({
                    'park_id': df.get('park_id', df.get('unique_id', range(len(df)))),
                    'park_name': df.get('park_name', f'Park_{borough}'),
                    'borough': borough.replace('_', ' ').title(),
                    'cover_type': cover_type.replace('_', ' ').title(),
                    'area_m2': df[area_col],
                    'pixel_count': df[pixel_col] if pixel_col in df.columns else 0,
                    'geometry_area_m2': df.get('geometry_area_m2', 0)
                })
                
                # Only include parks with non-zero coverage for this type
                park_df = park_df[park_df['area_m2'] > 0]
                
                if not park_df.empty:
                    all_data.append(park_df)
                    print(f"  Added {len(park_df)} parks with {cover_type} coverage")
                else:
                    print(f"  No parks with {cover_type} coverage in {borough}")
            else:
                print(f"  Column {area_col} not found in {file_path}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if not all_data:
    print("No data found! Check your file paths and output directory.")
else:
    # Combine all dataframes
    print(f"\nCombining data from {len(all_data)} borough/cover type combinations...")
    citywide_df = pd.concat(all_data, ignore_index=True)
    
    # Add log transform (only for areas > 0)
    citywide_df = citywide_df[citywide_df['area_m2'] > 0]
    citywide_df['log_area_m2'] = np.log10(citywide_df['area_m2'])
    
    print(f"Combined dataset: {len(citywide_df)} records")
    print(f"Cover types: {citywide_df['cover_type'].unique()}")
    print(f"Boroughs: {citywide_df['borough'].unique()}")
    
    # Save combined data
    citywide_df.to_csv(f'{output_dir}/citywide_combined_park_lidar.csv', index=False)
    
    # Compute CDF for each cover type
    citywide_df['cdf'] = citywide_df.groupby(['cover_type', 'borough'])['area_m2'].rank(method='average', pct=True)
    
    # Create straight-edge stacked area plot with bin ranges
    log_bins = np.linspace(citywide_df['log_area_m2'].min(), citywide_df['log_area_m2'].max(), 20)
    bin_labels = [f'[{log_bins[i]:.1f}, {log_bins[i+1]:.1f}]' for i in range(len(log_bins) - 1)]
    
    # Bin the data using closed intervals
    citywide_df['log_bin'] = pd.cut(
        citywide_df['log_area_m2'],
        bins=log_bins,
        labels=bin_labels,
        include_lowest=True,
        right=True
    )
    
    # Compute proportions for each bin and cover type
    proportions = (
        citywide_df.groupby(['log_bin', 'cover_type'])['area_m2']
        .sum()
        .unstack(fill_value=0)
    )
    
    # Normalize by row to get proportions
    available_types = proportions.columns.tolist()
    desired_order = ['Ground', 'High Vegetation', 'Medium Vegetation', 'Low Vegetation', 'Building', 'Water']
    # Only use cover types that are actually in the data
    plot_order = [ct for ct in desired_order if ct in available_types]
    
    proportions_relative = proportions.div(proportions.sum(axis=1), axis=0)
    proportions_relative = proportions_relative[plot_order]
    
    # Plot the stacked area chart
    plt.figure(figsize=(16, 10))
    
    # Plot each cover type as a filled area for histogram-like bins
    y_bottom = np.zeros(len(proportions_relative))
    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_order)))
    
    for idx, cover_type in enumerate(plot_order):
        plt.bar(
            range(len(proportions_relative)),
            proportions_relative[cover_type],
            bottom=y_bottom,
            color=colors[idx],
            label=cover_type,
            width=1.0,
            align='center'
        )
        y_bottom += proportions_relative[cover_type]
    
    # Customize the plot
    plt.xticks(
        ticks=range(len(bin_labels)),
        labels=bin_labels,
        rotation=45,
        fontsize=10
    )
    plt.xlabel('Log₁₀(Coverage Area) Bin Ranges (m²)', fontsize=12)
    plt.ylabel('Proportion of Cover Type', fontsize=12)
    # plt.title('Proportion of LIDAR Cover Types by Managed Coverage Area', fontsize=14, weight='bold')
    plt.legend(title='Cover Type', fontsize=10, title_fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.savefig(f'{output_dir}/park_lidar_stacked_histogram.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create a borough comparison plot
    plt.figure(figsize=(14, 8))
    
    # Calculate total coverage by borough and cover type
    borough_totals = (
        citywide_df.groupby(['borough', 'cover_type'])['area_m2']
        .sum()
        .unstack(fill_value=0)
    )
    
    # Create stacked bar chart by borough
    borough_totals_relative = borough_totals.div(borough_totals.sum(axis=1), axis=0)
    borough_totals_relative[plot_order].plot(kind='bar', stacked=True, figsize=(12, 8), 
                                           color=colors[:len(plot_order)])
    
    plt.title('LIDAR Cover Type Distribution by Borough', fontsize=14, weight='bold')
    plt.xlabel('Borough', fontsize=12)
    plt.ylabel('Proportion of Total Coverage', fontsize=12)
    plt.legend(title='Cover Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(f'{output_dir}/borough_comparison_lidar.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Calculate summary statistics - treat each row as a unique shape
    total_shapes = len(citywide_df)  # Each row is a unique shape
    min_area = citywide_df['area_m2'].min()
    max_area = citywide_df['area_m2'].max()
    total_coverage_area = citywide_df['area_m2'].sum()
    
    # Print the summary statistics
    print(f"\nSUMMARY STATISTICS:")
    print(f"Total unique park shapes: {total_shapes}")
    print(f"Total coverage records: {len(citywide_df)}")
    print(f"Minimum coverage area (m²): {min_area:.2f}")
    print(f"Maximum coverage area (m²): {max_area:.2f}")
    print(f"Total coverage area (m²): {total_coverage_area:.2f}")
    
    # Coverage by type
    print(f"\nCOVERAGE BY TYPE:")
    coverage_by_type = citywide_df.groupby('cover_type')['area_m2'].sum().sort_values(ascending=False)
    for cover_type, area in coverage_by_type.items():
        print(f"  {cover_type}: {area:.2f} m² ({area/total_coverage_area*100:.1f}%)")
    
    # Save summary statistics
    summary_stats_path = f'{output_dir}/park_lidar_summary_statistics.txt'
    with open(summary_stats_path, 'w') as file:
        file.write("PARK LIDAR ANALYSIS SUMMARY\n")
        file.write("=" * 40 + "\n\n")
        file.write(f"Total unique park shapes analyzed: {total_shapes}\n")
        file.write(f"Total coverage records: {len(citywide_df)}\n")
        file.write(f"Minimum coverage area (m²): {min_area:.2f}\n")
        file.write(f"Maximum coverage area (m²): {max_area:.2f}\n")
        file.write(f"Total coverage area (m²): {total_coverage_area:.2f}\n\n")
        
        file.write("COVERAGE BY TYPE:\n")
        for cover_type, area in coverage_by_type.items():
            file.write(f"  {cover_type}: {area:.2f} m² ({area/total_coverage_area*100:.1f}%)\n")
        
        file.write("\nCOVERAGE BY BOROUGH:\n")
        borough_coverage = citywide_df.groupby('borough')['area_m2'].sum().sort_values(ascending=False)
        for borough, area in borough_coverage.items():
            file.write(f"  {borough}: {area:.2f} m² ({area/total_coverage_area*100:.1f}%)\n")
        
        file.write("\nSHAPES BY BOROUGH AND COVER TYPE:\n")
        shapes_by_borough_type = citywide_df.groupby(['borough', 'cover_type']).size().unstack(fill_value=0)
        file.write(shapes_by_borough_type.to_string())
        file.write("\n")
    
    print(f"\nSummary statistics saved to {summary_stats_path}")
    print(f"Visualizations saved to {output_dir}/")
    print("Files created:")
    print(f"  - citywide_combined_park_lidar.csv")
    print(f"  - park_lidar_stacked_histogram.png")
    print(f"  - borough_comparison_lidar.png")
    print(f"  - park_lidar_summary_statistics.txt")

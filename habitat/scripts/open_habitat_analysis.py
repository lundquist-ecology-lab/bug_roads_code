import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob

# Create a directory to save the graphics
output_dir = '../figures'
os.makedirs(output_dir, exist_ok=True)

# Define boroughs and cover types
boroughs = ['manhattan', 'bronx', 'brooklyn', 'queens', 'staten_island']
cover_types = ['high_vegetation', 'medium_vegetation', 'low_vegetation', 'building', 'ground']

# Define the base pattern for filenames
base_pattern = '../outputs/lidar_results_open/{borough}_{cover_type}_*.csv'

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
        
        df = pd.read_csv(file_path)
        df['borough'] = borough.capitalize()
        df['cover_type'] = cover_type.replace('_', ' ').capitalize()
        all_data.append(df)

# Combine all dataframes and add log transform
citywide_df = pd.concat(all_data)
citywide_df['log_area_m2'] = np.log10(citywide_df['area_m2'])
citywide_df.to_csv(f'{output_dir}/citywide_combined.csv', index=False)

# Compute CDF for each cover type
citywide_df['cdf'] = citywide_df.groupby(['cover_type', 'borough'])['area_m2'].rank(method='average', pct=True)

# Create straight-edge stacked area plot# Create straight-edge stacked area plot with bin ranges
log_bins = np.linspace(-1.5, 6.7, 20)  # Define log-area bins
bin_labels = [f'[{log_bins[i]:.2f}, {log_bins[i+1]:.2f}]' for i in range(len(log_bins) - 1)]

# Bin the data using closed intervals [ ]
citywide_df['log_bin'] = pd.cut(
    citywide_df['log_area_m2'], 
    bins=log_bins, 
    labels=bin_labels,
    include_lowest=True, 
    right=True  # Closed interval on the right ([])
)

# Compute proportions for each bin and cover type
proportions = (
    citywide_df.groupby(['log_bin', 'cover_type'])['area_m2']
    .sum()
    .unstack(fill_value=0)
)

# Normalize by row to get proportions
desired_order = ['Ground', 'High vegetation', 'Medium vegetation', 'Low vegetation', 'Building']
proportions_relative = proportions.div(proportions.sum(axis=1), axis=0)
proportions_relative = proportions_relative[desired_order]

# Plot the stacked area chart with histogram-style bins
plt.figure(figsize=(16, 10))

# Plot each cover type as a filled area for histogram-like bins
y_bottom = np.zeros(len(proportions_relative))
colors = plt.cm.viridis(np.linspace(0, 1, len(desired_order)))

for idx, cover_type in enumerate(desired_order):
    plt.bar(
        proportions_relative.index,
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
plt.xlabel('Log₁₀(Area) Bin Ranges (m²)', fontsize=12)
plt.ylabel('Proportion of Cover Type', fontsize=12)
# plt.title('Proportion of Cover Types by Park Size (Log Transformed)', fontsize=14, weight='bold')
plt.legend(title='Cover Type', fontsize=10, title_fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.savefig(f'{output_dir}/stacked_area_histogram_bins.png', bbox_inches='tight', dpi=300)
plt.close()

# Calculate total point_id, minimum and maximum area_m2
total_points = citywide_df['point_id'].nunique() if 'point_id' in citywide_df.columns else None
min_area = citywide_df['area_m2'].min()
max_area = citywide_df['area_m2'].max()

# Print the summary statistics
print(f"Total unique point IDs: {total_points}" if total_points is not None else "No 'point_id' column found.")
print(f"Minimum area (m²): {min_area}")
print(f"Maximum area (m²): {max_area}")

# Save these summary stats to a text file for record-keeping
summary_stats_path = f'{output_dir}/summary_statistics.txt'
with open(summary_stats_path, 'w') as file:
    if total_points is not None:
        file.write(f"Total unique point IDs: {total_points}\n")
    file.write(f"Minimum area (m²): {min_area}\n")
    file.write(f"Maximum area (m²): {max_area}\n")

print(f"Summary statistics saved to {summary_stats_path}")

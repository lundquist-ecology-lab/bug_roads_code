import pandas as pd
import glob
import os

# Create output directory
output_dir = '../outputs'
os.makedirs(output_dir, exist_ok=True)

# Define boroughs and all cover types
boroughs = ['manhattan', 'bronx', 'brooklyn', 'queens', 'staten_island']
veg_types = ['high_vegetation', 'medium_vegetation', 'low_vegetation']
all_cover_types = veg_types + ['building', 'ground']

# Base pattern for filenames
base_pattern = '../outputs/lidar_results_open/{borough}_{cover_type}_*.csv'

all_data = []

# Load all data
for borough in boroughs:
    for cover_type in all_cover_types:
        pattern = base_pattern.format(borough=borough, cover_type=cover_type)
        matching_files = glob.glob(pattern)
        
        if matching_files:
            file_path = matching_files[0]
            df = pd.read_csv(file_path)
            df['borough'] = borough.capitalize()
            df['cover_type'] = cover_type.replace('_', ' ').capitalize()
            all_data.append(df)

# Combine all data
full_df = pd.concat(all_data)

# Calculate statistics for each cover type
stats = full_df.groupby('cover_type').agg({
    'area_m2': ['count', 'mean', 'sum']
}).round(2)

stats.columns = ['Count', 'Mean Area (m²)', 'Total Area (m²)']
stats['Total Area (km²)'] = stats['Total Area (m²)'] / 1_000_000
stats = stats.drop('Total Area (m²)', axis=1)

# Create summary text
summary = f"""Land Coverage Statistics in New York City

Statistics by Cover Type:
{stats.to_string()}

Notes:
- Mean Area shows the average size of individual features
- Total Area is converted to square kilometers
- Count shows number of individual features
"""

# Save to text file
with open(f'{output_dir}/land_coverage_means.txt', 'w') as f:
    f.write(summary)

print("Summary has been saved to land_coverage_means.txt")
#%%
import pandas as pd
import numpy as np

# Load the parks data
parks_df = pd.read_csv('all_parks_combined.csv')

# Load the park centroids CSV file to get acres associated with each name311
centroids_df = pd.read_csv('nyc_parks_shapes.csv')
centroids_df = centroids_df[['name311', 'acres']]
centroids_df['area_m2'] = centroids_df['acres'] * 4046.86  # Convert acres to m²

# Merge the parks data with the centroids data to get area_m2
merged_df = pd.merge(parks_df, centroids_df, on='name311')

# Define the radii and calculate the corresponding areas
radii = [100, 250, 500, 1000]
areas = [np.pi * r**2 for r in radii]
percentages = [11.2, 16.7]

# Calculate the minimum required areas for the given percentages
required_areas = {area: (area * percentages[0] / 100, area * percentages[1] / 100) for area in areas}

# Vegetation types and total area to analyze
area_types = ['area_m2', 'low_vegetation_area_m2', 'medium_vegetation_area_m2', 'high_vegetation_area_m2']

# Total number of parks
total_parks = len(parks_df['name311'].unique())

# Initialize results dictionary
all_results = {}

# Determine how many parks are large enough to support the required areas for each type
for area_type in area_types:
    results = {}
    for area, (min_area, max_area) in required_areas.items():
        count_min = len(merged_df[merged_df[area_type] >= min_area])
        count_max = len(merged_df[merged_df[area_type] >= max_area])
        count_additional = count_min - count_max
        percent_min = (count_min / total_parks) * 100
        percent_max = (count_max / total_parks) * 100
        percent_additional = (count_additional / total_parks) * 100
        results[area] = {
            'Min': count_min, 'Max': count_max, 'Additional': count_additional,
            'Min %': percent_min, 'Max %': percent_max, 'Additional %': percent_additional
        }
    all_results[area_type] = results

# Print the results
for area_type, results in all_results.items():
    type_name = 'Total Area' if area_type == 'area_m2' else area_type.replace('_area_m2', '').replace('_', ' ').capitalize()
    print(f"\nNumber of parks that can support the given area requirements for {type_name}:")
    for area, counts in results.items():
        print(f"For a circle with radius {np.sqrt(area / np.pi):.0f} m (area = {area:.2f} m²):")
        print(f"  Parks supporting at least {percentages[0]}% ({required_areas[area][0]:.2f} m²): {counts['Min']} ({counts['Min %']:.2f}%)")
        print(f"  Parks supporting at least {percentages[1]}% ({required_areas[area][1]:.2f} m²): {counts['Max']} ({counts['Max %']:.2f}%)")
        print(f"  Additional parks supporting at least {percentages[0]}% but less than {percentages[1]}%: {counts['Additional']} ({counts['Additional %']:.2f}%)")

# %%

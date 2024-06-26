# %%
import pandas as pd
import numpy as np

# Load the parks data with vegetation covers
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

# Vegetation types to analyze
vegetation_types = ['low_vegetation_area_m2', 'medium_vegetation_area_m2', 'high_vegetation_area_m2']

# Function to calculate total vegetation cover for clusters
def calculate_cluster_coverage(cluster_file, radius, veg_type):
    cluster_df = pd.read_csv(cluster_file)
    cluster_df['name311_list'] = cluster_df['name311'].str.split(', ')
    
    # Calculate total vegetation cover for each cluster
    total_coverage = []
    for index, row in cluster_df.iterrows():
        total_cover = merged_df[merged_df['name311'].isin(row['name311_list'])][veg_type].sum()
        total_coverage.append(total_cover)
    
    cluster_df['total_coverage_m2'] = total_coverage
    
    # Calculate the minimum required areas for the given radius
    min_area, max_area = required_areas[np.pi * radius**2]
    
    # Determine how many clusters can support the required areas
    count_min = len(cluster_df[cluster_df['total_coverage_m2'] >= min_area])
    count_max = len(cluster_df[cluster_df['total_coverage_m2'] >= max_area])
    count_additional = count_min - count_max
    percent_min = (count_min / len(cluster_df)) * 100
    percent_max = (count_max / len(cluster_df)) * 100
    percent_additional = (count_additional / len(cluster_df)) * 100
    
    return {
        'Min': count_min, 'Max': count_max, 'Additional': count_additional,
        'Min %': percent_min, 'Max %': percent_max, 'Additional %': percent_additional,
        'Total Parks': len(cluster_df)
    }

# Files and corresponding radii
cluster_files = [
    ('nyc_clusters_at_100.csv', 100),
    ('nyc_clusters_at_250.csv', 250),
    ('nyc_clusters_at_500.csv', 500),
    ('nyc_clusters_at_1000.csv', 1000)
]

# Analyze each cluster file for each vegetation type
for veg_type in vegetation_types:
    type_name = veg_type.replace('_area_m2', '').replace('_', ' ').capitalize()
    print(f"\nAnalysis for {type_name}:")
    for cluster_file, radius in cluster_files:
        results = calculate_cluster_coverage(cluster_file, radius, veg_type)
        area = np.pi * radius**2
        print(f"\nResults for clusters with radius {radius} m:")
        print(f"For a circle with radius {radius} m (area = {area:.2f} m²):")
        print(f"  Clusters supporting at least {percentages[0]}% ({required_areas[area][0]:.2f} m²): {results['Min']} ({results['Min %']:.2f}%)")
        print(f"  Clusters supporting at least {percentages[1]}% ({required_areas[area][1]:.2f} m²): {results['Max']} ({results['Max %']:.2f}%)")
        print(f"  Additional clusters supporting at least {percentages[0]}% but less than {percentages[1]}%: {results['Additional']} ({results['Additional %']:.2f}%)")
        print(f"  Total parks: {results['Total Parks']}")


# %%

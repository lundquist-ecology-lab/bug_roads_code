#%%
import pandas as pd
import numpy as np
from ast import literal_eval
from scipy.stats import sem

# Function to safely evaluate string representations of lists
def safe_literal_eval(val):
    try:
        return literal_eval(val)
    except (ValueError, SyntaxError):
        return []

# Read the existing CSV file into a DataFrame
df = pd.read_csv('your_file_with_buildings.csv')

# Select only the columns of interest
columns_of_interest = [
    'Park1_ID', 'Park1_name311', 'Park1_borough',
    'Park2_ID', 'Park2_name311', 'Park2_borough',
    'Distance_ft', 'Distance_m', 'Count_Intersected_Buildings',
    'Average_heightroof_m', 'Intersected_heightroofs_m'
]
df = df[columns_of_interest]

# Convert Park1_ID and Park2_ID to strings to handle mixed data types
df['Park1_ID'] = df['Park1_ID'].astype(str)
df['Park2_ID'] = df['Park2_ID'].astype(str)

# Remove paths where Park1_ID is equal to Park2_ID
df = df[df['Park1_ID'] != df['Park2_ID']]

# Remove duplicate paths (i.e., where the same pair appears in reverse order)
df['unique_pair'] = df.apply(lambda row: tuple(sorted([row['Park1_ID'], row['Park2_ID']])), axis=1)
df = df.drop_duplicates(subset='unique_pair')

# Ensure the column 'Intersected_heightroofs_m' is converted to lists
df['Intersected_heightroofs_m'] = df['Intersected_heightroofs_m'].apply(lambda x: safe_literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else []))

# Filter paths where all intersecting buildings are 1000 m or below
df['all_heights_below_1000m'] = df['Intersected_heightroofs_m'].apply(lambda lst: all(h <= 1000 for h in lst if not np.isnan(h)))

# Filter the DataFrame to only include paths where all buildings are 1000 m or below
df_below_1000m = df[df['all_heights_below_1000m']]

# Calculate total number of such paths
total_paths_below_1000m = len(df_below_1000m)

# Calculate number of such paths for each borough
paths_below_1000m_by_borough = df_below_1000m.groupby('Park1_borough')['Park1_borough'].count() + df_below_1000m.groupby('Park2_borough')['Park2_borough'].count()

# Calculate the total number of paths for each borough
total_paths_by_borough = df.groupby('Park1_borough')['Park1_borough'].count() + df.groupby('Park2_borough')['Park2_borough'].count()

# Calculate the ratio of paths below 1000 m to all paths for each borough
ratio_below_1000m_by_borough = paths_below_1000m_by_borough / total_paths_by_borough

# Calculate average and standard error of distance_m for those paths
average_distance_m_below_1000m = df_below_1000m['Distance_m'].mean()
se_distance_m_below_1000m = df_below_1000m['Distance_m'].sem()

# Filter out empty lists and lists with all NaN values before calculating the average height
filtered_heights = df_below_1000m['Intersected_heightroofs_m'].apply(lambda lst: [h for h in lst if not np.isnan(h)])
filtered_heights = filtered_heights[filtered_heights.apply(len) > 0]

# Calculate the average height of the buildings in the 1000m or less paths
average_height_below_1000m = filtered_heights.apply(np.mean).mean()

# Calculate the total number of paths
total_paths = len(df)

# Identify paths that cross multiple boroughs
df_below_1000m['crosses_multiple_boroughs'] = df_below_1000m['Park1_borough'] != df_below_1000m['Park2_borough']

# Filter paths that cross multiple boroughs
paths_crossing_multiple_boroughs = df_below_1000m[df_below_1000m['crosses_multiple_boroughs']]

# Calculate total number of paths that cross multiple boroughs
total_paths_crossing_multiple_boroughs = len(paths_crossing_multiple_boroughs)

# Calculate total and average distance of the multi-borough paths
total_distance_crossing_multiple_boroughs = paths_crossing_multiple_boroughs['Distance_m'].sum()
average_distance_crossing_multiple_boroughs = paths_crossing_multiple_boroughs['Distance_m'].mean()

# Calculate which boroughs are connected by these multi-borough paths
# Combine borough combinations such as BQ and QB
paths_crossing_multiple_boroughs['borough_combo'] = paths_crossing_multiple_boroughs.apply(
    lambda row: '-'.join(sorted([row['Park1_borough'], row['Park2_borough']])), axis=1)

borough_connections = paths_crossing_multiple_boroughs.groupby('borough_combo').size()

# Calculate mean and SE of the path distances by borough
mean_distance_by_borough = df_below_1000m.groupby('Park1_borough')['Distance_m'].mean()
mean_distance_by_borough.update(df_below_1000m.groupby('Park2_borough')['Distance_m'].mean())
se_distance_by_borough = df_below_1000m.groupby('Park1_borough')['Distance_m'].apply(sem)
se_distance_by_borough.update(df_below_1000m.groupby('Park2_borough')['Distance_m'].apply(sem))

# Calculate mean and SE for mixed borough paths
mean_distance_mixed_boroughs = paths_crossing_multiple_boroughs['Distance_m'].mean()
se_distance_mixed_boroughs = sem(paths_crossing_multiple_boroughs['Distance_m'])

# Calculate mean and SE for each borough combination
mean_distance_by_combo = paths_crossing_multiple_boroughs.groupby('borough_combo')['Distance_m'].mean()
se_distance_by_combo = paths_crossing_multiple_boroughs.groupby('borough_combo')['Distance_m'].apply(sem)

# Calculate the number of unique parks in the data
unique_parks = pd.unique(df[['Park1_name311', 'Park2_name311']].values.ravel('K'))
total_unique_parks = len(unique_parks)

# Output the results
print(f"Total number of paths where all buildings are 1000 m or below: {total_paths_below_1000m}")
print(f"Number of such paths by borough:\n{paths_below_1000m_by_borough}")
print(f"Total number of paths by borough:\n{total_paths_by_borough}")
print(f"Ratio of paths below 1000 m to all paths by borough:\n{ratio_below_1000m_by_borough}")
print(f"Average distance of those paths: {average_distance_m_below_1000m:.2f} meters")
print(f"Standard error of the distance: {se_distance_m_below_1000m:.2f} meters")
print(f"Average height of buildings in the 1000m or less paths: {average_height_below_1000m:.2f} meters")
print(f"Total number of paths: {total_paths}")
print(f"Total number of paths crossing multiple boroughs: {total_paths_crossing_multiple_boroughs}")
print(f"Total distance of paths crossing multiple boroughs: {total_distance_crossing_multiple_boroughs:.2f} meters")
print(f"Average distance of paths crossing multiple boroughs: {average_distance_crossing_multiple_boroughs:.2f} meters")
print(f"Borough connections for paths crossing multiple boroughs:\n{borough_connections}")
print(f"Mean distance by borough:\n{mean_distance_by_borough}")
print(f"Standard error of distance by borough:\n{se_distance_by_borough}")
print(f"Mean distance for mixed borough paths: {mean_distance_mixed_boroughs:.2f} meters")
print(f"Standard error of distance for mixed borough paths: {se_distance_mixed_boroughs:.2f} meters")
print(f"Mean distance by borough combination:\n{mean_distance_by_combo}")
print(f"Standard error of distance by borough combination:\n{se_distance_by_combo}")
print(f"Total number of unique parks: {total_unique_parks}")

# Save results to a CSV file
results = {
    'Total Paths Below 1000m': total_paths_below_1000m,
    'Average Distance (m)': average_distance_m_below_1000m,
    'SE Distance (m)': se_distance_m_below_1000m,
    'Average Height Below 1000m (m)': average_height_below_1000m,
    'Total Paths': total_paths,
    'Paths by Borough Below 1000m': paths_below_1000m_by_borough.to_dict(),
    'Total Paths by Borough': total_paths_by_borough.to_dict(),
    'Ratio of Paths Below 1000m to All Paths by Borough': ratio_below_1000m_by_borough.to_dict(),
    'Total Paths Crossing Multiple Boroughs': total_paths_crossing_multiple_boroughs,
    'Total Distance Crossing Multiple Boroughs (m)': total_distance_crossing_multiple_boroughs,
    'Average Distance Crossing Multiple Boroughs (m)': average_distance_crossing_multiple_boroughs,
    'Borough Connections for Paths Crossing Multiple Boroughs': borough_connections.to_dict(),
    'Mean Distance by Borough': mean_distance_by_borough.to_dict(),
    'SE Distance by Borough': se_distance_by_borough.to_dict(),
    'Mean Distance for Mixed Borough Paths (m)': mean_distance_mixed_boroughs,
    'SE Distance for Mixed Borough Paths (m)': se_distance_mixed_boroughs,
    'Mean Distance by Borough Combination': mean_distance_by_combo.to_dict(),
    'SE Distance by Borough Combination': se_distance_by_combo.to_dict(),
    'Total Unique Parks': total_unique_parks
}

results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Value'])
results_df.to_csv('paths_below_1000m_statistics.csv')

print("Results saved to 'paths_below_1000m_statistics.csv'")

# %%

#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

# Bug road distances
df = pd.read_csv("/csv/nyc_paths_filtered.csv")
df = df[df['distance_meters'] > 1]

print(f'Number of paths = {len(df["distance_meters"])}')

# Concatenate the two columns into a single series
combined_series = pd.concat([df['start_name'], df['end_name']])

# Get the number of unique values in the combined series
unique_combined_count = combined_series.nunique()

print(f'Number of unique names between start_name and end_name = {unique_combined_count}')

distance_mean = df['distance_meters'].mean()
distance_stderr = df['distance_meters'].sem()

print(f"Distance mean: {distance_mean} m, standard error: {distance_stderr} m")


min_distance = df['distance_meters'].min()
max_distance = df['distance_meters'].max()

print(f"Minimum Distance: {min_distance} m", f"Maxium Distance: {max_distance} m")

# Load the data
df2 = pd.read_csv("/gis/nyc_parks_centroids.csv")

# Convert 'name311' column to categorical type
df2['name311'] = df2['name311'].astype('category')

# Print the count of unique 'name311' values
unique_count = df2['name311'].nunique()
print(f"The count of unique 'name311' values is: {unique_count}")


# Create a new column to check if start_borough and end_borough are the same
df['same_borough'] = df['start_borough'] == df['end_borough']

# Count where start_borough and end_borough are the same
same_count = df['same_borough'].value_counts()[True]

# Count where start_borough and end_borough are different
different_count = df['same_borough'].value_counts()[False]

print(f"Count where start and end borough are the same: {same_count}")
print(f"Count where start and end borough are different: {different_count}")

# Group by start_borough and end_borough and count the occurrences where they are different
borough_difference_counts = df[df['same_borough'] == False].groupby(['start_borough', 'end_borough']).size().reset_index(name='counts')

# Group by start_borough to count occurrences where start_borough and end_borough are the same
same_borough_counts = df[df['same_borough'] == True].groupby('start_borough').size().reset_index(name='counts')

# Calculate total paths, mean distance, and standard error of distance_meters for each combination of start_borough and end_borough
grouped = df.groupby(['start_borough', 'end_borough'])
path_counts = grouped.size().reset_index(name='total_paths')
mean_distances = grouped['distance_meters'].mean().reset_index(name='mean_distance')
se_distances = grouped['distance_meters'].apply(lambda x: np.std(x, ddof=1) / np.sqrt(len(x))).reset_index(name='se_distance')

# Merge all DataFrames into one
combined_stats = pd.merge(path_counts, mean_distances, on=['start_borough', 'end_borough'])
combined_stats = pd.merge(combined_stats, se_distances, on=['start_borough', 'end_borough'])

# Print the DataFrames
print("Differences between start and end boroughs with counts:")
print(borough_difference_counts)

print("\nCounts of same start and end borough by borough:")
print(same_borough_counts)

print("\nTotal paths, mean distance (meters), and standard error for each start and end borough combination:")
print(combined_stats)

# Function to parse the coordinates
def parse_coords(coord_str):
    x, y = map(float, coord_str.strip('"').split(','))
    return x, y

# Apply the function to get coordinates
df['start_coords'] = df['start_park'].apply(parse_coords)
df['end_coords'] = df['end_park'].apply(parse_coords)

# Function to calculate Euclidean distance
def euclidean_distance(coords1, coords2):
    return np.sqrt((coords1[0] - coords2[0])**2 + (coords1[1] - coords2[1])**2)

# Calculate the Euclidean distance in feet
df['euclidean_distance_feet'] = df.apply(lambda row: euclidean_distance(row['start_coords'], row['end_coords']), axis=1)

# Convert distance from feet to meters
df['euclidean_distance_meters'] = df['euclidean_distance_feet'] * 0.3048

# Calculate the percentage difference
df['percentage_difference'] = ((df['distance_meters'] - df['euclidean_distance_meters']) / df['distance_meters']) * 100

print(df['percentage_difference'].head(10))

# Summary statistics for all data
mean_distance = df['euclidean_distance_meters'].mean()
se_distance = df['euclidean_distance_meters'].std(ddof=1) / np.sqrt(len(df))
mean_percentage_diff = df['percentage_difference'].mean()
se_percentage_diff = df['percentage_difference'].std(ddof=1) / np.sqrt(len(df))

# Print overall summary statistics
print("Overall Summary Statistics:")
print(f"Mean Euclidean Distance (meters): {mean_distance}")
print(f"SE Euclidean Distance (meters): {se_distance}")
print(f"Mean Percentage Difference: {mean_percentage_diff}")
print(f"SE Percentage Difference: {se_percentage_diff}")

# Summary statistics by borough combination
grouped = df.groupby(['start_borough', 'end_borough'])

mean_distances_by_borough = grouped['euclidean_distance_meters'].mean().reset_index(name='mean_distance')
se_distances_by_borough = grouped['euclidean_distance_meters'].apply(lambda x: x.std(ddof=1) / np.sqrt(len(x))).reset_index(name='se_distance')

mean_percentage_diff_by_borough = grouped['percentage_difference'].mean().reset_index(name='mean_percentage_diff')
se_percentage_diff_by_borough = grouped['percentage_difference'].apply(lambda x: x.std(ddof=1) / np.sqrt(len(x))).reset_index(name='se_percentage_diff')

# Merge all summary statistics into one DataFrame
borough_stats = pd.merge(mean_distances_by_borough, se_distances_by_borough, on=['start_borough', 'end_borough'])
borough_stats = pd.merge(borough_stats, mean_percentage_diff_by_borough, on=['start_borough', 'end_borough'])
borough_stats = pd.merge(borough_stats, se_percentage_diff_by_borough, on=['start_borough', 'end_borough'])

# Print summary statistics by borough combination
print("\nSummary Statistics by Borough Combination:")
print(borough_stats)

# %%

#%%
import pandas as pd
import numpy as np
from ast import literal_eval

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

# Filter paths where all intersecting buildings are 10 m or below
df['all_heights_below_10m'] = df['Intersected_heightroofs_m'].apply(lambda lst: all(h <= 10 for h in lst if not np.isnan(h)))

# Filter the DataFrame to only include paths where all buildings are 10 m or below
df_below_10m = df[df['all_heights_below_10m']]

# Calculate total number of such paths
total_paths_below_10m = len(df_below_10m)

# Calculate number of such paths for each borough
paths_below_10m_by_borough = df_below_10m['Park1_borough'].value_counts()

# Calculate the total number of paths for each borough
total_paths_by_borough = df['Park1_borough'].value_counts()

# Calculate the ratio of paths below 10 m to all paths for each borough
ratio_below_10m_by_borough = paths_below_10m_by_borough / total_paths_by_borough

# Calculate average and standard error of distance_m for those paths
average_distance_m_below_10m = df_below_10m['Distance_m'].mean()
se_distance_m_below_10m = df_below_10m['Distance_m'].sem()

# Filter out empty lists and lists with all NaN values before calculating the average height
filtered_heights = df_below_10m['Intersected_heightroofs_m'].apply(lambda lst: [h for h in lst if not np.isnan(h)])
filtered_heights = filtered_heights[filtered_heights.apply(len) > 0]

# Calculate the average height of the buildings in the 10m or less paths
average_height_below_10m = filtered_heights.apply(np.mean).mean()

# Calculate the total number of paths
total_paths = len(df)

# Output the results
print(f"Total number of paths where all buildings are 10 m or below: {total_paths_below_10m}")
print(f"Number of such paths by borough:\n{paths_below_10m_by_borough}")
print(f"Total number of paths by borough:\n{total_paths_by_borough}")
print(f"Ratio of paths below 10 m to all paths by borough:\n{ratio_below_10m_by_borough}")
print(f"Average distance of those paths: {average_distance_m_below_10m:.2f} meters")
print(f"Standard error of the distance: {se_distance_m_below_10m:.2f} meters")
print(f"Average height of buildings in the 10m or less paths: {average_height_below_10m:.2f} meters")
print(f"Total number of paths: {total_paths}")

# Save results to a CSV file
results = {
    'Total Paths Below 10m': total_paths_below_10m,
    'Average Distance (m)': average_distance_m_below_10m,
    'SE Distance (m)': se_distance_m_below_10m,
    'Average Height Below 10m (m)': average_height_below_10m,
    'Total Paths': total_paths,
    'Paths by Borough Below 10m': paths_below_10m_by_borough.to_dict(),
    'Total Paths by Borough': total_paths_by_borough.to_dict(),
    'Ratio of Paths Below 10m to All Paths by Borough': ratio_below_10m_by_borough.to_dict()
}

results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Value'])
results_df.to_csv('paths_below_10m_statistics.csv')

print("Results saved to 'paths_below_10m_statistics.csv'")

# %%

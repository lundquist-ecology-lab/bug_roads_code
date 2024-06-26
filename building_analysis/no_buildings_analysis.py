#%%
import pandas as pd
import numpy as np

# Load the CSV file into a DataFrame, skipping lines with errors
df = pd.read_csv('your_file_no_building_paths.csv', on_bad_lines='skip')

# Filter out paths where Park1_name311 is the same as Park2_name311
df_filtered = df[df['Park1_name311'] != df['Park2_name311']]

# Total number of paths
total_paths = len(df_filtered)

# Number of unique parks in the dataset
total_parks = len(pd.unique(df_filtered[['Park1_ID', 'Park2_ID']].values.ravel('K')))

# Number of paths per borough
paths_per_borough = df_filtered['Park1_borough'].value_counts().reset_index()
paths_per_borough.columns = ['Borough', 'Number_of_Paths']

# Number of unique parks per borough
parks_per_borough = df_filtered.groupby('Park1_borough').apply(lambda x: len(pd.unique(x[['Park1_ID', 'Park2_ID']].values.ravel('K')))).reset_index()
parks_per_borough.columns = ['Borough', 'Number_of_Parks']

# Separate paths within the same borough and those in different boroughs
within_borough = df_filtered[df_filtered['Park1_borough'] == df_filtered['Park2_borough']]
different_boroughs = df_filtered[df_filtered['Park1_borough'] != df_filtered['Park2_borough']]

# Average and standard error of distance in meters for all paths
avg_distance_m_all = df_filtered['Distance_m'].mean()
se_distance_m_all = df_filtered['Distance_m'].sem()

# Average and standard error of distance in meters for within borough paths
avg_distance_m_within_borough = within_borough.groupby('Park1_borough')['Distance_m'].mean().reset_index()
se_distance_m_within_borough = within_borough.groupby('Park1_borough')['Distance_m'].sem().reset_index()
avg_distance_m_within_borough.columns = ['Borough', 'Average_Distance_m']
se_distance_m_within_borough.columns = ['Borough', 'SE_Distance_m']

# Average and standard error of distance in meters for different borough paths
# Group by unique combinations of Park1_borough and Park2_borough
different_boroughs['Borough_Combination'] = different_boroughs.apply(
    lambda x: tuple(sorted([x['Park1_borough'], x['Park2_borough']])), axis=1)

avg_distance_m_diff_borough = different_boroughs.groupby('Borough_Combination')['Distance_m'].mean().reset_index()
se_distance_m_diff_borough = different_boroughs.groupby('Borough_Combination')['Distance_m'].sem().reset_index()
avg_distance_m_diff_borough.columns = ['Borough_Combination', 'Average_Distance_m']
se_distance_m_diff_borough.columns = ['Borough_Combination', 'SE_Distance_m']

# Number of unique parks in different boroughs paths
parks_diff_borough = len(pd.unique(different_boroughs[['Park1_ID', 'Park2_ID']].values.ravel('K')))

# Prepare the summary statistics DataFrame
summary_stats = pd.DataFrame({
    'Statistic': ['Total number of paths', 'Average distance (m) for all paths', 'SE of distance (m) for all paths', 'Total number of parks'],
    'Value': [total_paths, avg_distance_m_all, se_distance_m_all, total_parks]
})

# Add total number of parks per borough
paths_per_borough['Number_of_Parks'] = parks_per_borough['Number_of_Parks']

# Prepare DataFrame for different boroughs parks count
parks_diff_borough_df = pd.DataFrame({
    'Statistic': ['Total number of parks in different boroughs paths'],
    'Value': [parks_diff_borough]
})

# Combine all results into one DataFrame
combined_stats = pd.concat([summary_stats,
                            paths_per_borough,
                            avg_distance_m_within_borough,
                            se_distance_m_within_borough,
                            avg_distance_m_diff_borough,
                            se_distance_m_diff_borough,
                            parks_diff_borough_df], axis=0, ignore_index=True)

# Save the combined results to a single CSV file
combined_stats.to_csv('combined_summary_stats.csv', index=False)

print("Summary statistics saved to combined_summary_stats.csv.")




# %%
import pandas as pd
import numpy as np

# Load the CSV files into DataFrames, skipping lines with errors
df1 = pd.read_csv('your_file_no_buildings.csv', on_bad_lines='skip')
df2 = pd.read_csv('your_file_with_buildings.csv', on_bad_lines='skip')

# Filter out paths where Park1_name311 is the same as Park2_name311 in both dataframes
df1_filtered = df1[df1['Park1_name311'] != df1['Park2_name311']]
df2_filtered = df2[df2['Park1_name311'] != df2['Park2_name311']]

# Combine the park IDs from both DataFrames
park_ids_df1 = pd.concat([df1_filtered['Park1_name311'], df1_filtered['Park2_name311']])
park_ids_df2 = pd.concat([df2_filtered['Park1_name311'], df2_filtered['Park2_name311']])
combined_park_ids = pd.concat([park_ids_df1, park_ids_df2])

# Calculate the number of unique parks in the combined DataFrame
unique_parks_combined = combined_park_ids.nunique()

# Display the result
print(f"Total number of unique parks in the combined files: {unique_parks_combined}")

# Save the result to a CSV file
unique_parks_df = pd.DataFrame({
    'Statistic': ['Total number of unique parks in combined files'],
    'Value': [unique_parks_combined]
})
unique_parks_df.to_csv('unique_parks_combined.csv', index=False)

print("Number of unique parks in combined files saved to unique_parks_combined.csv.")


# %%

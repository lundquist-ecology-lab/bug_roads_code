#%%
# Import necessary libraries
import os
import pandas as pd
from scipy.spatial import distance

# Directory containing the CSV files
csv_directory = '/csv/nyc_paths_1000m_or_less'

# List all CSV files in the directory
csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

# Initialize an empty list to hold dataframes
dataframes = []

# Read each CSV file and append the dataframe to the list
for csv_file in csv_files:
    file_path = os.path.join(csv_directory, csv_file)
    df = pd.read_csv(file_path)
    dataframes.append(df)

# Concatenate all dataframes into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Load the CSV file with park centroids
nyc_parks_centroids = pd.read_csv('nyc_parks_centroids.csv')

# Extract the specific columns and combine X and Y into one column
extracted_df = nyc_parks_centroids[['X', 'Y', 'fid']]
extracted_df['X_Y'] = extracted_df['X'].astype(str) + ', ' + extracted_df['Y'].astype(str)
extracted_columns_combined = extracted_df[['X_Y', 'fid']]
extracted_columns_combined[['x', 'y']] = extracted_columns_combined['X_Y'].str.split(', ', expand=True).astype(float)

# Parse the coordinates from the combined_csv dataframe
combined_df[['start_x', 'start_y']] = combined_df['start_park'].str.split(', ', expand=True).astype(float)
combined_df[['end_x', 'end_y']] = combined_df['end_park'].str.split(', ', expand=True).astype(float)

# Function to find the closest fid
def find_closest_fid(park_coords, fid_coords):
    distances = distance.cdist([park_coords], fid_coords, 'euclidean')
    closest_index = distances.argmin()
    return extracted_columns_combined.iloc[closest_index]['fid']

# Apply the function to find the closest fid for start_park and end_park
combined_df['start_fid'] = combined_df.apply(lambda row: find_closest_fid((row['start_x'], row['start_y']), extracted_columns_combined[['x', 'y']].values), axis=1)
combined_df['end_fid'] = combined_df.apply(lambda row: find_closest_fid((row['end_x'], row['end_y']), extracted_columns_combined[['x', 'y']].values), axis=1)

# Drop the intermediate coordinate columns
combined_df.drop(columns=['start_x', 'start_y', 'end_x', 'end_y'], inplace=True)

# Merge to get start park details
merged_start = pd.merge(combined_df, nyc_parks_centroids, how='left', left_on='start_fid', right_on='fid')
merged_start = merged_start.rename(columns={'name311': 'start_name', 'acres': 'start_acres', 'borough': 'start_borough'})
merged_start = merged_start.drop(columns=['fid'])

# Merge to get end park details
merged_end = pd.merge(merged_start, nyc_parks_centroids, how='left', left_on='end_fid', right_on='fid')
merged_end = merged_end.rename(columns={'name311': 'end_name', 'acres': 'end_acres', 'borough': 'end_borough'})
merged_end = merged_end.drop(columns=['fid'])

# Select and rename columns as needed
final_columns = ['start_park', 'end_park', 'distance', 'start_fid', 'start_name', 'start_acres', 'start_borough', 'end_fid', 'end_name', 'end_acres', 'end_borough']
final_df = merged_end[final_columns]

# Remove rows where distance is greater than 3300
final_df = final_df[final_df['distance'] <= 3300]

# Add a column to convert distance from feet to meters (1 foot = 0.3048 meters)
final_df['distance_meters'] = final_df['distance'] * 0.3048

# Filter out rows where start_fid equals end_fid
final_df = final_df[final_df['start_fid'] != final_df['end_fid']]

# Remove duplicate rows based on start_fid and end_fid combination
final_df = final_df.drop_duplicates(subset=['start_fid', 'end_fid'])

# Remove rows that have same names
final_df = final_df[final_df['start_name'] != final_df['end_name']]

# Save the final filtered DataFrame to a new CSV file
output_file_path = '/csv/nyc_paths_filtered.csv'
final_df.to_csv(output_file_path, index=False)

print(f"Final filtered data saved to {output_file_path}")

# %%

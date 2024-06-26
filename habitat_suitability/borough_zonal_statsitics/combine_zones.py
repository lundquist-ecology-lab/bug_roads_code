#%%
import pandas as pd

# List of boroughs
boroughs = ['bronx', 'brooklyn', 'manhattan', 'queens', 'si']

combined_df = pd.DataFrame()

for borough in boroughs:
    # Construct file names based on the borough
    buildings_file = f'../csv/{borough}_buildings.csv'
    ground_file = f'../csv/{borough}_ground.csv'
    high_veg_file = f'../csv/{borough}_high_vegetation.csv'
    low_veg_file = f'../csv/{borough}_low_vegetation.csv'
    medium_veg_file = f'../csv/{borough}_medium_vegetation.csv'

    # Read the CSV files
    buildings_df = pd.read_csv(buildings_file)
    ground_df = pd.read_csv(ground_file)
    high_veg_df = pd.read_csv(high_veg_file)
    low_veg_df = pd.read_csv(low_veg_file)
    medium_veg_df = pd.read_csv(medium_veg_file)

    # Select necessary columns from each dataframe
    buildings_df = buildings_df[['fid', 'boro_name', '_count']]
    ground_df = ground_df[['fid', 'boro_name', '_count']]
    high_veg_df = high_veg_df[['fid', 'boro_name', '_count']]
    low_veg_df = low_veg_df[['fid', 'boro_name', '_count']]
    medium_veg_df = medium_veg_df[['fid', 'boro_name', '_count']]

    # Rename the count columns before merging to avoid conflicts
    buildings_df.rename(columns={'_count': 'buildings_count'}, inplace=True)
    ground_df.rename(columns={'_count': 'ground_count'}, inplace=True)
    high_veg_df.rename(columns={'_count': 'high_vegetation_count'}, inplace=True)
    low_veg_df.rename(columns={'_count': 'low_vegetation_count'}, inplace=True)
    medium_veg_df.rename(columns={'_count': 'medium_vegetation_count'}, inplace=True)

    # Merge the dataframes on 'fid' and 'boro_name'
    merged_df = buildings_df.merge(ground_df, on=['fid', 'boro_name'])
    merged_df = merged_df.merge(high_veg_df, on=['fid', 'boro_name'])
    merged_df = merged_df.merge(low_veg_df, on=['fid', 'boro_name'])
    merged_df = merged_df.merge(medium_veg_df, on=['fid', 'boro_name'])

    # Calculate the area in m² for each count column
    merged_df['buildings_area_m2'] = merged_df['buildings_count'] * 0.0833
    merged_df['ground_area_m2'] = merged_df['ground_count'] * 0.0833
    merged_df['high_vegetation_area_m2'] = merged_df['high_vegetation_count'] * 0.0833
    merged_df['low_vegetation_area_m2'] = merged_df['low_vegetation_count'] * 0.0833
    merged_df['medium_vegetation_area_m2'] = merged_df['medium_vegetation_count'] * 0.0833

    # Calculate the area in km² for each count column
    merged_df['buildings_area_km2'] = merged_df['buildings_area_m2'] / 1e6
    merged_df['ground_area_km2'] = merged_df['ground_area_m2'] / 1e6
    merged_df['high_vegetation_area_km2'] = merged_df['high_vegetation_area_m2'] / 1e6
    merged_df['low_vegetation_area_km2'] = merged_df['low_vegetation_area_m2'] / 1e6
    merged_df['medium_vegetation_area_km2'] = merged_df['medium_vegetation_area_m2'] / 1e6

    # Append the merged dataframe to the combined dataframe
    combined_df = pd.concat([combined_df, merged_df], ignore_index=True)

# Select only the necessary columns for the final combined dataframe
final_combined_df = combined_df[['fid', 'boro_name', 
                                 'buildings_count', 'ground_count', 'high_vegetation_count', 'low_vegetation_count', 'medium_vegetation_count',
                                 'buildings_area_m2', 'ground_area_m2', 'high_vegetation_area_m2', 'low_vegetation_area_m2', 'medium_vegetation_area_m2',
                                 'buildings_area_km2', 'ground_area_km2', 'high_vegetation_area_km2', 'low_vegetation_area_km2', 'medium_vegetation_area_km2']]

# Save the final combined dataframe to a new CSV file
final_combined_df.to_csv('../statistics/combined_boroughs.csv', index=False)
print('Combined file for all boroughs saved as combined_boroughs.csv')



# %%

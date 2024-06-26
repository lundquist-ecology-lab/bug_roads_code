#%%
import pandas as pd

# Define the boroughs and corresponding file patterns
boroughs = ['bronx', 'manhattan', 'brooklyn', 'queens', 'si']
file_patterns = ['bronx_parks_', 'manhattan_parks_', 'brooklyn_parks_', 'queens_parks_', 'si_parks_']

# Create an empty list to store dataframes
all_dataframes = []

# Iterate through each borough and process the files
for borough, pattern in zip(boroughs, file_patterns):
    # Read the CSV files
    buildings_df = pd.read_csv(f'{pattern}buildings.csv')
    ground_df = pd.read_csv(f'{pattern}ground.csv')
    high_veg_df = pd.read_csv(f'{pattern}high_vegetation.csv')
    low_veg_df = pd.read_csv(f'{pattern}low_vegetation.csv')
    medium_veg_df = pd.read_csv(f'{pattern}medium_vegetation.csv')

    # Select necessary columns from each dataframe
    buildings_df = buildings_df[['fid', 'borough', 'name311', '_count']]
    ground_df = ground_df[['fid', 'borough', 'name311', '_count']]
    high_veg_df = high_veg_df[['fid', 'borough', 'name311', '_count']]
    low_veg_df = low_veg_df[['fid', 'borough', 'name311', '_count']]
    medium_veg_df = medium_veg_df[['fid', 'borough', 'name311', '_count']]
    
    # Rename the count columns before merging to avoid conflicts
    buildings_df.rename(columns={'_count': 'buildings_count'}, inplace=True)
    ground_df.rename(columns={'_count': 'ground_count'}, inplace=True)
    high_veg_df.rename(columns={'_count': 'high_vegetation_count'}, inplace=True)
    low_veg_df.rename(columns={'_count': 'low_vegetation_count'}, inplace=True)
    medium_veg_df.rename(columns={'_count': 'medium_vegetation_count'}, inplace=True)
    
    # Merge the dataframes on 'fid', 'borough', and 'name311'
    merged_df = buildings_df.merge(ground_df, on=['fid', 'borough', 'name311'])
    merged_df = merged_df.merge(high_veg_df, on=['fid', 'borough', 'name311'])
    merged_df = merged_df.merge(low_veg_df, on=['fid', 'borough', 'name311'])
    merged_df = merged_df.merge(medium_veg_df, on=['fid', 'borough', 'name311'])
    
    # Calculate the area in m² for each count column
    merged_df['buildings_area_m2'] = merged_df['buildings_count'] * 0.0833
    merged_df['ground_area_m2'] = merged_df['ground_count'] * 0.0833
    merged_df['high_vegetation_area_m2'] = merged_df['high_vegetation_count'] * 0.0833
    merged_df['low_vegetation_area_m2'] = merged_df['low_vegetation_count'] * 0.0833
    merged_df['medium_vegetation_area_m2'] = merged_df['medium_vegetation_count'] * 0.0833
    
    # Select only the necessary columns
    final_df = merged_df[['fid', 'borough', 'name311', 
                          'buildings_count', 'ground_count', 'high_vegetation_count', 'low_vegetation_count', 'medium_vegetation_count',
                          'buildings_area_m2', 'ground_area_m2', 'high_vegetation_area_m2', 'low_vegetation_area_m2', 'medium_vegetation_area_m2']]
    
    # Save the final dataframe to a new CSV file
    final_df.to_csv(f'{pattern}combined.csv', index=False)
    
    # Append the dataframe to the list
    all_dataframes.append(final_df)

# Concatenate all the dataframes into a single dataframe
full_df = pd.concat(all_dataframes)

# Save the full dataframe to a new CSV file
full_df.to_csv('all_parks_combined.csv', index=False)



# %%

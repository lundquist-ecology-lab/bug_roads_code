# %%
import pandas as pd
from scipy.stats import sem
import numpy as np

# Read the existing CSV file into a DataFrame
df = pd.read_csv('your_csv_file.csv')

## Below should be modified to fit column names of your csv file ##

# Columns of interest
columns_of_interest = [
    'Park1_ID', 'Park1_name311', 'Park1_borough',
    'Park2_ID', 'Park2_name311', 'Park2_borough',
    'Distance_ft', 'Distance_m', 'Count_Intersected_Buildings',
    'Average_heightroof_m'
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

# Filter out zero heights by setting them to NaN
df['Average_heightroof_m'] = df['Average_heightroof_m'].replace(0, np.nan)

# Identify multi-borough paths
df['crosses_multiple_boroughs'] = df['Park1_borough'] != df['Park2_borough']

# Separate dataframes for single-borough and multi-borough paths
single_borough_df = df[~df['crosses_multiple_boroughs']]
multi_borough_df = df[df['crosses_multiple_boroughs']]

# Define a function to handle empty slices in groupby operations
def safe_nanmean(x):
    filtered_x = [i for i in x if not np.isnan(i)]
    return np.nanmean(filtered_x) if len(filtered_x) > 0 else np.nan

def safe_sem(x):
    filtered_x = [i for i in x if not np.isnan(i)]
    return sem(filtered_x, nan_policy='omit') if len(filtered_x) > 1 else np.nan

def safe_nanmin(x):
    filtered_x = [i for i in x if not np.isnan(i)]
    return np.nanmin(filtered_x) if len(filtered_x) > 0 else np.nan

def safe_nanmax(x):
    filtered_x = [i for i in x if not np.isnan(i)]
    return np.nanmax(filtered_x) if len(filtered_x) > 0 else np.nan

def safe_nanmedian(x):
    filtered_x = [i for i in x if not np.isnan(i)]
    return np.nanmedian(filtered_x) if len(filtered_x) > 0 else np.nan

# Calculate the statistics for all parks
def calculate_statistics(df):
    stats = {
        'Average_Number_of_Buildings_Crossed': safe_nanmean(df['Count_Intersected_Buildings']),
        'Standard_Error_Number_of_Buildings_Crossed': safe_sem(df['Count_Intersected_Buildings']),
        'Min_Number_of_Buildings_Crossed': safe_nanmin(df['Count_Intersected_Buildings']),
        'Max_Number_of_Buildings_Crossed': safe_nanmax(df['Count_Intersected_Buildings']),
        'Median_Number_of_Buildings_Crossed': safe_nanmedian(df['Count_Intersected_Buildings']),
        'Average_Height_of_Buildings_Crossed': safe_nanmean(df['Average_heightroof_m']),
        'Standard_Error_Height_of_Buildings_Crossed': safe_sem(df['Average_heightroof_m']),
        'Min_Height_of_Buildings_Crossed': safe_nanmin(df['Average_heightroof_m']),
        'Max_Height_of_Buildings_Crossed': safe_nanmax(df['Average_heightroof_m']),
        'Median_Height_of_Buildings_Crossed': safe_nanmedian(df['Average_heightroof_m']),
        'Number_of_Paths_Considered': df.shape[0]
    }
    return stats

all_parks_stats = calculate_statistics(df)
single_borough_stats = calculate_statistics(single_borough_df)
multi_borough_stats = calculate_statistics(multi_borough_df)

# Calculate the statistics by borough for single-borough paths
borough_stats = single_borough_df.groupby('Park1_borough').agg(
    avg_num_buildings=pd.NamedAgg(column='Count_Intersected_Buildings', aggfunc=safe_nanmean),
    se_num_buildings=pd.NamedAgg(column='Count_Intersected_Buildings', aggfunc=safe_sem),
    min_num_buildings=pd.NamedAgg(column='Count_Intersected_Buildings', aggfunc=safe_nanmin),
    max_num_buildings=pd.NamedAgg(column='Count_Intersected_Buildings', aggfunc=safe_nanmax),
    median_num_buildings=pd.NamedAgg(column='Count_Intersected_Buildings', aggfunc=safe_nanmedian),
    avg_height=pd.NamedAgg(column='Average_heightroof_m', aggfunc=safe_nanmean),
    se_height=pd.NamedAgg(column='Average_heightroof_m', aggfunc=safe_sem),
    min_height=pd.NamedAgg(column='Average_heightroof_m', aggfunc=safe_nanmin),
    max_height=pd.NamedAgg(column='Average_heightroof_m', aggfunc=safe_nanmax),
    median_height=pd.NamedAgg(column='Average_heightroof_m', aggfunc=safe_nanmedian),
    num_paths=pd.NamedAgg(column='unique_pair', aggfunc='count')
).reset_index()

# Calculate mean and SE for each borough combination for multi-borough paths
multi_borough_df['borough_combo'] = multi_borough_df.apply(
    lambda row: '-'.join(sorted([row['Park1_borough'], row['Park2_borough']])), axis=1)

combo_stats = multi_borough_df.groupby('borough_combo').agg(
    avg_num_buildings=pd.NamedAgg(column='Count_Intersected_Buildings', aggfunc=safe_nanmean),
    se_num_buildings=pd.NamedAgg(column='Count_Intersected_Buildings', aggfunc=safe_sem),
    min_num_buildings=pd.NamedAgg(column='Count_Intersected_Buildings', aggfunc=safe_nanmin),
    max_num_buildings=pd.NamedAgg(column='Count_Intersected_Buildings', aggfunc=safe_nanmax),
    median_num_buildings=pd.NamedAgg(column='Count_Intersected_Buildings', aggfunc=safe_nanmedian),
    avg_height=pd.NamedAgg(column='Average_heightroof_m', aggfunc=safe_nanmean),
    se_height=pd.NamedAgg(column='Average_heightroof_m', aggfunc=safe_sem),
    min_height=pd.NamedAgg(column='Average_heightroof_m', aggfunc=safe_nanmin),
    max_height=pd.NamedAgg(column='Average_heightroof_m', aggfunc=safe_nanmax),
    median_height=pd.NamedAgg(column='Average_heightroof_m', aggfunc=safe_nanmedian),
    num_paths=pd.NamedAgg(column='unique_pair', aggfunc='count')
).reset_index()

# Combine all parks statistics into a single DataFrame
all_parks_stats_df = pd.DataFrame({
    'Category': ['All Parks'],
    'Average_Number_of_Buildings_Crossed': [all_parks_stats['Average_Number_of_Buildings_Crossed']],
    'Standard_Error_Number_of_Buildings_Crossed': [all_parks_stats['Standard_Error_Number_of_Buildings_Crossed']],
    'Min_Number_of_Buildings_Crossed': [all_parks_stats['Min_Number_of_Buildings_Crossed']],
    'Max_Number_of_Buildings_Crossed': [all_parks_stats['Max_Number_of_Buildings_Crossed']],
    'Median_Number_of_Buildings_Crossed': [all_parks_stats['Median_Number_of_Buildings_Crossed']],
    'Average_Height_of_Buildings_Crossed': [all_parks_stats['Average_Height_of_Buildings_Crossed']],
    'Standard_Error_Height_of_Buildings_Crossed': [all_parks_stats['Standard_Error_Height_of_Buildings_Crossed']],
    'Min_Height_of_Buildings_Crossed': [all_parks_stats['Min_Height_of_Buildings_Crossed']],
    'Max_Height_of_Buildings_Crossed': [all_parks_stats['Max_Height_of_Buildings_Crossed']],
    'Median_Height_of_Buildings_Crossed': [all_parks_stats['Median_Height_of_Buildings_Crossed']],
    'Number_of_Paths_Considered': [all_parks_stats['Number_of_Paths_Considered']]
})

single_borough_stats_df = pd.DataFrame({
    'Category': ['Single-Borough Paths'],
    'Average_Number_of_Buildings_Crossed': [single_borough_stats['Average_Number_of_Buildings_Crossed']],
    'Standard_Error_Number_of_Buildings_Crossed': [single_borough_stats['Standard_Error_Number_of_Buildings_Crossed']],
    'Min_Number_of_Buildings_Crossed': [single_borough_stats['Min_Number_of_Buildings_Crossed']],
    'Max_Number_of_Buildings_Crossed': [single_borough_stats['Max_Number_of_Buildings_Crossed']],
    'Median_Number_of_Buildings_Crossed': [single_borough_stats['Median_Number_of_Buildings_Crossed']],
    'Average_Height_of_Buildings_Crossed': [single_borough_stats['Average_Height_of_Buildings_Crossed']],
    'Standard_Error_Height_of_Buildings_Crossed': [single_borough_stats['Standard_Error_Height_of_Buildings_Crossed']],
    'Min_Height_of_Buildings_Crossed': [single_borough_stats['Min_Height_of_Buildings_Crossed']],
    'Max_Height_of_Buildings_Crossed': [single_borough_stats['Max_Height_of_Buildings_Crossed']],
    'Median_Height_of_Buildings_Crossed': [single_borough_stats['Median_Height_of_Buildings_Crossed']],
    'Number_of_Paths_Considered': [single_borough_stats['Number_of_Paths_Considered']]
})

multi_borough_stats_df = pd.DataFrame({
    'Category': ['Multi-Borough Paths'],
    'Average_Number_of_Buildings_Crossed': [multi_borough_stats['Average_Number_of_Buildings_Crossed']],
    'Standard_Error_Number_of_Buildings_Crossed': [multi_borough_stats['Standard_Error_Number_of_Buildings_Crossed']],
    'Min_Number_of_Buildings_Crossed': [multi_borough_stats['Min_Number_of_Buildings_Crossed']],
    'Max_Number_of_Buildings_Crossed': [multi_borough_stats['Max_Number_of_Buildings_Crossed']],
    'Median_Number_of_Buildings_Crossed': [multi_borough_stats['Median_Number_of_Buildings_Crossed']],
    'Average_Height_of_Buildings_Crossed': [multi_borough_stats['Average_Height_of_Buildings_Crossed']],
    'Standard_Error_Height_of_Buildings_Crossed': [multi_borough_stats['Standard_Error_Height_of_Buildings_Crossed']],
    'Min_Height_of_Buildings_Crossed': [multi_borough_stats['Min_Height_of_Buildings_Crossed']],
    'Max_Height_of_Buildings_Crossed': [multi_borough_stats['Max_Height_of_Buildings_Crossed']],
    'Median_Height_of_Buildings_Crossed': [multi_borough_stats['Median_Height_of_Buildings_Crossed']],
    'Number_of_Paths_Considered': [multi_borough_stats['Number_of_Paths_Considered']]
})

# Rename columns in borough_stats and combo_stats to match the format
borough_stats.rename(columns={
    'Park1_borough': 'Category',
    'avg_num_buildings': 'Average_Number_of_Buildings_Crossed',
    'se_num_buildings': 'Standard_Error_Number_of_Buildings_Crossed',
    'min_num_buildings': 'Min_Number_of_Buildings_Crossed',
    'max_num_buildings': 'Max_Number_of_Buildings_Crossed',
    'median_num_buildings': 'Median_Number_of_Buildings_Crossed',
    'avg_height': 'Average_Height_of_Buildings_Crossed',
    'se_height': 'Standard_Error_Height_of_Buildings_Crossed',
    'min_height': 'Min_Height_of_Buildings_Crossed',
    'max_height': 'Max_Height_of_Buildings_Crossed',
    'median_height': 'Median_Height_of_Buildings_Crossed',
    'num_paths': 'Number_of_Paths_Considered'
}, inplace=True)

combo_stats.rename(columns={
    'borough_combo': 'Category',
    'avg_num_buildings': 'Average_Number_of_Buildings_Crossed',
    'se_num_buildings': 'Standard_Error_Number_of_Buildings_Crossed',
    'min_num_buildings': 'Min_Number_of_Buildings_Crossed',
    'max_num_buildings': 'Max_Number_of_Buildings_Crossed',
    'median_num_buildings': 'Median_Number_of_Buildings_Crossed',
    'avg_height': 'Average_Height_of_Buildings_Crossed',
    'se_height': 'Standard_Error_Height_of_Buildings_Crossed',
    'min_height': 'Min_Height_of_Buildings_Crossed',
    'max_height': 'Max_Height_of_Buildings_Crossed',
    'median_height': 'Median_Height_of_Buildings_Crossed',
    'num_paths': 'Number_of_Paths_Considered'
}, inplace=True)

# Combine all parks stats, single-borough stats, multi-borough stats, borough stats, and combo stats into one dataframe
final_stats = pd.concat([all_parks_stats_df, single_borough_stats_df, multi_borough_stats_df, borough_stats, combo_stats], ignore_index=True)

# Write the final statistics to a CSV file
output_stats_file = 'parks_buildings_statistics.csv'
final_stats.to_csv(output_stats_file, index=False)

print(f"CSV file '{output_stats_file}' has been created with the statistics.")

# %%

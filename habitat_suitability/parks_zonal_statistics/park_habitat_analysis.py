#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import skew
import os


# Create a directory to save the graphics
output_dir = '/figures'
os.makedirs(output_dir, exist_ok=True)

# Define the boroughs and corresponding file patterns
boroughs = ['bronx', 'manhattan', 'brooklyn', 'queens', 'si']
file_patterns = ['../csv/bronx_parks_', '../csv/manhattan_parks_', '../csv/brooklyn_parks_', '../csv/queens_parks_', '../csv/si_parks_']

# Load the park centroids CSV file to get acres associated with each name311
centroids_df = pd.read_csv('../../../gis/nyc_parks_shapes.csv')
centroids_df = centroids_df[['name311', 'borough', 'acres']]
centroids_df['area_m2'] = centroids_df['acres'] * 4046.86  # Convert acres to m²

# Create an empty list to store dataframes
all_dataframes = []

# Function to generate summary statistics and plots for a given dataframe
def generate_summary_stats(df, borough_name):
    # Merge the combined dataframe with the centroids dataframe on 'name311'
    merged_df = pd.merge(df, centroids_df, on='name311')

    # Print column names for debugging
    print(f"Columns after merging for {borough_name}: {merged_df.columns}")

    # Handle borough columns
    merged_df['borough'] = merged_df['borough_x'].combine_first(merged_df['borough_y'])
    merged_df.drop(columns=['borough_x', 'borough_y'], inplace=True)

    # Aggregate data by 'name311'
    area_columns_m2 = ['buildings_area_m2', 'ground_area_m2', 'high_vegetation_area_m2', 'low_vegetation_area_m2', 'medium_vegetation_area_m2']
    aggregated_df = merged_df.groupby('name311')[area_columns_m2 + ['area_m2']].sum().reset_index()

    # Calculate summary statistics
    mean_values = aggregated_df[area_columns_m2].mean()
    se_values = aggregated_df[area_columns_m2].sem()
    quartiles = aggregated_df[area_columns_m2].quantile([0.25, 0.5, 0.75])
    range_values = aggregated_df[area_columns_m2].apply(lambda x: x.max() - x.min())
    skew_values = aggregated_df[area_columns_m2].apply(lambda x: skew(x))
    min_values = aggregated_df[area_columns_m2].min()
    max_values = aggregated_df[area_columns_m2].max()
    min_park_size_m2 = aggregated_df['area_m2'].min()
    max_park_size_m2 = aggregated_df['area_m2'].max()

    # Get the names and boroughs of the smallest and largest parks
    smallest_park = merged_df.loc[merged_df['area_m2'].idxmin(), ['name311', 'borough', 'area_m2']]
    largest_park = merged_df.loc[merged_df['area_m2'].idxmax(), ['name311', 'borough', 'area_m2']]

    # Create a DataFrame to store all summary statistics
    summary_stats = pd.DataFrame({
        'Mean': mean_values,
        'Standard Error': se_values,
        'Q1': quartiles.loc[0.25],
        'Median': quartiles.loc[0.5],
        'Q3': quartiles.loc[0.75],
        'Range': range_values,
        'Skewness': skew_values,
        'Min': min_values,
        'Max': max_values
    })

    # Add min and max park sizes to the summary statistics DataFrame
    summary_stats.loc['Park Size'] = [None] * len(summary_stats.columns)
    summary_stats.at['Park Size', 'Min'] = min_park_size_m2
    summary_stats.at['Park Size', 'Max'] = max_park_size_m2

    # Save the summary statistics to a CSV file
    summary_stats.to_csv(f'{output_dir}/{borough_name}_summary_statistics_m2.csv', index_label='Cover Type')

    # Create box plots with different colors
    area_columns_renamed = {
        'buildings_area_m2': 'B',
        'ground_area_m2': 'G',
        'low_vegetation_area_m2': 'LV',
        'medium_vegetation_area_m2': 'MV',
        'high_vegetation_area_m2': 'HV'
    }
    aggregated_df_renamed = aggregated_df.rename(columns=area_columns_renamed)
    plt.figure(figsize=(14, 10))
    df_melted = aggregated_df_renamed.melt(value_vars=area_columns_renamed.values(), var_name='Cover Type', value_name='Area (m²)')

    # Use different colors for each box
    palette = sns.color_palette("husl", len(area_columns_renamed))
    sns.boxplot(x='Cover Type', y='Area (m²)', data=df_melted, palette=palette, hue='Cover Type')
    plt.yscale('log')  # Use log scale for the y-axis to handle wide range of values
    plt.tick_params(axis='y', which='minor', length=0)  # Remove minor tick marks
    # plt.title(f'{borough_name.capitalize()}', fontsize=14, loc='left')
    plt.xlabel('Cover Type', fontsize=12)
    plt.ylabel('Log-Area (m²)', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Cover Type')
    plt.savefig(f'{output_dir}/{borough_name}_boxplot.png', bbox_inches='tight')
    plt.close()

    # Calculate Spearman's correlation matrix
    spearman_corr = aggregated_df_renamed[area_columns_renamed.values()].corr(method='spearman')

    # Create a heatmap for the Spearman's correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(spearman_corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True, annot_kws={"size": 10})
    # plt.title(f'{borough_name.capitalize()} Spearman Correlation Matrix Heatmap', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig(f'{output_dir}/{borough_name}_heatmap.png')
    plt.close()

    return aggregated_df, smallest_park, largest_park

# Iterate through each borough and process the files
for borough, pattern in zip(boroughs, file_patterns):
    # Read the CSV files
    buildings_df = pd.read_csv(f'{pattern}buildings.csv')
    ground_df = pd.read_csv(f'{pattern}ground.csv')
    high_veg_df = pd.read_csv(f'{pattern}high_vegetation.csv')
    low_veg_df = pd.read_csv(f'{pattern}low_vegetation.csv')
    medium_veg_df = pd.read_csv(f'{pattern}medium_vegetation.csv')

    # Select necessary columns from each dataframe
    buildings_df = buildings_df[['name311', 'borough', '_count']]
    ground_df = ground_df[['name311', 'borough', '_count']]
    high_veg_df = high_veg_df[['name311', 'borough', '_count']]
    low_veg_df = low_veg_df[['name311', 'borough', '_count']]
    medium_veg_df = medium_veg_df[['name311', 'borough', '_count']]
    
    # Rename the count columns before merging to avoid conflicts
    buildings_df.rename(columns={'_count': 'buildings_count'}, inplace=True)
    ground_df.rename(columns={'_count': 'ground_count'}, inplace=True)
    high_veg_df.rename(columns={'_count': 'high_vegetation_count'}, inplace=True)
    low_veg_df.rename(columns={'_count': 'low_vegetation_count'}, inplace=True)
    medium_veg_df.rename(columns={'_count': 'medium_vegetation_count'}, inplace=True)
    
    # Merge the dataframes on 'name311' and 'borough'
    merged_df = buildings_df.merge(ground_df, on=['name311', 'borough'])
    merged_df = merged_df.merge(high_veg_df, on=['name311', 'borough'])
    merged_df = merged_df.merge(low_veg_df, on=['name311', 'borough'])
    merged_df = merged_df.merge(medium_veg_df, on=['name311', 'borough'])
    
    # Calculate the area in m² for each count column
    merged_df['buildings_area_m2'] = merged_df['buildings_count'] * 0.0833
    merged_df['ground_area_m2'] = merged_df['ground_count'] * 0.0833
    merged_df['high_vegetation_area_m2'] = merged_df['high_vegetation_count'] * 0.0833
    merged_df['low_vegetation_area_m2'] = merged_df['low_vegetation_count'] * 0.0833
    merged_df['medium_vegetation_area_m2'] = merged_df['medium_vegetation_count'] * 0.0833
    
    # Select only the necessary columns
    final_df = merged_df[['name311', 'borough', 
                          'buildings_count', 'ground_count', 'high_vegetation_count', 'low_vegetation_count', 'medium_vegetation_count',
                          'buildings_area_m2', 'ground_area_m2', 'high_vegetation_area_m2', 'low_vegetation_area_m2', 'medium_vegetation_area_m2']]
    
    # Save the final dataframe to a new CSV file
    final_df.to_csv(f'{pattern}combined.csv', index=False)
    
    # Append the dataframe to the list
    all_dataframes.append(final_df)
    
    # Generate summary statistics and plots for the current borough
    generate_summary_stats(final_df, borough)

# Concatenate all the dataframes into a single dataframe
full_df = pd.concat(all_dataframes)

# Save the full dataframe to a new CSV file
full_df.to_csv('all_parks_combined.csv', index=False)

# Generate summary statistics and plots for the whole city
aggregated_full_df, smallest_park, largest_park = generate_summary_stats(full_df, 'city')

from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation

# Calculate the quartiles for park sizes
park_size_quartiles = aggregated_full_df['area_m2'].quantile([0.25, 0.5, 0.75])

# Compute the cumulative distribution function (CDF)
aggregated_full_df['cdf'] = aggregated_full_df['area_m2'].rank(method='average', pct=True)

# Custom formatter for logarithmic scale
def log_formatter(x, pos):
    return f'{x:.1e}'

# Plot the CDF with logarithmic x-axis
plt.figure(figsize=(14, 10))

# Plot with vibrant colors
sns.lineplot(x='area_m2', y='cdf', data=aggregated_full_df, marker='o', linestyle='-', markersize=4, color='blue', markerfacecolor='blue', markeredgecolor='blue')
# plt.title('Cumulative Distribution of Park Sizes in NYC', fontsize=14, color='black')
plt.xlabel('Park Size (m²) (Log Scale)', fontsize=12, color='black')
plt.ylabel('Cumulative proportion', fontsize=12, color='black')
plt.xscale('log')
plt.xticks(fontsize=10, color='black')
plt.yticks(fontsize=10, color='black')
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='black')

# Set x-axis to logarithmic scale with scientific notation
ax = plt.gca()
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0e}'))
ax.xaxis.set_minor_formatter(LogFormatterSciNotation())

# Set custom ticks for better readability
custom_ticks = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
ax.set_xticks(custom_ticks)
ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

# Save the plot as a PNG file
plt.savefig(f'{output_dir}/nyc_park_size_cdf_log.png', bbox_inches='tight', dpi=300)
plt.close()

# Calculate the percentage of parks within each quartile
total_parks = len(aggregated_full_df)
q1_parks = len(aggregated_full_df[aggregated_full_df['area_m2'] <= park_size_quartiles[0.25]])
q2_parks = len(aggregated_full_df[(aggregated_full_df['area_m2'] > park_size_quartiles[0.25]) & 
                                  (aggregated_full_df['area_m2'] <= park_size_quartiles[0.5])])
q3_parks = len(aggregated_full_df[(aggregated_full_df['area_m2'] > park_size_quartiles[0.5]) & 
                                  (aggregated_full_df['area_m2'] <= park_size_quartiles[0.75])])
q4_parks = len(aggregated_full_df[aggregated_full_df['area_m2'] > park_size_quartiles[0.75]])

q1_percent = (q1_parks / total_parks) * 100
q2_percent = (q2_parks / total_parks) * 100
q3_percent = (q3_parks / total_parks) * 100
q4_percent = (q4_parks / total_parks) * 100

# Print the results
print("Park Size Quartiles (m²):")
print(f"Q1: {park_size_quartiles[0.25]:.4f} m² - {q1_percent:.2f}% of parks")
print(f"Median: {park_size_quartiles[0.5]:.4f} m² - {q2_percent:.2f}% of parks")
print(f"Q3: {park_size_quartiles[0.75]:.4f} m² - {q3_percent:.2f}% of parks")
print(f"Q4: > {park_size_quartiles[0.75]:.4f} m² - {q4_percent:.2f}% of parks")

# Print the smallest and largest parks
print("\nSmallest Park:")
print(f"Name: {smallest_park['name311']}, Borough: {smallest_park['borough']}, Size: {smallest_park['area_m2']:.4f} m²")

print("\nLargest Park:")
print(f"Name: {largest_park['name311']}, Borough: {largest_park['borough']}, Size: {largest_park['area_m2']:.4f} m²")

# Histogram for park sizes with log transformation
plt.figure(figsize=(14, 10))
sns.histplot(np.log1p(aggregated_full_df['area_m2']), bins=30, kde=True, color='blue')
# plt.title('Log-Transformed Distribution of Park Sizes in NYC', fontsize=14)
plt.xlabel('Log(Park Size (m²))', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.savefig(f'{output_dir}/nyc_park_size_histogram_log.png')
plt.close()

# Histogram for cover amounts in NYC with log transformation
cover_columns = ['buildings_area_m2', 'ground_area_m2', 'high_vegetation_area_m2', 'low_vegetation_area_m2', 'medium_vegetation_area_m2']
plt.figure(figsize=(14, 10))
palette = sns.color_palette("Set2", len(cover_columns))
for i, column in enumerate(cover_columns):
    sns.histplot(np.log1p(aggregated_full_df[column]), bins=30, kde=True, label=column, color=palette[i])
# plt.title('Log-Transformed Distribution of Cover Amounts in NYC', fontsize=14)
plt.xlabel('Log(Cover Amount (m²))', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Cover Type')
plt.savefig(f'{output_dir}/nyc_cover_amounts_histogram_log.png', bbox_inches='tight')
plt.close()

# Prepare the cover data
cover_columns = ['high_vegetation_area_m2', 'medium_vegetation_area_m2', 'low_vegetation_area_m2', 'buildings_area_m2', 'ground_area_m2']
cover_data = aggregated_full_df[cover_columns].melt(var_name='Cover Type', value_name='Area (m²)')

# Compute the cumulative distribution function (CDF) for each cover type
cover_data['cdf'] = cover_data.groupby('Cover Type')['Area (m²)'].rank(method='average', pct=True)

# Custom formatter for logarithmic scale
def log_formatter(x, pos):
    return f'{x:.1e}'

# Define vibrant colors
vibrant_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

custom_labels = ['High Vegetation', 'Low Vegetation', 'Medium Vegetation', 'Buildings', 'Ground']

# Plot the CDF with logarithmic x-axis for each cover type
plt.figure(figsize=(16, 10))

# Loop through each cover type to assign a different vibrant color
for i, cover_type in enumerate(cover_data['Cover Type'].unique()):
    subset = cover_data[cover_data['Cover Type'] == cover_type]
    sns.lineplot(
        x='Area (m²)', y='cdf', 
        data=subset, 
        marker='o', 
        linestyle='-', 
        markersize=4, 
        color=vibrant_colors[i % len(vibrant_colors)],  # Set color to a vibrant color
        markerfacecolor=vibrant_colors[i % len(vibrant_colors)],  # Set marker face color to a vibrant color
        markeredgecolor=vibrant_colors[i % len(vibrant_colors)],  # Set marker edge color to a vibrant color
        label=cover_type
    )

# plt.title('Cumulative Distribution of Cover Types in NYC', fontsize=14, color='black')
plt.xlabel('Area (m²; Log Scale)', fontsize=12, color='black')
plt.ylabel('Cumulative proportion', fontsize=12, color='black')
plt.xscale('log')
plt.xticks(fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
plt.grid(True, which='both', linestyle='-', linewidth=1, color='black')

# Set x-axis to logarithmic scale with scientific notation
ax = plt.gca()
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0e}'))
ax.xaxis.set_minor_formatter(LogFormatterSciNotation())

# Set custom ticks for better readability
custom_ticks = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
ax.set_xticks(custom_ticks)
ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

# Create custom legend handles
legend_handles = [mpatches.Patch(color=vibrant_colors[i], label=label) for i, label in enumerate(custom_labels)]

# Add a legend to the right of the plot with custom handles and labels
plt.legend(handles=legend_handles, title='Cover Type', fontsize=10, title_fontsize=12, frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))

# Save the plot as a PNG file
plt.savefig(f'{output_dir}/nyc_cover_types_cdf_log.png', bbox_inches='tight', dpi=300)
plt.close()
  
# Calculate the percentage of areas within each quartile for each cover type
cover_quartiles = cover_data.groupby('Cover Type')['Area (m²)'].quantile([0.25, 0.5, 0.75]).unstack()
cover_totals = cover_data.groupby('Cover Type')['Area (m²)'].count()

quartile_percentages = cover_data.groupby('Cover Type').apply(lambda df: pd.Series({
    'Q1 (%)': (df['Area (m²)'] <= df['Area (m²)'].quantile(0.25)).mean() * 100,
    'Median (%)': (df['Area (m²)'] <= df['Area (m²)'].quantile(0.5)).mean() * 100,
    'Q3 (%)': (df['Area (m²)'] <= df['Area (m²)'].quantile(0.75)).mean() * 100,
    'Q4 (%)': (df['Area (m²)'] > df['Area (m²)'].quantile(0.75)).mean() * 100,
}))

# Print the results
print("Quartile Percentages for Each Cover Type:")
print(quartile_percentages)

# %%

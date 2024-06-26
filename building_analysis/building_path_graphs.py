#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
import seaborn as sns
import matplotlib.gridspec as gridspec

# Read the existing CSV file into a dataframe
df = pd.read_csv('your_file.csv')

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
def safe_literal_eval(val):
    try:
        return literal_eval(val)
    except (ValueError, SyntaxError):
        return []

df['Intersected_heightroofs_m'] = df['Intersected_heightroofs_m'].apply(lambda x: safe_literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else []))

# Remove zero heights and convert lists to a numpy array
df['Intersected_heightroofs_m'] = df['Intersected_heightroofs_m'].apply(lambda lst: [h if h > 0 else np.nan for h in lst])

# Determine the maximum length of the lists
max_length = df['Intersected_heightroofs_m'].apply(len).max()

# Pad lists with NaNs to make them the same length, adding NaNs only at the end
df['Intersected_heightroofs_m'] = df['Intersected_heightroofs_m'].apply(lambda x: x + [np.nan] * (max_length - len(x)))

# Function to calculate statistics and plot
def plot_summary_roof_heights_and_heatmap(ax, df, title, tick_interval):
    # Convert lists to a numpy array
    height_matrix = np.array(df['Intersected_heightroofs_m'].tolist())
    
    # Calculate mean, min, and max for each segment, handling all-NaN slices
    with np.errstate(all='ignore'):
        mean_heights = np.nanmean(height_matrix, axis=0)
        min_heights = np.nanmin(height_matrix, axis=0)
        max_heights = np.nanmax(height_matrix, axis=0)
    
    # Replace all-NaN results with NaNs
    mean_heights[np.isnan(mean_heights)] = np.nan
    min_heights[np.isnan(min_heights)] = np.nan
    max_heights[np.isnan(max_heights)] = np.nan
    
    # Calculate the number of paths that have a particular number of buildings in their path
    num_buildings = np.sum(~np.isnan(height_matrix), axis=1)
    building_counts = np.bincount(num_buildings.astype(int), minlength=max_length + 1)
    
    # Determine the maximum number of buildings in this subset
    max_num_buildings = np.max(num_buildings)

    # Plot mean roof heights
    ax[0].plot(range(1, max_num_buildings + 1), mean_heights[:max_num_buildings], label='Mean building height', color='black', linestyle='-')
    ax[0].fill_between(range(1, max_num_buildings + 1), min_heights[:max_num_buildings], max_heights[:max_num_buildings], color='skyblue', alpha=0.5, label='Min-Max building height')
    ax[0].set_title(title, loc='left', fontsize=20, color='black')
    ax[0].set_xlabel('$i^{th}$ building', fontsize=16, color='black')
    ax[0].set_ylabel('Roof Height (m)', fontsize=16, color='black')
    ax[0].legend(fontsize=14)
    # ax[0].grid(False, linestyle='--', alpha=0.7)
    ax[0].set_xlim(1, max_num_buildings + 1)  # Extend x-axis slightly beyond max number of buildings
    ax[0].set_xticks(range(1, max_num_buildings + 1, tick_interval))
    ax[0].tick_params(axis='x', labelsize=12, colors='black')
    ax[0].tick_params(axis='y', labelsize=12, colors='black')

    # Create a heatmap of building counts, adjusted to the max number of buildings in this group
    sns.heatmap([building_counts[1:max_num_buildings + 1]], cmap="Blues", ax=ax[1], cbar=True, xticklabels=True, yticklabels=False, cbar_kws={'orientation': 'vertical'}, annot_kws={"size": 12})
    ax[1].set_xlabel('Paths with $i$ buildings', fontsize=16, color='black')
    ax[1].set_yticks([])
    ax[1].set_xlim(1, max_num_buildings)  # Keep heatmap x-axis confined to the max number of buildings
    ax[1].set_xticks(range(1, max_num_buildings + 1, tick_interval))
    
    tick_positions = range(1, max_num_buildings + 1, tick_interval)
    ax[1].set_xticks(tick_positions)
    ax[1].set_xticklabels([str(pos) for pos in tick_positions], rotation=0, fontsize=12, color='black')  # Set rotation to 0 degrees for flat labels

# Create the figure and a grid spec layout
borough_order = ['X', 'M', 'B', 'Q', 'R']
borough_names = ['Bronx', 'Manhattan', 'Brooklyn', 'Queens', 'Staten Island']
boroughs = df['Park1_borough'].unique()
num_boroughs = len(borough_order)

fig = plt.figure(figsize=(18, 6 * (1 + num_boroughs)))  # Adjust the figure size for better display
gs = gridspec.GridSpec(2 + num_boroughs, 2, height_ratios=[4, 2] + [2] * num_boroughs, width_ratios=[3, 1])

# Plot for all parks
ax_all_parks_mean = plt.subplot(gs[0, :])
ax_all_parks_heatmap = plt.subplot(gs[1, :])
plot_summary_roof_heights_and_heatmap([ax_all_parks_mean, ax_all_parks_heatmap], df, 'New York City', tick_interval=10)

# Plot for each borough in the specified order
for i, (borough_code, borough_name) in enumerate(zip(borough_order, borough_names)):
    ax_borough_mean = plt.subplot(gs[i + 2, 0])
    ax_borough_heatmap = plt.subplot(gs[i + 2, 1])
    borough_df = df[df['Park1_borough'] == borough_code]
    if not borough_df.empty:
        plot_summary_roof_heights_and_heatmap([ax_borough_mean, ax_borough_heatmap], borough_df, borough_name, tick_interval=8)
    else:
        print(f"No valid roof height data for borough: {borough_name}")

plt.tight_layout()
plt.show()

# %%

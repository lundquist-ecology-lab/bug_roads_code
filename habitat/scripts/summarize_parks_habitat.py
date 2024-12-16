import pandas as pd
import numpy as np

# Load the data
file_path = '../outputs/lidar_results_parks_121024/combined_results_summary.csv'
data = pd.read_csv(file_path)

# Remove '*' from column names and values if present
data.columns = data.columns.str.replace('*', '', regex=False)
data['category'] = data['category'].str.replace('*', '', regex=False)

# Debugging: Check unique values in 'category'
print("Unique categories:", data['category'].unique())

# Combine vegetation categories into one unit
data['vegetation'] = data['category'].isin(['low_vegetation', 'medium_vegetation', 'high_vegetation'])
data_combined = data[data['vegetation']]

# Debugging: Check data after combining vegetation
print("Data combined shape:", data_combined.shape)
if data_combined.empty:
    print("Error: No data after filtering for vegetation categories!")

# Group by borough and calculate mean and standard error for combined vegetation
summary = (
    data_combined.groupby(['borough'])
    .agg(
        mean_pixel_count=('pixel_count', 'mean'),
        se_pixel_count=('pixel_count', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        mean_area_m2=('area_m2', 'mean'),
        se_area_m2=('area_m2', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
    )
    .reset_index()
)

# Calculate min and max area for each point_id (or unique_id)
min_max_area = (
    data_combined.groupby('point_id')
    .agg(
        min_area=('area_m2', 'min'),
        max_area=('area_m2', 'max')
    )
    .reset_index()
)

# Debugging: Check min and max areas
print("Min and Max Areas per Point ID:")
print(min_max_area.head())

# Merge min and max areas into the summary grouped by borough
summary_with_min_max = (
    data_combined.groupby(['borough'])
    .agg(
        min_area=('area_m2', 'min'),
        max_area=('area_m2', 'max')
    )
    .reset_index()
)
summary = pd.merge(summary, summary_with_min_max, on='borough', how='left')

# Calculate the % of properties meeting vegetation thresholds for different radii
def calculate_percentage(data, radius):
    required_area = (np.pi * (radius ** 2)) * 0.116  # 11.6% of the circle area
    grouped = data.groupby(['borough', 'unique_id']).agg(total_vegetation=('area_m2', 'sum')).reset_index()
    print(f"Radius {radius} m - Required Area: {required_area}")  # Debugging
    print("Grouped data preview:")
    print(grouped.head())  # Debugging: Show grouped data
    filtered = grouped[grouped['total_vegetation'] >= required_area]
    print(f"Filtered Properties: {len(filtered)} / {len(grouped)}")  # Debugging
    percentage = (len(filtered) / len(grouped)) * 100 if len(grouped) > 0 else 0
    return percentage

radii = [100, 250, 500, 1000]
percentages = {radius: calculate_percentage(data_combined, radius) for radius in radii}

# Save the summary to a text file
output_file = '../outputs/summary_parks_habitat.txt'
with open(output_file, 'w') as f:
    for _, row in summary.iterrows():
        f.write(f"Borough: {row['borough']}\n")
        f.write(f"  Mean Pixel Count: {row['mean_pixel_count']:.2f}\n")
        f.write(f"  SE Pixel Count: {row['se_pixel_count']:.2f}\n")
        f.write(f"  Mean Area (m^2): {row['mean_area_m2']:.2f}\n")
        f.write(f"  SE Area (m^2): {row['se_area_m2']:.2f}\n")
        f.write(f"  Min Area (m^2): {row['min_area']:.2f}\n")
        f.write(f"  Max Area (m^2): {row['max_area']:.2f}\n\n")

    f.write("Percentage of properties with sufficient vegetation:\n")
    for radius, percentage in percentages.items():
        f.write(f"  Radius {radius} m: {percentage:.2f}%\n")

print(f"Summary saved to {output_file}")

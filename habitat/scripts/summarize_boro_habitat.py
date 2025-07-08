import pandas as pd
import numpy as np

# Load the data
file_path = r'..\outputs\citywide_lidar_results\combined_results_summary.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Remove '*' from column names and values if present
data.columns = data.columns.str.replace('*', '', regex=False)
data['category'] = data['category'].str.replace('*', '', regex=False)

# Debugging: Check unique values in 'category'
print("Unique categories:", data['category'].unique())

# Convert pixel count to area in m² and then to km²
data['area_m2'] = data['pixel_count'] / 12  # 12 pixels = 1 m²
data['area_km2'] = data['area_m2'] / 1e6   # Convert m² to km²

# Group by borough and category to calculate total area in km²
borough_land_type_summary = (
    data.groupby(['borough', 'category'])
    .agg(total_area_km2=('area_km2', 'sum'))  # Sum the total area in km²
    .reset_index()
)

# Debugging: Check the borough and land type summary
print("Borough and land type summary:")
print(borough_land_type_summary)

# Save the results to a text file
output_text_file = r'..\outputs\borough_land_type_summary.txt'
with open(output_text_file, 'w') as f:
    f.write("Total Area by Borough and Land Type (in km²):\n")
    for borough in borough_land_type_summary['borough'].unique():
        f.write(f"\nBorough: {borough}\n")
        borough_data = borough_land_type_summary[borough_land_type_summary['borough'] == borough]
        for _, row in borough_data.iterrows():
            f.write(f"  Land Type: {row['category']}, Total Area: {row['total_area_km2']:.2f} km²\n")

# Additional Step: Combine vegetation categories into one unit for analysis
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
        mean_area_km2=('area_km2', 'mean'),
        se_area_km2=('area_km2', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
    )
    .reset_index()
)

# Save the vegetation analysis summary to the same text file
with open(output_text_file, 'a') as f:  # Append mode
    f.write("\nBorough-Level Vegetation Analysis:\n")
    for _, row in summary.iterrows():
        f.write(f"\nBorough: {row['borough']}\n")
        f.write(f"  Mean Pixel Count: {row['mean_pixel_count']:.2f}\n")
        f.write(f"  SE Pixel Count: {row['se_pixel_count']:.2f}\n")
        f.write(f"  Mean Area (km²): {row['mean_area_km2']:.4f}\n")
        f.write(f"  SE Area (km²): {row['se_area_km2']:.4f}\n")

print(f"Complete summary saved to {output_text_file}")

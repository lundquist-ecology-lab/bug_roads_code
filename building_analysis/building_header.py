#%%
import fiona

file_path = 'your_file.gpkg'

# Open the file with fiona
with fiona.open(file_path) as src:
    # Get the schema
    schema = src.schema

# Print the column names
print("Column names in the GeoPackage file:")
print(schema['properties'].keys())

# %%
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

# Load the GeoPackage file
gdf = gpd.read_file(r"your_file.gpkg")

# Ensure the 'heightroof' column exists (Change for other data types)
if 'heightroof' not in gdf.columns:
    raise ValueError("The 'heightroof' column is not found in the GeoPackage file.")

# Convert heightroof from feet to meters
gdf['heightroof_m'] = gdf['heightroof'] * 0.3048

# Calculate statistics
heightroof_m = gdf['heightroof_m'].dropna()  # Remove NaN values

count = heightroof_m.count()
average = heightroof_m.mean()
standard_error = sem(heightroof_m)
median = heightroof_m.median()

# Display statistics
print(f"Count: {count}")
print(f"Average (meters): {average}")
print(f"Standard Error (meters): {standard_error}")
print(f"Median (meters): {median}")

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(heightroof_m, bins=30, edgecolor='black')
plt.title('Histogram of Roof Heights (in meters)')
plt.xlabel('Roof Height (meters)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# %%
# Extract borough from BIN
gdf['borough'] = gdf['bin'].astype(str).str[0]

# Define a mapping from BIN first digit to borough name
borough_mapping = {
    '1': 'Manhattan',
    '2': 'The Bronx',
    '3': 'Brooklyn',
    '4': 'Queens',
    '5': 'Staten Island'
}

# Apply the mapping
gdf['borough'] = gdf['borough'].map(borough_mapping)

# Function to calculate statistics
def calculate_statistics(df, borough_name):
    heightroof_m = df['heightroof_m'].dropna()
    count = heightroof_m.count()
    average = heightroof_m.mean()
    standard_error = sem(heightroof_m)
    median = heightroof_m.median()
    print(f"\nBorough: {borough_name}")
    print(f"Count: {count}")
    print(f"Average (meters): {average}")
    print(f"Standard Error (meters): {standard_error}")
    print(f"Median (meters): {median}")

    plt.figure(figsize=(10, 6))
    plt.hist(heightroof_m, bins=30, edgecolor='black')
    plt.title(f'Histogram of Roof Heights in {borough_name} (in meters)')
    plt.xlabel('Roof Height (meters)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Calculate and display statistics for each borough
for borough, borough_name in borough_mapping.items():
    borough_df = gdf[gdf['borough'] == borough_name]
    if not borough_df.empty:
        calculate_statistics(borough_df, borough_name)

# %%

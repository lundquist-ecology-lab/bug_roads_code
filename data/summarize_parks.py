import geopandas as gpd
import pandas as pd

# Load the input GeoPackage files
park_data_path = "../open_and_park_single_26918.gpkg"
borough_boundaries_path = "../borough_boundaries_26918.gpkg"

# Read the GeoPackage files
park_data = gpd.read_file(park_data_path)
borough_boundaries = gpd.read_file(borough_boundaries_path)

# Ensure both datasets are in the same coordinate reference system (CRS)
if park_data.crs != borough_boundaries.crs:
    park_data = park_data.to_crs(borough_boundaries.crs)

# Calculate the total number of rows in the park data
total_rows = len(park_data)

# Assign each park to the borough it intersects the most
def assign_borough_by_largest_intersection(park_row):
    intersecting_boroughs = borough_boundaries[borough_boundaries.geometry.intersects(park_row.geometry)]
    if not intersecting_boroughs.empty:
        # Calculate the intersection areas
        intersecting_boroughs = intersecting_boroughs.copy()
        intersecting_boroughs['intersection_area'] = intersecting_boroughs.geometry.intersection(park_row.geometry).area
        # Return the borough with the largest intersection area
        largest_borough = intersecting_boroughs.sort_values(by='intersection_area', ascending=False).iloc[0]
        return largest_borough['boro_name']
    return None

# Apply the function to assign boroughs
park_data['boro_name_initial'] = park_data.apply(assign_borough_by_largest_intersection, axis=1)

# Group rows by unique_id and assign the borough with the majority count
def assign_majority_borough(group):
    if not group.mode().empty:
        return group.mode()[0]
    return None

if 'unique_id' in park_data.columns:
    park_data['boro_name'] = park_data.groupby('unique_id')['boro_name_initial'].transform(assign_majority_borough)

# Drop the initial column used for intermediate calculation
park_data = park_data.drop(columns=['boro_name_initial'])

# Count the number of unmatched rows (parks that do not intersect any borough)
unmatched_count = park_data['boro_name'].isna().sum()

# Count the number of rows in each borough
borough_counts = park_data['boro_name'].value_counts()

# Display results
print(f"Total number of rows in park data: {total_rows}")
print(f"Number of rows not matching any borough: {unmatched_count}")
print("\nNumber of rows in each borough:")
print(borough_counts)

# Optional: Save results to a TXT file
with open("borough_row_counts.txt", "w") as file:
    file.write(f"Total number of rows in park data: {total_rows}\n")
    file.write(f"Number of rows not matching any borough: {unmatched_count}\n\n")
    file.write("Number of rows in each borough:\n")
    file.write(borough_counts.to_string())
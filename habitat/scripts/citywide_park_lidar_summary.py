import pandas as pd

# Load the CSV file
csv_file = "..\outputs\lidar_results_parks_121024\combined_results_summary.csv"
data = pd.read_csv(csv_file)

# Calculate total area per category for all boroughs and each borough
summary = data.groupby(['borough', 'category']).agg(
    total_area_km2=('area_m2', lambda x: x.sum() / 1e6),  # Convert m² to km²
    total_unique_ids=('point_id', 'nunique'),
    total_pixel_count=('pixel_count', 'sum')
).reset_index()

# Total for all boroughs combined
overall = data.groupby(['category']).agg(
    total_area_km2=('area_m2', lambda x: x.sum() / 1e6),
    total_unique_ids=('point_id', 'nunique'),
    total_pixel_count=('pixel_count', 'sum')
).reset_index()
overall.insert(0, 'borough', 'All')  # Add a column for 'All' boroughs

# Combine borough and overall summary
summary = pd.concat([summary, overall])

# Save the summary to a text file
output_file = "borough_summary.txt"
summary.to_csv(output_file, sep='\t', index=False)

print(f"Summary saved to {output_file}")

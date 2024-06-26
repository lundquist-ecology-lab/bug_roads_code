# %%
import pandas as pd

# Load the CSV file
file_path = '/data/nyc_paths_1000m.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Filter out rows where start_fid equals end_fid
df_filtered = df[df['start_fid'] != df['end_fid']]

# Remove duplicate rows based on start_fid and end_fid combination
df_unique = df_filtered.drop_duplicates(subset=['start_fid', 'end_fid'])

# Remove rows that have same names
df_final = df_unique[df_unique['start_name'] != df_unique['end_name']]

# Save the final filtered DataFrame to a new CSV file
output_file_path = 'NYC_paths_filtered.csv'  # Replace with your desired output file path
df_final.to_csv(output_file_path, index=False)

print(f"Filtered data saved to {output_file_path}")


# %%

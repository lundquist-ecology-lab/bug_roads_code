import geopandas as gpd
import pandas as pd

def separate_duplicates_from_gpkg(input_file, unique_output_file, duplicates_output_file):
    """
    Separates duplicate rows from a GPKG file into two files:
    one with unique rows (first occurrence) and one with duplicates.
    
    Parameters:
    input_file (str): Path to input GPKG file
    unique_output_file (str): Path to save the unique rows
    duplicates_output_file (str): Path to save the duplicate rows
    
    Returns:
    tuple: (total_rows, unique_rows, num_duplicates)
    """
    # Read the GPKG file
    gdf = gpd.read_file(input_file)
    
    # Store initial count
    total_rows = len(gdf)
    
    # Columns to check for duplicates
    columns_to_check = ['park1_unique_id', 'park1_point_id', 
                       'park2_unique_id', 'park2_point_id']
    
    # Create a mask for duplicate rows (marking all duplicates including first occurrence)
    duplicate_mask = gdf.duplicated(subset=columns_to_check, keep=False)
    
    # Split the data
    duplicates_df = gdf[duplicate_mask]
    unique_df = gdf.drop_duplicates(subset=columns_to_check, keep='first')
    
    # Save the unique rows
    unique_df.to_file(unique_output_file, driver='GPKG')
    
    # Save the duplicates (if any exist)
    if len(duplicates_df) > 0:
        duplicates_df.to_file(duplicates_output_file, driver='GPKG')
    
    return total_rows, len(unique_df), len(duplicates_df)

if __name__ == "__main__":
    # Example usage
    input_gpkg = "results.gpkg"
    unique_output = "results_unique.gpkg"
    duplicates_output = "results_duplicates.gpkg"
    
    total, unique, duplicates = separate_duplicates_from_gpkg(
        input_gpkg, unique_output, duplicates_output
    )
    
    print(f"Total rows processed: {total}")
    print(f"Unique rows saved to {unique_output}: {unique}")
    print(f"Duplicate rows saved to {duplicates_output}: {duplicates}")
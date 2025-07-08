import pandas as pd
import numpy as np

# Load the data from new output location
file_path = '../outputs/park_lidar_results/combined_results_summary.csv'
data = pd.read_csv(file_path)

# Remove '*' from column names and values if present
data.columns = data.columns.str.replace('*', '', regex=False)
data['category'] = data['category'].str.replace('*', '', regex=False)

# Debugging: Check the structure of the new data
print("Data columns:", data.columns.tolist())
print("Data shape:", data.shape)
print("Unique categories:", data['category'].unique())
print("Sample of data:")
print(data.head())

# Create a unified identifier column
# Only use UNIT_NAME, name311, or Name_2 - exclude rows that don't have any of these
def create_unified_id(row):
    """Create a single identifier from UNIT_NAME, name311, or Name_2 only"""
    if pd.notna(row.get('UNIT_NAME')):
        return f"UNIT_{row['UNIT_NAME']}"
    elif pd.notna(row.get('name311')):
        return f"NAME311_{row['name311']}"
    elif pd.notna(row.get('Name_2')):
        return f"NAME2_{row['Name_2']}"
    else:
        # Return None for rows without any of the three main identifiers
        return None

# Add unified identifier
data['unified_id'] = data.apply(create_unified_id, axis=1)

# Filter out rows that don't have any of the three main identifiers
print(f"Original data shape: {data.shape}")
data_with_ids = data[data['unified_id'].notna()].copy()
print(f"Data shape after filtering for main identifiers only: {data_with_ids.shape}")

# Update data to only include rows with proper identifiers
data = data_with_ids

print(f"\nCreated unified IDs. Sample:")
print(data[['unified_id', 'UNIT_NAME', 'name311', 'Name_2']].head(10))
print(f"Unique unified IDs: {data['unified_id'].nunique()}")

# Identify the vegetation categories
vegetation_categories = ['low_vegetation', 'medium_vegetation', 'high_vegetation']

# Filter for vegetation categories
data_vegetation = data[data['category'].isin(vegetation_categories)]

# Debugging: Check data after filtering
print("\nVegetation data shape:", data_vegetation.shape)
if data_vegetation.empty:
    print("Error: No vegetation data found!")
    print("Available categories:", data['category'].unique())
else:
    print("Vegetation categories found:", data_vegetation['category'].unique())

# Determine the correct column names for pixel counts and areas
pixel_count_cols = [col for col in data.columns if 'pixel_count' in col]
area_cols = [col for col in data.columns if 'total_area_m2' in col]

print(f"\nPixel count columns: {pixel_count_cols}")
print(f"Area columns: {area_cols}")

if data_vegetation.empty:
    print("Error: No vegetation data could be processed!")
    print("Check that your LIDAR processing completed successfully and generated vegetation categories.")
else:
    # For each vegetation category, get the appropriate columns
    vegetation_data_combined = []

    for category in vegetation_categories:
        category_data = data_vegetation[data_vegetation['category'] == category].copy()
        if not category_data.empty:
            # Get the pixel count and area columns for this category
            pixel_col = f'{category}_pixel_count'
            area_col = f'{category}_total_area_m2'

            if pixel_col in category_data.columns and area_col in category_data.columns:
                # Rename columns to standard names for combining
                category_data['vegetation_pixel_count'] = category_data[pixel_col]
                category_data['vegetation_area_m2'] = category_data[area_col]
                vegetation_data_combined.append(category_data)

    if vegetation_data_combined:
        # Combine all vegetation data
        data_combined = pd.concat(vegetation_data_combined, ignore_index=True)

        print(f"\nCombined vegetation data shape: {data_combined.shape}")
        print(f"Unique parks/shapes before aggregation: {data_combined['unified_id'].nunique()}")

        # Group by unified_id and other relevant columns, then pivot
        # Only use columns that are essential for grouping to avoid NaN issues
        preserve_cols = ['unified_id', 'borough']
        
        print(f"Preserving columns: {preserve_cols}")
        print(f"Sample data for aggregation:")
        print(data_combined[preserve_cols + ['category', 'vegetation_pixel_count', 'vegetation_area_m2']].head())
        
        # Check for any NaN values that might cause issues
        print(f"NaN values in key columns:")
        for col in preserve_cols + ['category', 'vegetation_pixel_count', 'vegetation_area_m2']:
            nan_count = data_combined[col].isna().sum()
            print(f"  {col}: {nan_count} NaN values")

        # Aggregate by unified_id first to handle multiple rows per park/category combination
        # Handle NaN values by filling them with 0 for numeric columns
        data_for_agg = data_combined.copy()
        data_for_agg['vegetation_pixel_count'] = data_for_agg['vegetation_pixel_count'].fillna(0)
        data_for_agg['vegetation_area_m2'] = data_for_agg['vegetation_area_m2'].fillna(0)
        
        aggregated_data = data_for_agg.groupby(preserve_cols + ['category']).agg({
            'vegetation_pixel_count': 'sum',
            'vegetation_area_m2': 'sum'
        }).reset_index()

        print(f"After aggregation by unified_id and category: {aggregated_data.shape}")
        
        # Debug: Check the aggregated data
        print(f"Sample aggregated data:")
        print(aggregated_data.head())
        print(f"Sample vegetation values:")
        sample_rows = aggregated_data.head(3)
        for idx, row in sample_rows.iterrows():
            print(f"  {row['unified_id']}, {row['category']}: {row['vegetation_pixel_count']} pixels, {row['vegetation_area_m2']} m²")
        
        # Check if pixel counts and areas are unreasonably similar
        pixel_area_ratio = aggregated_data['vegetation_area_m2'] / aggregated_data['vegetation_pixel_count'].replace(0, 1)
        print(f"Pixel to area ratio stats:")
        print(f"  Mean ratio: {pixel_area_ratio.mean():.2f}")
        print(f"  Min ratio: {pixel_area_ratio.min():.2f}")
        print(f"  Max ratio: {pixel_area_ratio.max():.2f}")
        print(f"  Unique ratios: {pixel_area_ratio.nunique()}")
        
        # Check if areas are actually equal to pixel counts
        equal_values = (aggregated_data['vegetation_pixel_count'] == aggregated_data['vegetation_area_m2']).sum()
        print(f"Rows where pixel_count equals area_m2: {equal_values} out of {len(aggregated_data)}")
        
        if equal_values > len(aggregated_data) * 0.8:  # If more than 80% are equal
            print("WARNING: Pixel counts and areas are nearly identical - this suggests a data issue!")
            print("The original data might already be in area units, not pixel counts.")
            
            # Let's check the original column names and values
            print(f"Original vegetation area columns: {area_cols}")
            print(f"Original vegetation pixel columns: {pixel_count_cols}")
            
            # Sample the original data
            print("Sample from original data:")
            sample_original = data_vegetation.head(3)
            for idx, row in sample_original.iterrows():
                cat = row['category']
                pixel_col = f'{cat}_pixel_count'
                area_col = f'{cat}_total_area_m2'
                if pixel_col in row and area_col in row:
                    print(f"  {cat}: {row[pixel_col]} pixels, {row[area_col]} m²")

        # Now pivot to get all vegetation types for each unique park/shape
        park_vegetation = aggregated_data.pivot_table(
            index=preserve_cols,
            columns='category',
            values=['vegetation_pixel_count', 'vegetation_area_m2'],
            fill_value=0,
            aggfunc='sum'
        ).reset_index()
        
        print(f"Pivot successful. Shape: {park_vegetation.shape}")
        print(f"Columns after pivot: {park_vegetation.columns.tolist()}")

        # Flatten column names
        park_vegetation.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in park_vegetation.columns]

        print(f"After pivot: {park_vegetation.shape}")
        print("Column names after pivot:", park_vegetation.columns.tolist())

        # Calculate total vegetation for each park (sum across all vegetation types)
        vegetation_area_cols = [col for col in park_vegetation.columns if 'vegetation_area_m2' in col and any(veg in col for veg in vegetation_categories)]
        vegetation_pixel_cols = [col for col in park_vegetation.columns if 'vegetation_pixel_count' in col and any(veg in col for veg in vegetation_categories)]

        print(f"Found vegetation area columns: {vegetation_area_cols}")
        print(f"Found vegetation pixel columns: {vegetation_pixel_cols}")

        if vegetation_area_cols and vegetation_pixel_cols:
            park_vegetation['total_vegetation_area_m2'] = park_vegetation[vegetation_area_cols].sum(axis=1)
            park_vegetation['total_vegetation_pixels'] = park_vegetation[vegetation_pixel_cols].sum(axis=1)

            print(f"\nFinal park vegetation data shape: {park_vegetation.shape}")
            print("Sample of final data:")
            print(park_vegetation[['unified_id', 'borough', 'total_vegetation_area_m2', 'total_vegetation_pixels']].head())

            # Add diagnostic information about park sizes
            print(f"\nDIAGNOSTIC: Park size analysis")
            # Get geometry areas if available
            id_mapping = data_combined.groupby('unified_id').agg({
                'UNIT_NAME': 'first',
                'name311': 'first', 
                'Name_2': 'first',
                'park_name': 'first',
                'park_id': 'first',
                'unique_id': 'first',
                'geometry_area_m2': 'first'
            }).reset_index()
            
            # Merge geometry info for analysis
            park_with_geom = park_vegetation.merge(id_mapping[['unified_id', 'geometry_area_m2']], on='unified_id', how='left')
            
            if 'geometry_area_m2' in park_with_geom.columns:
                geom_areas = park_with_geom['geometry_area_m2'].dropna()
                if len(geom_areas) > 0:
                    print(f"Park geometry sizes:")
                    print(f"  Min: {geom_areas.min():,.0f} m² ({geom_areas.min()/10000:.1f} hectares)")
                    print(f"  Mean: {geom_areas.mean():,.0f} m² ({geom_areas.mean()/10000:.1f} hectares)")
                    print(f"  Median: {geom_areas.median():,.0f} m² ({geom_areas.median()/10000:.1f} hectares)")
                    print(f"  Max: {geom_areas.max():,.0f} m² ({geom_areas.max()/10000:.1f} hectares)")
                    
                    # Show size distribution
                    print(f"\nPark size distribution:")
                    print(f"  < 1 hectare: {(geom_areas < 10000).sum()}")
                    print(f"  1-10 hectares: {((geom_areas >= 10000) & (geom_areas < 100000)).sum()}")
                    print(f"  10-100 hectares: {((geom_areas >= 100000) & (geom_areas < 1000000)).sum()}")
                    print(f"  100-1000 hectares: {((geom_areas >= 1000000) & (geom_areas < 10000000)).sum()}")
                    print(f"  > 1000 hectares: {(geom_areas >= 10000000).sum()}")
                    
                    # Show largest parks
                    print(f"\nLargest 10 parks/shapes:")
                    largest_parks = park_with_geom.nlargest(10, 'geometry_area_m2')
                    for _, row in largest_parks.iterrows():
                        name = row.get('UNIT_NAME') or row.get('name311') or row.get('Name_2') or 'Unknown'
                        if len(name) > 50:
                            name = name[:47] + "..."
                        area_ha = row['geometry_area_m2'] / 10000
                        veg_ha = row['total_vegetation_area_m2'] / 10000
                        print(f"    {name}: {area_ha:.0f} ha total, {veg_ha:.0f} ha vegetation")
                    
                    # Flag suspiciously large parks
                    huge_parks = park_with_geom[park_with_geom['geometry_area_m2'] > 5000000]  # > 500 hectares
                    if len(huge_parks) > 0:
                        print(f"\n*** WARNING: Found {len(huge_parks)} parks larger than 500 hectares ***")
                        print("These might be administrative boundaries rather than individual parks:")
                        for _, row in huge_parks.iterrows():
                            name = row.get('UNIT_NAME') or row.get('name311') or row.get('Name_2') or 'Unknown'
                            area_ha = row['geometry_area_m2'] / 10000
                            print(f"    {name}: {area_ha:.0f} hectares")
                        print("Consider filtering out these large administrative areas if analyzing individual parks.")

            # Group by borough and calculate mean and standard error for vegetation per park
            summary = (
                park_vegetation.groupby(['borough'])
                .agg(
                    mean_pixel_count=('total_vegetation_pixels', 'mean'),
                    se_pixel_count=('total_vegetation_pixels', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0),
                    mean_area_m2=('total_vegetation_area_m2', 'mean'),
                    se_area_m2=('total_vegetation_area_m2', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0),
                    min_area=('total_vegetation_area_m2', 'min'),
                    max_area=('total_vegetation_area_m2', 'max'),
                    park_count=('total_vegetation_area_m2', 'count')  # Count of individual parks
                )
                .reset_index()
            )

            print("\nSummary by borough (per individual park/shape):")
            print(summary)
            print(f"Total parks/shapes across all boroughs: {summary['park_count'].sum()}")

            # Calculate the % of parks meeting vegetation thresholds for different radii
            def calculate_percentage(park_data, radius):
                required_area = (np.pi * (radius ** 2)) * 0.116  # 11.6% of the circle area
                print(f"\nRadius {radius} m - Required Area: {required_area:.2f} m²")

                # Count parks meeting the threshold
                parks_meeting_threshold = park_data[park_data['total_vegetation_area_m2'] >= required_area]

                print(f"Parks/shapes meeting threshold: {len(parks_meeting_threshold)} / {len(park_data)}")

                if len(park_data) > 0:
                    percentage = (len(parks_meeting_threshold) / len(park_data)) * 100
                else:
                    percentage = 0

                return percentage

            radii = [100, 250, 500, 1000]
            percentages = {radius: calculate_percentage(park_vegetation, radius) for radius in radii}

            # Save the summary to a text file
            output_file = '../outputs/summary_parks_habitat.txt'
            with open(output_file, 'w') as f:
                f.write("PARK/SHAPE VEGETATION ANALYSIS SUMMARY\n")
                f.write("(Each row represents one unique park/shape identified by UNIT_NAME, name311, or Name_2)\n")
                f.write("=" * 70 + "\n\n")

                for _, row in summary.iterrows():
                    f.write(f"Borough: {row['borough']}\n")
                    f.write(f"  Number of Parks/Shapes: {int(row['park_count'])}\n")
                    f.write(f"  Mean Vegetation Pixel Count per Park/Shape: {row['mean_pixel_count']:.2f}\n")
                    f.write(f"  SE Vegetation Pixel Count: {row['se_pixel_count']:.2f}\n")
                    f.write(f"  Mean Vegetation Area per Park/Shape (m²): {row['mean_area_m2']:.2f}\n")
                    f.write(f"  SE Vegetation Area (m²): {row['se_area_m2']:.2f}\n")
                    f.write(f"  Min Vegetation Area (m²): {row['min_area']:.2f}\n")
                    f.write(f"  Max Vegetation Area (m²): {row['max_area']:.2f}\n\n")

                f.write("VEGETATION THRESHOLD ANALYSIS\n")
                f.write("Percentage of parks/shapes with sufficient vegetation coverage:\n")
                f.write("(Based on 11.6% vegetation within circular buffer)\n\n")

                for radius, percentage in percentages.items():
                    required_area = (np.pi * (radius ** 2)) * 0.116
                    f.write(f"  Radius {radius} m (≥{required_area:.0f} m² vegetation): {percentage:.2f}%\n")

                f.write(f"\nTOTAL DATASET SUMMARY:\n")
                f.write(f"Original dataset rows: {len(data)}\n")
                f.write(f"Vegetation rows processed: {len(data_combined)}\n")
                f.write(f"Total individual parks/shapes analyzed: {len(park_vegetation)}\n")
                f.write(f"Total vegetation area across all parks/shapes: {park_vegetation['total_vegetation_area_m2'].sum():.2f} m²\n")
                f.write(f"Average vegetation area per park/shape: {park_vegetation['total_vegetation_area_m2'].mean():.2f} m²\n")
                f.write(f"Data reduction ratio: {len(data_combined)/len(park_vegetation):.1f} rows per park/shape\n")

                # Add identifier breakdown
                f.write(f"\nIDENTIFIER BREAKDOWN:\n")
                id_counts = park_vegetation['unified_id'].apply(lambda x: x.split('_')[0]).value_counts()
                for id_type, count in id_counts.items():
                    f.write(f"  {id_type}: {count} parks/shapes\n")

            print(f"\nSummary saved to {output_file}")

            # Also save detailed park data with additional identifier columns
            # Merge back the identifier information
            park_vegetation_detailed = park_vegetation.merge(id_mapping, on='unified_id', how='left')
            
            park_output_file = '../outputs/detailed_park_vegetation.csv'
            park_vegetation_detailed.to_csv(park_output_file, index=False)
            print(f"Detailed park/shape data saved to {park_output_file}")

            # Print some additional insights
            print(f"\nADDITIONAL INSIGHTS:")
            print(f"Total original rows in dataset: {len(data)}")
            print(f"Total vegetation rows processed: {len(data_combined)}")
            print(f"Total individual parks/shapes: {len(park_vegetation)}")
            print(f"Parks/shapes with any vegetation: {(park_vegetation['total_vegetation_area_m2'] > 0).sum()}")
            print(f"Parks/shapes with no vegetation: {(park_vegetation['total_vegetation_area_m2'] == 0).sum()}")

            # Show breakdown by identifier type
            print(f"\nBreakdown by identifier type:")
            id_counts = park_vegetation['unified_id'].apply(lambda x: x.split('_')[0]).value_counts()
            for id_type, count in id_counts.items():
                print(f"  {id_type}: {count} parks/shapes")
            
            print(f"\nData processing summary:")
            print(f"  Original dataset rows: {len(data)}")
            print(f"  Rows after vegetation filtering: {len(data_vegetation)}")
            print(f"  Rows in combined vegetation data: {len(data_combined)}")
            print(f"  Final unique parks/shapes: {len(park_vegetation)}")
            print(f"  Data reduction ratio: {len(data_combined)/len(park_vegetation):.1f} rows per park/shape")

            # Show average vegetation by type
            if vegetation_area_cols:
                print(f"\nAverage vegetation area by type:")
                for col in vegetation_area_cols:
                    avg_area = park_vegetation[col].mean()
                    print(f"  {col}: {avg_area:.2f} m²")

        else:
            print("Error: Could not find vegetation area or pixel count columns after pivot!")
            print("Available columns:", park_vegetation.columns.tolist())
    else:
        print("Error: No vegetation data could be combined!")
        print("Check that the expected column names exist in your data.")

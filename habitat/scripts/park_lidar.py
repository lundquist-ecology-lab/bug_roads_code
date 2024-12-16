import rasterio
import geopandas as gpd
import numpy as np
import os
from pathlib import Path
import pandas as pd
from shapely.geometry import mapping
from rasterio.crs import CRS

def check_and_fix_projection(tiff_path):
    """
    Check if TIFF has correct projection (EPSG:2263), if not, create new file and delete original.
    Returns the path to the file with correct projection.
    """
    print(f"Checking projection for: {tiff_path}")
    with rasterio.open(tiff_path) as src:
        if src.crs is None or src.crs != CRS.from_epsg(2263):
            print(f"Incorrect projection found for {tiff_path}. Creating new file with EPSG:2263...")
            
            # Create temporary file with correct projection
            temp_path = tiff_path.with_suffix(".temp.tif")
            profile = src.profile.copy()
            profile.update(crs=CRS.from_epsg(2263))
            
            # Read data and write to temporary file
            data = src.read()
            with rasterio.open(temp_path, 'w', **profile) as temp_src:
                temp_src.write(data)

    # Close the source file before attempting any file operations
    src.close()
    
    if 'temp_path' in locals():
        # Remove original file
        print(f"Removing original file: {tiff_path}")
        os.remove(tiff_path)
        
        # Rename temporary file to original name
        print(f"Renaming temporary file to: {tiff_path}")
        os.rename(temp_path, tiff_path)
        
        print("Projection correction completed")
    else:
        print("Projection is correct, no changes needed")
    
    return tiff_path

def find_tiff_files(base_dir, borough):
    """
    Find all relevant TIFF files for a borough.
    """
    borough_clean = borough.replace(" ", "_").lower()
    categories = ['building', 'ground', 'low_vegetation', 'medium_vegetation', 'high_vegetation']
    
    tiff_files = []
    for category in categories:
        pattern = f"{borough_clean}_{category}_merged.tif"
        path = base_dir / pattern
        if path.exists():
            tiff_files.append(path)
    
    return tiff_files

def transform_geometries(gdf, raster_crs):
    """
    Transform geometries to match the CRS of the raster.
    """
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
    return gdf

def get_borough_boundaries(borough_gpkg_path):
    """
    Load borough boundaries from GeoPackage.
    """
    return gpd.read_file(borough_gpkg_path)

def ensure_majority_borough_assignment(shapes_gdf):
    """
    Ensure all point_id with the same unique_id are in the same borough.
    Assign the borough based on the majority vote of all point_id under the same unique_id.
    """
    majority_borough = (
        shapes_gdf.groupby('unique_id')['boro_name']
        .apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        .reset_index()
        .rename(columns={'boro_name': 'majority_borough'})
    )
    
    # Merge majority borough back to the original GeoDataFrame
    shapes_gdf = shapes_gdf.merge(majority_borough, on='unique_id', how='left')
    shapes_gdf['boro_name'] = shapes_gdf['majority_borough']
    shapes_gdf.drop(columns=['majority_borough'], inplace=True)
    
    return shapes_gdf

def process_single_tiff(tiff_path, shapes_gpkg_path, borough_boundaries_gdf, borough_name, output_dir):
    """
    Process a single LIDAR TIFF file and save results immediately.
    """
    category = next((cat for cat in ['building', 'ground', 'low_vegetation', 'medium_vegetation', 'high_vegetation'] 
                    if cat in str(tiff_path).lower()), None)
    if not category:
        print(f"Could not determine category for {tiff_path}")
        return

    print(f"\nProcessing {borough_name} - {category}")
    
    # Check and fix projection before processing
    tiff_path = check_and_fix_projection(tiff_path)
    
    # Read and filter shapes
    shapes = gpd.read_file(shapes_gpkg_path)
    shapes = filter_shapes_by_borough(shapes, borough_boundaries_gdf, borough_name)
    
    if len(shapes) == 0:
        print(f"No shapes found in {borough_name}")
        return

    # Transform geometries to match raster CRS
    with rasterio.open(tiff_path) as src:
        shapes = transform_geometries(shapes, src.crs)
        print("Transformed shapes to match raster CRS")
    
    # Assign area directly from geometry
    shapes['area_m2'] = shapes.geometry.area
    
    # Ensure all point_id with the same unique_id are in the same borough
    shapes = ensure_majority_borough_assignment(shapes)

    # Save processed results
    save_results(shapes, output_dir, borough_name, category)

def save_results(results_gdf, output_dir, borough_name, category):
    """
    Save processed results to a CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{borough_name.lower().replace(' ', '_')}_{category}_{timestamp}.csv"
    output_path = os.path.join(output_dir, filename)
    
    results_gdf.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")

def filter_shapes_by_borough(shapes_gdf, borough_gdf, borough_name):
    """
    Filter shapes by borough.
    """
    if shapes_gdf.crs != borough_gdf.crs:
        print("Converting borough boundaries to shape CRS")
        borough_gdf = borough_gdf.to_crs(shapes_gdf.crs)
    
    shapes_in_borough = gpd.sjoin(shapes_gdf, 
                                 borough_gdf[borough_gdf['boro_name'] == borough_name],
                                 how='inner',
                                 predicate='intersects')
    
    return shapes_in_borough

def combine_results(output_dir):
    """
    Combine all CSV files in the output directory into a single summary file.
    """
    all_files = list(Path(output_dir).glob("*.csv"))
    if not all_files:
        print("No result files found to combine")
        return

    print(f"Combining {len(all_files)} result files...")
    dfs = []
    for file in all_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    summary_path = os.path.join(output_dir, "combined_results_summary.csv")
    combined_df.to_csv(summary_path, index=False)
    
    print("\nSummary by borough and category:")
    summary = combined_df.groupby(['borough', 'category']).agg({
        'area_m2': 'sum'
    })
    print(summary)
    
    return summary_path

def main():
    """
    Main function to process lidar data for all boroughs.
    """
    base_dir = Path(r"E:\NYC_LIDAR_Merged_TIFF")
    borough_boundaries_path = "../../data/borough_boundaries_26918.gpkg"
    shapes_gpkg_path = "../../data/open_and_park_single_26918.gpkg"
    output_dir = "../outputs/lidar_results"
    
    borough_boundaries_gdf = get_borough_boundaries(borough_boundaries_path)
    boroughs = borough_boundaries_gdf['boro_name'].tolist()
    print(f"Processing boroughs: {', '.join(boroughs)}")
    
    for borough in boroughs:
        print(f"\nProcessing {borough}...")
        
        tiff_files = find_tiff_files(base_dir, borough)
        if not tiff_files:
            print(f"Warning: No TIFF files found for {borough}")
            continue
            
        print(f"Found {len(tiff_files)} TIFF files for {borough}")
        
        for tiff_file in tiff_files:
            print(f"\nProcessing file: {tiff_file}")
            process_single_tiff(
                tiff_file,
                shapes_gpkg_path,
                borough_boundaries_gdf,
                borough,
                output_dir
            )
            print(f"Completed processing: {tiff_file}")
    
    # Combine all results at the end
    summary_path = combine_results(output_dir)
    if summary_path:
        print(f"\nFinal combined results saved to {summary_path}")
    
    print("\nScript completed!")

if __name__ == "__main__":
    main()

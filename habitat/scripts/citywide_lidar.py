import rasterio
import geopandas as gpd
import numpy as np
from rasterio.mask import mask
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

def find_tiff_files(base_dir, borough):
    """
    Find all relevant TIFF files for a borough.
    """
    borough_clean = borough.replace(" ", "_")
    categories = ['building', 'ground', 'low_vegetation', 'medium_vegetation', 'high_vegetation']
    
    tiff_files = []
    for category in categories:
        pattern = f"{borough_clean}_{category}_merged.tif"
        path = base_dir / pattern
        if path.exists():
            tiff_files.append(path)
    
    return tiff_files

def process_borough_tiff(tiff_path, borough_boundaries_gdf, borough_name, output_dir, pixel_size=0.0833):
    """
    Process a single LIDAR TIFF file for a borough and save results immediately.
    """
    category = next((cat for cat in ['building', 'ground', 'low_vegetation', 'medium_vegetation', 'high_vegetation']
                     if cat in str(tiff_path).lower()), None)
    if not category:
        print(f"Could not determine category for {tiff_path}")
        return

    print(f"\nProcessing {borough_name} - {category}")
    
    # Check and fix projection before processing
    tiff_path = check_and_fix_projection(tiff_path)
    
    borough = borough_boundaries_gdf[borough_boundaries_gdf['boro_name'] == borough_name]
    if borough.empty:
        print(f"No boundary found for borough: {borough_name}")
        return

    results = []
    
    with rasterio.open(tiff_path) as src:
        borough = transform_geometries(borough, src.crs)
        print("Transformed borough boundary to match raster CRS")
        
        geom = [mapping(borough.geometry.unary_union)]
        try:
            out_image, out_transform = mask(src, geom, crop=True, nodata=0, all_touched=True)
            binary_image = (out_image != 0).astype(np.uint8)
            pixel_count = np.sum(binary_image)
            
            if pixel_count > 0:
                area_m2 = pixel_count * (pixel_size ** 2)
                results.append({
                    'borough': borough_name,
                    'category': category,
                    'pixel_count': int(pixel_count),
                    'area_m2': float(area_m2)
                })
        except ValueError as ve:
            if "Input shapes do not overlap raster" in str(ve):
                print(f"No overlap between raster and borough: {borough_name}")
            else:
                raise ve
        except Exception as e:
            print(f"Error processing borough {borough_name}: {str(e)}")
    
    # Save results
    if results:
        save_batch_results(results, output_dir, borough_name, category)

def save_batch_results(results, output_dir, borough_name, category):
    """
    Save a batch of results to a CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{borough_name.lower().replace(' ', '_')}_{category}_{timestamp}.csv"
    output_path = os.path.join(output_dir, filename)
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Saved batch results to {output_path}")

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
        'pixel_count': 'sum',
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
    output_dir = "../outputs/citywide_lidar_results"
    
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
            process_borough_tiff(
                tiff_file,
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

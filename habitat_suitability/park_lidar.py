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
    Check if TIFF has correct projection, if not, create new file and delete original.
    Returns the path to the file with correct projection.
    """
    print(f"Checking projection for: {tiff_path}")
    try:
        with rasterio.open(tiff_path) as src:
            print(f"Current raster CRS: {src.crs}")

            # Check if we can read the CRS at all
            if src.crs is None:
                print("Warning: Raster has no CRS defined")
                return tiff_path
            else:
                print(f"Raster CRS is defined: {src.crs}")
                # Continue without changing projection
                return tiff_path

    except Exception as e:
        print(f"Error reading raster CRS: {e}")
        print("Continuing without projection check...")
        return tiff_path

def transform_geometries(gdf, raster_crs):
    """
    Transform geometries to match the CRS of the raster.
    """
    print(f"Input GDF CRS: {gdf.crs}")
    print(f"Target raster CRS: {raster_crs}")

    if gdf.crs != raster_crs:
        print("Transforming geometries to match raster CRS...")
        try:
            gdf = gdf.to_crs(raster_crs)
            print("✓ CRS transformation successful")
        except Exception as e:
            print(f"✗ CRS transformation failed: {e}")
            print("Continuing without transformation - results may be inaccurate")
    else:
        print("CRS already matches, no transformation needed")

    return gdf

def get_borough_boundaries(borough_gpkg_path):
    """
    Load borough boundaries from GeoPackage.
    """
    return gpd.read_file(borough_gpkg_path)

def assign_parks_to_boroughs(shapes_gdf, borough_gdf):
    """
    Assign each park to exactly one borough based on centroid or largest overlap.
    This prevents duplicates from spatial joins.
    """
    if shapes_gdf.crs != borough_gdf.crs:
        print("Converting borough boundaries to shape CRS")
        borough_gdf = borough_gdf.to_crs(shapes_gdf.crs)
    
    # Calculate centroids for park assignment
    shapes_with_centroids = shapes_gdf.copy()
    shapes_with_centroids['centroid'] = shapes_with_centroids.geometry.centroid
    
    # Spatial join using centroids (each park gets exactly one borough)
    parks_with_borough = gpd.sjoin(
        shapes_with_centroids.set_geometry('centroid'),
        borough_gdf[['boro_name', 'geometry']], 
        how='left',
        predicate='within'
    )
    
    # Restore original geometry
    parks_with_borough = parks_with_borough.set_geometry('geometry')
    parks_with_borough = parks_with_borough.drop(columns=['centroid'])
    
    # For parks whose centroids don't fall within any borough, 
    # assign based on largest intersection
    unassigned = parks_with_borough[parks_with_borough['boro_name'].isna()].copy()
    
    if len(unassigned) > 0:
        print(f"Assigning {len(unassigned)} parks based on largest intersection...")
        
        for idx, park in unassigned.iterrows():
            max_area = 0
            assigned_borough = None
            
            for _, borough in borough_gdf.iterrows():
                try:
                    intersection = park.geometry.intersection(borough.geometry)
                    if intersection.area > max_area:
                        max_area = intersection.area
                        assigned_borough = borough['boro_name']
                except:
                    continue
            
            if assigned_borough:
                parks_with_borough.loc[idx, 'boro_name'] = assigned_borough
    
    # Remove any parks that still couldn't be assigned
    parks_with_borough = parks_with_borough[parks_with_borough['boro_name'].notna()].copy()
    
    return parks_with_borough

def filter_shapes_by_borough(shapes_gdf, borough_name):
    """
    Filter shapes that have already been assigned to this borough.
    """
    return shapes_gdf[shapes_gdf['boro_name'] == borough_name].copy()

def find_tiff_files(base_dir, borough):
    """
    Find all relevant TIFF files for a borough.
    """
    borough_clean = borough.replace(" ", "_")
    categories = ['building', 'ground', 'low_vegetation', 'medium_vegetation', 'high_vegetation', 'water']

    tiff_files = []
    for category in categories:
        pattern = f"{borough_clean}_{category}_merged.tif"
        path = base_dir / pattern
        if path.exists():
            tiff_files.append(path)

    return tiff_files

def process_parks_with_raster(tiff_path, parks_with_boroughs, borough_name, output_dir):
    """
    Process all parks in a borough with a single LIDAR raster file using high-memory approach.
    """
    category = next((cat for cat in ['building', 'ground', 'low_vegetation', 'medium_vegetation', 'high_vegetation', 'water']
                     if cat in str(tiff_path).lower()), None)
    if not category:
        print(f"Could not determine category for {tiff_path}")
        return

    print(f"\nProcessing {borough_name} - {category}")

    # Check and fix projection before processing
    tiff_path = check_and_fix_projection(tiff_path)

    # Filter shapes for this borough
    shapes = filter_shapes_by_borough(parks_with_boroughs, borough_name)

    if len(shapes) == 0:
        print(f"No shapes found in {borough_name}")
        return

    results = []

    with rasterio.open(tiff_path) as src:
        # Transform geometries to match raster CRS
        shapes = transform_geometries(shapes, src.crs)
        print(f"Raster dimensions: {src.width} x {src.height}, Resolution: {src.res}")
        print(f"Processing {len(shapes)} parks for {category} in {borough_name}")

        # Get pixel area
        pixel_area_m2 = 1.0 / 12.0  # 12 pixels per m² = 0.0833 m² per pixel
        print(f"Pixel area: {pixel_area_m2} m²")

        successful_parks = 0

        for idx, row in shapes.iterrows():
            if idx % 50 == 0:  # Progress indicator every 50 parks
                print(f"  Processed {len(results)}/{len(shapes)} parks...")

            try:
                # Create geometry for masking
                geom = [mapping(row.geometry)]

                # Mask the raster with this park's geometry
                out_image, out_transform = mask(src, geom, crop=True, nodata=0, all_touched=True)

                # Count non-zero pixels
                pixel_count = np.count_nonzero(out_image)
                total_area_m2 = pixel_count * pixel_area_m2

                # Create a unique identifier for this specific park
                unique_park_id = f"{row.get('unique_id', idx)}_{category}_{borough_name}"

                # Store results for this park
                park_result = {
                    'park_id': row.get('unique_id', idx),
                    'park_name': row.get('name', f'Park_{idx}'),
                    'borough': borough_name,
                    'category': category,
                    f'{category}_pixel_count': int(pixel_count),
                    f'{category}_total_area_m2': float(total_area_m2),
                    'geometry_area_m2': float(row.geometry.area),
                    'processing_id': unique_park_id  # For debugging duplicates
                }

                # Add all original columns (but be careful not to overwrite key columns)
                for col in row.index:
                    if col not in park_result and col != 'geometry' and not col.startswith('index_'):
                        park_result[col] = row[col]

                results.append(park_result)

                if pixel_count > 0:
                    successful_parks += 1

            except ValueError as ve:
                if "Input shapes do not overlap raster" in str(ve):
                    # Park doesn't overlap with raster - add zero result
                    unique_park_id = f"{row.get('unique_id', idx)}_{category}_{borough_name}"
                    park_result = {
                        'park_id': row.get('unique_id', idx),
                        'park_name': row.get('name', f'Park_{idx}'),
                        'borough': borough_name,
                        'category': category,
                        f'{category}_pixel_count': 0,
                        f'{category}_total_area_m2': 0.0,
                        'geometry_area_m2': float(row.geometry.area),
                        'processing_id': unique_park_id
                    }
                    
                    # Add original columns
                    for col in row.index:
                        if col not in park_result and col != 'geometry' and not col.startswith('index_'):
                            park_result[col] = row[col]
                    
                    results.append(park_result)
                else:
                    print(f"    Error processing park {idx}: {ve}")
            except Exception as e:
                print(f"    Error processing park {idx}: {e}")
                # Add zero result for failed parks
                unique_park_id = f"{row.get('unique_id', idx)}_{category}_{borough_name}"
                park_result = {
                    'park_id': row.get('unique_id', idx),
                    'park_name': row.get('name', f'Park_{idx}'),
                    'borough': borough_name,
                    'category': category,
                    f'{category}_pixel_count': 0,
                    f'{category}_total_area_m2': 0.0,
                    'geometry_area_m2': float(row.geometry.area),
                    'processing_id': unique_park_id
                }
                
                # Add original columns
                for col in row.index:
                    if col not in park_result and col != 'geometry' and not col.startswith('index_'):
                        park_result[col] = row[col]
                
                results.append(park_result)

        print(f"  Total parks with {category} pixels: {successful_parks}/{len(shapes)}")

    if results:
        results_df = pd.DataFrame(results)
        save_results(results_df, output_dir, borough_name, category)
        return len(results)

    return 0

def save_results(results_df, output_dir, borough_name, category):
    """
    Save processed results to a CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{borough_name.lower().replace(' ', '_')}_{category}_{timestamp}.csv"
    output_path = os.path.join(output_dir, filename)

    results_df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")

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
        if file.name != "combined_results_summary.csv":  # Skip existing combined file
            df = pd.read_csv(file)
            dfs.append(df)

    if not dfs:
        print("No new files to combine")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Check for and remove duplicates
    print(f"Before duplicate removal: {len(combined_df)} rows")
    if 'processing_id' in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=['processing_id'])
        print(f"After duplicate removal: {len(combined_df)} rows")
    else:
        # Fallback duplicate removal
        id_cols = ['park_id', 'borough', 'category']
        available_cols = [col for col in id_cols if col in combined_df.columns]
        if available_cols:
            combined_df = combined_df.drop_duplicates(subset=available_cols)
            print(f"After duplicate removal: {len(combined_df)} rows")
    
    summary_path = os.path.join(output_dir, "combined_results_summary.csv")
    combined_df.to_csv(summary_path, index=False)

    print("\nData summary:")
    print(f"Total rows: {len(combined_df)}")
    print(f"Unique parks: {combined_df['park_id'].nunique() if 'park_id' in combined_df.columns else 'Unknown'}")
    print(f"Categories: {combined_df['category'].unique() if 'category' in combined_df.columns else 'Unknown'}")
    print(f"Boroughs: {combined_df['borough'].unique() if 'borough' in combined_df.columns else 'Unknown'}")

    return summary_path

def main():
    """
    Main function to process lidar data for all boroughs using high-memory approach.
    """
    base_dir = Path("../../../lidar")
    borough_boundaries_path = "../data/borough_boundaries_26918.gpkg"
    shapes_gpkg_path = "../data/open_and_park_single_26918.gpkg"
    output_dir = "../outputs/park_lidar_results"

    # Load data once
    print("Loading park shapes and borough boundaries...")
    borough_boundaries_gdf = get_borough_boundaries(borough_boundaries_path)
    shapes = gpd.read_file(shapes_gpkg_path)
    
    print(f"Original park shapes: {len(shapes)}")
    
    # Assign each park to exactly one borough
    print("Assigning parks to boroughs...")
    parks_with_boroughs = assign_parks_to_boroughs(shapes, borough_boundaries_gdf)
    
    print(f"Parks after borough assignment: {len(parks_with_boroughs)}")
    print("Parks per borough:")
    borough_counts = parks_with_boroughs['boro_name'].value_counts()
    for borough, count in borough_counts.items():
        print(f"  {borough}: {count} parks")

    boroughs = borough_boundaries_gdf['boro_name'].tolist()
    print(f"\nProcessing boroughs: {', '.join(boroughs)}")

    total_files_processed = 0

    for borough in boroughs:
        print(f"\n{'='*50}")
        print(f"Processing {borough}...")

        tiff_files = find_tiff_files(base_dir, borough)
        if not tiff_files:
            print(f"Warning: No TIFF files found for {borough}")
            continue

        print(f"Found {len(tiff_files)} TIFF files for {borough}")

        for tiff_file in tiff_files:
            print(f"\nProcessing file: {tiff_file}")
            parks_processed = process_parks_with_raster(
                tiff_file,
                parks_with_boroughs,
                borough,
                output_dir
            )
            if parks_processed > 0:
                total_files_processed += 1
                print(f"Completed processing: {tiff_file} ({parks_processed} parks)")

    # Combine all results at the end
    print(f"\n{'='*50}")
    print("Combining results...")
    summary_path = combine_results(output_dir)
    if summary_path:
        print(f"\nFinal combined results saved to {summary_path}")

    print(f"\nScript completed! Processed {total_files_processed} raster files.")

if __name__ == "__main__":
    main()

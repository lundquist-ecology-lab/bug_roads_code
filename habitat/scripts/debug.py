import rasterio
import geopandas as gpd
import numpy as np
from rasterio.mask import mask
from shapely.geometry import mapping
from rasterio.crs import CRS

def transform_geometries(gdf, raster_crs):
    """
    Transform geometries to match the CRS of the raster.
    """
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
    return gdf

def process_shape(shape, src, pixel_size):
    """
    Process a single shape and calculate pixel count and area.
    """
    try:
        geom = [mapping(shape.geometry)]
        out_image, out_transform = mask(src, geom, crop=True, nodata=0, all_touched=True)
        binary_image = (out_image != 0).astype(np.uint8)
        pixel_count = np.sum(binary_image)
        
        if pixel_count > 0:
            area_m2 = pixel_count * (pixel_size ** 2)
            return pixel_count, area_m2
    except ValueError as ve:
        if "Input shapes do not overlap raster" in str(ve):
            print("Shape does not overlap with raster")
    except Exception as e:
        print(f"Error processing shape: {str(e)}")
    
    return None, None

def main():
    # Paths
    raster_path = r"E:\NYC_LIDAR_Merged_TIFF\Bronx_building_merged.tif"
    shapes_gpkg_path = "../../data/open_and_park_single_26918.gpkg"
    
    # Pixel size in feet
    pixel_size = 0.0833 * 3.28084  # Convert meters to feet

    # Load shapes
    shapes = gpd.read_file(shapes_gpkg_path)
    print(f"Original shapes CRS: {shapes.crs}")
    
    # Test for Bronx shapes
    shapes = shapes[shapes['borough'] == 'X']  # Filter for Bronx
    print(f"Found {len(shapes)} shapes in Bronx")

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            print("No CRS found for raster. Assigning EPSG:2263.")
            src_crs = CRS.from_epsg(2263)
        else:
            src_crs = src.crs
        
        # Transform shapes to match raster CRS
        shapes = transform_geometries(shapes, src_crs)
        print("Transformed shapes to match raster CRS")

        results = []
        for idx, shape in shapes.iterrows():
            pixel_count, area_m2 = process_shape(shape, src, pixel_size)
            if pixel_count is not None and pixel_count > 0:
                results.append({
                    'point_id': shape['point_id'],
                    'unique_id': shape['unique_id'],
                    'pixel_count': int(pixel_count),
                    'area_m2': float(area_m2)
                })
                print(f"Processed shape {shape['point_id']} with {pixel_count} pixels and {area_m2:.2f} m² area")

    print(f"\nTotal valid shapes processed: {len(results)}")
    if results:
        print("Sample result:", results[0])

if __name__ == "__main__":
    main()

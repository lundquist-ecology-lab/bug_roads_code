import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point, MultiPoint
from shapely.ops import nearest_points, split
from scipy.stats import sem
from tqdm import tqdm
import time
from itertools import combinations
import logging
from pathlib import Path

def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/logs/park_analysis.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_gdf(file_path: str, target_crs: int = 26918) -> gpd.GeoDataFrame:
    """Load and transform a GeoPackage file to the specified CRS."""
    try:
        gdf = gpd.read_file(file_path)
        return gdf.to_crs(epsg=target_crs)
    except Exception as e:
        raise RuntimeError(f"Failed to load {file_path}: {str(e)}")

def prepare_buildings_data(buildings_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Prepare buildings data by converting heights from feet to meters."""
    if 'heightroof' not in buildings_gdf.columns:
        raise ValueError("Required column 'heightroof' not found")
    
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf['heightroof_m'] = buildings_gdf['heightroof'] * 0.3048
    return buildings_gdf

def get_nearest_boundary_points(geom1, geom2):
    """
    Find the nearest points between two geometries' boundaries.
    Returns tuple of (Point on geom1 boundary, Point on geom2 boundary)
    """
    boundary1 = geom1.boundary
    boundary2 = geom2.boundary
    
    point1, point2 = nearest_points(boundary1, boundary2)
    return point1, point2

def get_close_park_pairs(parks_gdf: gpd.GeoDataFrame, max_distance_m: float = 1000) -> list:
    """Find pairs of parks within the maximum distance using boundary points."""
    close_pairs = []
    total_pairs = len(parks_gdf) * (len(parks_gdf) - 1) // 2
    
    with tqdm(total=total_pairs, desc="Finding close park pairs") as pbar:
        for (i, park1), (j, park2) in combinations(parks_gdf.iterrows(), 2):
            point1, point2 = get_nearest_boundary_points(park1.geometry, park2.geometry)
            
            distance_m = point1.distance(point2)
            if distance_m <= max_distance_m:
                close_pairs.append((park1, park2, point1, point2, distance_m))
            
            pbar.update(1)
    
    return close_pairs

def get_intersection_points(line: LineString, building_geometry) -> list:
    """Get the points where a line intersects with a building's boundary."""
    if line.intersects(building_geometry):
        intersection = line.intersection(building_geometry)
        if intersection.geom_type == 'Point':
            return [intersection]
        elif intersection.geom_type == 'MultiPoint':
            return list(intersection.geoms)
        elif intersection.geom_type == 'LineString':
            return [Point(intersection.coords[0]), Point(intersection.coords[-1])]
    return []

def analyze_close_park_pairs(close_pairs: list, buildings_gdf: gpd.GeoDataFrame):
    """Analyze all park pairs and create visualization data."""
    paths_data = []
    intersections_data = []
    building_sindex = buildings_gdf.sindex
    
    with tqdm(total=len(close_pairs), desc="Processing park pairs") as pbar:
        for park1, park2, point1, point2, distance_m in close_pairs:
            line = LineString([point1, point2])
            
            possible_matches_index = list(building_sindex.intersection(line.bounds))
            possible_matches = buildings_gdf.iloc[possible_matches_index]
            intersected_buildings = possible_matches[possible_matches.geometry.intersects(line)]
            
            # Collect all intersection points
            all_intersection_points = []
            heights = []
            
            for building in intersected_buildings.itertuples():
                intersection_points = get_intersection_points(line, building.geometry)
                for point in intersection_points:
                    all_intersection_points.append({
                        'geometry': point,
                        'height': building.heightroof_m,
                        'building_id': building.Index
                    })
                heights.append(building.heightroof_m)
            
            # Calculate statistics
            avg_height = 0.0
            std_error = 0.0
            if heights:
                valid_heights = [h for h in heights if not np.isnan(h)]
                if valid_heights:
                    avg_height = np.mean(valid_heights)
                    if len(valid_heights) > 1:
                        std_error = sem(valid_heights)
            
            # Store path data
            path_info = {
                'geometry': line,
                'Park1_point_id': park1['point_id'] if 'point_id' in park1 else None,
                'Park1_unique_id': park1['unique_id'] if 'unique_id' in park1 else None,
                'Park2_point_id': park2['point_id'] if 'point_id' in park2 else None,
                'Park2_unique_id': park2['unique_id'] if 'unique_id' in park2 else None,
                'Distance_m': distance_m,
                'Count_Intersected_Buildings': len(heights),
                'Average_heightroof_m': avg_height,
                'Standard_Error_heightroof_m': std_error,
                'Is_Zero_Building_Path': len(heights) == 0
            }
            paths_data.append(path_info)
            
            # Store intersection points data
            for point_info in all_intersection_points:
                intersection_info = {
                    'geometry': point_info['geometry'],
                    'height': point_info['height'],
                    'building_id': point_info['building_id'],
                    'Park1_point_id': park1['point_id'] if 'point_id' in park1 else None,
                    'Park2_point_id': park2['point_id'] if 'point_id' in park2 else None,
                    'Distance_m': distance_m
                }
                intersections_data.append(intersection_info)
            
            pbar.update(1)
    
    # Create GeoDataFrames
    paths_gdf = gpd.GeoDataFrame(paths_data, crs=buildings_gdf.crs)
    intersections_gdf = gpd.GeoDataFrame(intersections_data, crs=buildings_gdf.crs)
    
    return paths_gdf, intersections_gdf

def main():
    logger = setup_logging()
    start_time = time.time()
    
    try:
        # Load data
        parks_gdf = load_gdf("../../data/open_and_park_single_26918.gpkg")
        buildings_gdf = load_gdf("../../data/new_york_city_buildings_26918.gpkg")
        buildings_gdf = prepare_buildings_data(buildings_gdf)
        
        # Find close park pairs using boundary points
        close_pairs = get_close_park_pairs(parks_gdf, max_distance_m=1000)
        
        # Process close pairs and get visualization data
        paths_gdf, intersections_gdf = analyze_close_park_pairs(close_pairs, buildings_gdf)
        
        # Save to GeoPackage
        output_gpkg = Path('../outputs/park_analysis_visualization.gpkg')
        paths_gdf.to_file(output_gpkg, layer='paths', driver='GPKG')
        intersections_gdf.to_file(output_gpkg, layer='intersections', driver='GPKG')
        
        # Also save buildings for reference
        buildings_gdf.to_file(output_gpkg, layer='buildings', driver='GPKG')
        parks_gdf.to_file(output_gpkg, layer='parks', driver='GPKG')
        
        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Results saved to {output_gpkg}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
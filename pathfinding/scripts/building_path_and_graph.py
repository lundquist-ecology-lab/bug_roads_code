import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from scipy.stats import sem
from tqdm import tqdm
import logging
import time
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def load_gdf(file_path: str, layer=None, target_crs: int = 26918) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(file_path, layer=layer) if layer else gpd.read_file(file_path)
    return gdf.to_crs(epsg=target_crs)

def prepare_buildings_data(buildings_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf['heightroof_m'] = buildings_gdf['heightroof'] * 0.3048
    return buildings_gdf

def get_intersection_points(line: LineString, building_geometry):
    """Get intersection points between a line and building geometry."""
    if line.intersects(building_geometry):
        intersection = line.intersection(building_geometry)
        if intersection.geom_type == 'Point':
            return [intersection]
        elif intersection.geom_type == 'MultiPoint':
            return list(intersection.geoms)
        elif intersection.geom_type == 'LineString':
            # For LineString intersection, return the midpoint
            return [Point(intersection.interpolate(0.5, normalized=True))]
        elif intersection.geom_type == 'MultiLineString':
            # For MultiLineString, return midpoint of each line
            points = []
            for geom in intersection.geoms:
                points.append(Point(geom.interpolate(0.5, normalized=True)))
            return points
    return []

def analyze_paths(paths_gdf, buildings_gdf, logger):
    paths_data = []
    intersections_data = []
    sindex = buildings_gdf.sindex

    with tqdm(total=len(paths_gdf), desc="Analyzing paths") as pbar:
        for path in paths_gdf.itertuples():
            line = path.geometry
            matches_idx = list(sindex.intersection(line.bounds))
            matches = buildings_gdf.iloc[matches_idx]
            intersected = matches[matches.geometry.intersects(line)]

            # Track unique buildings for this path
            unique_building_heights = []
            building_ids_seen = set()

            for b in intersected.itertuples():
                # Only process each building once per path
                if b.Index not in building_ids_seen:
                    building_ids_seen.add(b.Index)
                    unique_building_heights.append(b.heightroof_m)
                    
                    # Get one representative intersection point per building
                    points = get_intersection_points(line, b.geometry)
                    if points:
                        # Use the first intersection point as representative
                        intersections_data.append({
                            'geometry': points[0],
                            'height': b.heightroof_m,  # Keep as 'height' for compatibility
                            'building_id': b.Index,
                            'Park1_point_id': getattr(path, 'park1_point_id', None),
                            'Park2_point_id': getattr(path, 'park2_point_id', None),
                            'Distance_m': path.euclidean_distance
                        })

            # Calculate statistics on unique buildings only
            valid_heights = [h for h in unique_building_heights if not np.isnan(h)]
            avg = np.mean(valid_heights) if valid_heights else 0.0
            stderr = sem(valid_heights) if len(valid_heights) > 1 else 0.0

            paths_data.append({
                'geometry': line,
                'Park1_point_id': getattr(path, 'park1_point_id', None),
                'Park2_point_id': getattr(path, 'park2_point_id', None),
                'Distance_m': path.euclidean_distance,
                'Count_Intersected_Buildings': len(unique_building_heights),
                'Average_heightroof_m': avg,
                'Standard_Error_heightroof_m': stderr,
                'Is_Zero_Building_Path': len(unique_building_heights) == 0,
                'Intersected_heightroofs_m': unique_building_heights  # Add the full list
            })

            pbar.update(1)

    paths_gdf_out = gpd.GeoDataFrame(paths_data, crs=paths_gdf.crs)
    intersections_gdf = gpd.GeoDataFrame(intersections_data, crs=paths_gdf.crs)
    
    # Log summary statistics
    logger.info(f"Total paths processed: {len(paths_gdf_out)}")
    logger.info(f"Paths with buildings: {(paths_gdf_out['Count_Intersected_Buildings'] > 0).sum()}")
    logger.info(f"Paths without buildings: {(paths_gdf_out['Count_Intersected_Buildings'] == 0).sum()}")
    logger.info(f"Total unique building intersections: {len(intersections_gdf)}")
    
    return paths_gdf_out, intersections_gdf

def main():
    logger = setup_logging()
    start = time.time()

    try:
        # Load paths from results
        logger.info("Loading paths data...")
        paths_gdf = load_gdf("../outputs/results.gpkg", layer="results_point_id_fixed")
        
        # Load buildings data
        logger.info("Loading buildings data...")
        buildings_gdf = load_gdf("../../data/new_york_city_buildings_26918.gpkg")
        buildings_gdf = prepare_buildings_data(buildings_gdf)
        logger.info(f"Loaded {len(buildings_gdf)} buildings")

        # Filter paths to <= 1000m
        logger.info("Filtering paths to <= 1000m...")
        paths_gdf = paths_gdf[paths_gdf['euclidean_distance'] <= 1000]
        logger.info(f"Filtered to {len(paths_gdf)} paths ≤ 1000m")

        # Analyze paths and buildings
        logger.info("Analyzing path-building intersections...")
        paths_out, intersections_out = analyze_paths(paths_gdf, buildings_gdf, logger)

        # Save results
        output_path = Path("../outputs/euclidean_path_building_analysis.gpkg")
        logger.info(f"Saving results to {output_path}...")
        
        paths_out.to_file(output_path, layer="paths", driver="GPKG")
        intersections_out.to_file(output_path, layer="intersections", driver="GPKG")
        
        logger.info(f"Saved to {output_path}")
        logger.info(f"Completed in {time.time() - start:.2f} seconds")

        # Print summary statistics for verification
        logger.info("\nSummary Statistics:")
        logger.info(f"Mean buildings per path: {paths_out['Count_Intersected_Buildings'].mean():.2f}")
        logger.info(f"Max buildings per path: {paths_out['Count_Intersected_Buildings'].max()}")
        logger.info(f"Mean building height: {intersections_out['height'].mean():.2f} m")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    main()

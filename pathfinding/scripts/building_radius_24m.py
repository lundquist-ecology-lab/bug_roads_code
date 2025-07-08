import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import logging
from pathlib import Path
import time

def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/park_building_analysis.log'),
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

def analyze_buildings_near_parks(parks_gdf: gpd.GeoDataFrame, 
                               buildings_gdf: gpd.GeoDataFrame, 
                               radius_m: float = 1000) -> gpd.GeoDataFrame:
    """
    Analyze buildings within specified radius of each park.
    Returns GeoDataFrame with analysis results for each park.
    """
    analysis_results = []
    building_sindex = buildings_gdf.sindex

    for idx, park in parks_gdf.iterrows():
        # Create buffer around park
        park_buffer = park.geometry.buffer(radius_m)
        
        # Find buildings within buffer using spatial index
        possible_matches_index = list(building_sindex.intersection(park_buffer.bounds))
        possible_matches = buildings_gdf.iloc[possible_matches_index]
        nearby_buildings = possible_matches[possible_matches.intersects(park_buffer)]
        
        # Calculate statistics
        building_heights = nearby_buildings['heightroof_m'].dropna()
        num_buildings = len(building_heights)
        
        if num_buildings > 0:
            avg_height = building_heights.mean()
            max_height = building_heights.max()
            num_low_buildings = len(building_heights[building_heights <= 24])
            pct_low_buildings = (num_low_buildings / num_buildings) * 100
        else:
            avg_height = 0
            max_height = 0
            pct_low_buildings = 0
        
        # Store results
        result = {
            'geometry': park.geometry,
            'park_id': park['point_id'] if 'point_id' in park else idx,
            'park_unique_id': park['unique_id'] if 'unique_id' in park else None,
            'num_buildings_within_radius': num_buildings,
            'avg_building_height_m': avg_height,
            'max_building_height_m': max_height,
            'pct_buildings_under_24m': pct_low_buildings
        }
        analysis_results.append(result)
    
    return gpd.GeoDataFrame(analysis_results, crs=parks_gdf.crs)

def main():
    logger = setup_logging()
    start_time = time.time()
    
    try:
        # Load data
        parks_gdf = load_gdf("../../data/open_and_park_single_26918.gpkg")
        buildings_gdf = load_gdf("../../data/new_york_city_buildings_26918.gpkg")
        buildings_gdf = prepare_buildings_data(buildings_gdf)
        
        # Analyze buildings near parks
        results_gdf = analyze_buildings_near_parks(parks_gdf, buildings_gdf, radius_m=1000)
        
        # Save results
        output_gpkg = Path('../outputs/park_24m_building_analysis.gpkg')
        results_gdf.to_file(output_gpkg, driver='GPKG')
        
        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Results saved to {output_gpkg}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

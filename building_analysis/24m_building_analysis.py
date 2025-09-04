import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

def load_data(park_analysis_path: str, borough_boundaries_path: str) -> tuple:
    """Load the park analysis results and borough boundaries."""
    park_stats = gpd.read_file(park_analysis_path)
    borough_boundaries = gpd.read_file(borough_boundaries_path)
    return park_stats, borough_boundaries

def assign_boroughs(park_stats: gpd.GeoDataFrame, borough_boundaries: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Assign borough names to each park using spatial join."""
    if park_stats.crs != borough_boundaries.crs:
        borough_boundaries = borough_boundaries.to_crs(park_stats.crs)
    
    park_stats_with_borough = gpd.sjoin(
        park_stats, 
        borough_boundaries[['boro_name', 'geometry']], 
        how='left', 
        predicate='intersects'
    )
    
    return park_stats_with_borough

def calculate_summary_statistics(df: pd.DataFrame, group_by_col: str = None) -> pd.DataFrame:
    """Calculate summary statistics for the entire dataset or by group."""
    stats_cols = [
        'num_buildings_within_radius',
        'avg_building_height_m',
        'max_building_height_m',
        'pct_buildings_under_24m'
    ]
    
    agg_dict = {
        'num_buildings_within_radius': ['count', 'mean', 'std', 'min', 'max'],
        'avg_building_height_m': ['mean', 'std', 'min', 'max'],
        'max_building_height_m': ['mean', 'std', 'min', 'max'],
        'pct_buildings_under_24m': ['mean', 'std', 'min', 'max']
    }
    
    if group_by_col:
        summary = df.groupby(group_by_col)[stats_cols].agg(agg_dict)
        # Flatten column names for borough stats
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    else:
        # For citywide stats, calculate each metric separately and combine
        summary = pd.DataFrame()
        
        # Number of buildings
        buildings_stats = df['num_buildings_within_radius'].agg(['count', 'mean', 'std', 'min', 'max'])
        summary['Number of Buildings'] = buildings_stats
        
        # Average building height
        height_stats = df['avg_building_height_m'].agg(['mean', 'std', 'min', 'max'])
        summary['Average Building Height (m)'] = pd.Series({
            'mean': height_stats['mean'],
            'std': height_stats['std'],
            'min': height_stats['min'],
            'max': height_stats['max']
        })
        
        # Maximum building height
        max_height_stats = df['max_building_height_m'].agg(['mean', 'std', 'min', 'max'])
        summary['Maximum Building Height (m)'] = pd.Series({
            'mean': max_height_stats['mean'],
            'std': max_height_stats['std'],
            'min': max_height_stats['min'],
            'max': max_height_stats['max']
        })
        
        # Percentage of buildings under 24m
        pct_stats = df['pct_buildings_under_24m'].agg(['mean', 'std', 'min', 'max'])
        summary['Buildings Under 24m (%)'] = pd.Series({
            'mean': pct_stats['mean'],
            'std': pct_stats['std'],
            'min': pct_stats['min'],
            'max': pct_stats['max']
        })
    
    return summary

def save_statistics_to_text(stats_df: pd.DataFrame, output_path: Path, title: str):
    """Save statistics DataFrame to a formatted text file."""
    with open(output_path, 'w') as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")
        f.write(stats_df.to_string(float_format=lambda x: '{:.2f}'.format(x)))
        f.write("\n")

def main():
    output_dir = Path('../outputs')
    output_dir.mkdir(exist_ok=True)
    
    park_stats, borough_boundaries = load_data(
        '../outputs/park_24m_building_analysis.gpkg',
        '../data/borough_boundaries_26918.gpkg'
    )
    
    park_stats_with_borough = assign_boroughs(park_stats, borough_boundaries)
    
    citywide_stats = calculate_summary_statistics(park_stats_with_borough)
    borough_stats = calculate_summary_statistics(park_stats_with_borough, 'boro_name')
    
    save_statistics_to_text(
        citywide_stats,
        output_dir / '24m_citywide_statistics.txt',
        'Citywide Park and Building Statistics'
    )
    
    save_statistics_to_text(
        borough_stats,
        output_dir / '24m_borough_statistics.txt',
        'Borough-Level Park and Building Statistics'
    )
    
    print("\nCitywide Statistics:")
    print("===================")
    print(citywide_stats)
    print("\nBorough Statistics:")
    print("==================")
    print(borough_stats)
    
    print(f"\nResults saved to:")
    print(f"- {output_dir / '24m_citywide_statistics.txt'}")
    print(f"- {output_dir / '24m_borough_statistics.txt'}")

if __name__ == "__main__":
    main()

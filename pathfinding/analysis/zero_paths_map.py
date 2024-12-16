import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import nearest_points
import ast
from pathlib import Path

def safe_eval_list(x):
    """Safely evaluate string representation of list or return empty list for NaN."""
    if pd.isna(x):
        return []
    try:
        if isinstance(x, str):
            return ast.literal_eval(x)
        elif isinstance(x, list):
            return x
        else:
            return []
    except:
        return []

def get_nearest_boundary_points(geom1, geom2):
    """Find the nearest points between two geometries' boundaries."""
    boundary1 = geom1.boundary
    boundary2 = geom2.boundary
    point1, point2 = nearest_points(boundary1, boundary2)
    return point1, point2

def create_zero_paths_gpkg(input_csv, parks_gpkg, output_gpkg):
    """Create a GeoPackage file containing paths with zero buildings."""
    print("Reading input files...")
    # Read the CSV file and parks GeoPackage
    df = pd.read_csv(input_csv)
    parks_gdf = gpd.read_file(parks_gpkg)
    
    # Identify zero building paths
    print("Identifying zero building paths...")
    df['is_zero_buildings'] = df['Intersected_heightroofs_m'].apply(
        lambda x: len(safe_eval_list(x)) == 0
    )
    
    # Filter for zero building paths
    zero_paths_df = df[df['is_zero_buildings']].copy()
    
    print("Creating path geometries...")
    # Create geometries by connecting park boundaries
    geometries = []
    for _, row in zero_paths_df.iterrows():
        try:
            # Get park geometries
            park1 = parks_gdf[parks_gdf['point_id'] == row['Park1_point_id']].iloc[0]
            park2 = parks_gdf[parks_gdf['point_id'] == row['Park2_point_id']].iloc[0]
            
            # Get nearest points on boundaries
            point1, point2 = get_nearest_boundary_points(park1.geometry, park2.geometry)
            
            # Create LineString
            line = LineString([point1, point2])
            geometries.append(line)
        except Exception as e:
            print(f"Warning: Could not create geometry for path between parks {row['Park1_point_id']} and {row['Park2_point_id']}: {str(e)}")
            geometries.append(None)
    
    # Create GeoDataFrame
    print("Creating GeoDataFrame...")
    gdf = gpd.GeoDataFrame(
        zero_paths_df[[
            'Park1_point_id', 'Park1_unique_id',
            'Park2_point_id', 'Park2_unique_id',
            'Distance_m', 'Park1_borough', 'Park2_borough'
        ]],
        geometry=geometries,
        crs=parks_gdf.crs  # Use the same CRS as the parks file
    )
    
    # Remove any rows where geometry creation failed
    gdf = gdf.dropna(subset=['geometry'])
    
    # Add flag for paths within same park
    gdf['same_park'] = gdf['Park1_unique_id'] == gdf['Park2_unique_id']
    
    # Save to GeoPackage
    print("Saving to GeoPackage...")
    gdf.to_file(output_gpkg, driver="GPKG")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total zero building paths: {len(gdf)}")
    print(f"Same park paths: {gdf['same_park'].sum()}")
    print(f"Different park paths: {(~gdf['same_park']).sum()}")
    print("\nPaths by borough:")
    for borough in gdf['Park1_borough'].unique():
        if pd.notna(borough):
            borough_name = {
                'X': 'Bronx',
                'M': 'Manhattan',
                'B': 'Brooklyn',
                'Q': 'Queens',
                'R': 'Staten Island'
            }.get(borough, str(borough))
            count = len(gdf[gdf['Park1_borough'] == borough])
            print(f"{borough_name}: {count} paths")
    
    return gdf

if __name__ == "__main__":
    # File paths
    base_dir = Path(__file__).parent
    input_csv = base_dir / "../output/parks_with_boroughs_and_analysis.csv"
    parks_gpkg = base_dir / "../../data/open_and_park_single_26918.gpkg"
    output_gpkg = base_dir / "../output/zero_building_paths.gpkg"
    
    # Create output directory if it doesn't exist
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print("Starting GeoPackage creation process...")
        gdf = create_zero_paths_gpkg(input_csv, parks_gpkg, output_gpkg)
        print(f"\nGeoPackage file created successfully at: {output_gpkg}")
    except Exception as e:
        print(f"Error creating GeoPackage: {str(e)}")
        import traceback
        traceback.print_exc()
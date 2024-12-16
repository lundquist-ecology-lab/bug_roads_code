import pandas as pd
import geopandas as gpd

# File paths
distance_files = [
    "park_clusters_100m.csv",
    "park_clusters_250m.csv",
    "park_clusters_500m.csv",
    "park_clusters_1000m.csv"
]
shapes_gpkg = "../../../data/open_and_park_single_26918.gpkg"
boroughs_gpkg = "../../../data/borough_boundaries_26918.gpkg"
output_file = "analysis_results.txt"

# Load GeoPackage files
shapes_gdf = gpd.read_file(shapes_gpkg)
boroughs_gdf = gpd.read_file(boroughs_gpkg)

# Ensure CRS are the same
shapes_gdf = shapes_gdf.to_crs(boroughs_gdf.crs)

# Open the output file
with open(output_file, "w") as f:
    for csv_file in distance_files:
        print(f"Processing {csv_file}...")
        
        # Load the current CSV file
        csv_data = pd.read_csv(csv_file)
        
        # Merge CSV with shapes based on unique_id
        shapes_with_csv = shapes_gdf.merge(csv_data, on="unique_id", how="left")
        
        # Use the correct point_id column
        if 'point_id_y' in shapes_with_csv.columns:
            shapes_with_csv['point_id'] = shapes_with_csv['point_id_y']

        # Perform a spatial join with boroughs
        shapes_with_boroughs = gpd.sjoin(shapes_with_csv, boroughs_gdf, how="left", predicate="intersects")

        # Remove clusters with only one point_id
        cluster_sizes = shapes_with_boroughs.groupby('cluster_id')['point_id'].nunique()
        valid_clusters = cluster_sizes[cluster_sizes > 1].index
        shapes_with_boroughs = shapes_with_boroughs[shapes_with_boroughs['cluster_id'].isin(valid_clusters)]

        # Identify multi-borough clusters
        borough_counts_per_cluster = shapes_with_boroughs.groupby('cluster_id')['boro_name'].nunique()
        multi_borough_clusters = borough_counts_per_cluster[borough_counts_per_cluster > 1].index
        single_borough_clusters = borough_counts_per_cluster[borough_counts_per_cluster == 1].index

        # Separate single-borough and multi-borough clusters
        single_borough_shapes = shapes_with_boroughs[shapes_with_boroughs['cluster_id'].isin(single_borough_clusters)]
        multi_borough_shapes = shapes_with_boroughs[shapes_with_boroughs['cluster_id'].isin(multi_borough_clusters)]

        # Total clusters citywide (including multi-borough)
        total_clusters = shapes_with_boroughs['cluster_id'].nunique()

        # Mean and SE of point_id per cluster citywide (including multi-borough)
        citywide_mean = (
            shapes_with_boroughs.groupby('cluster_id')['point_id']
            .nunique()
            .mean()
        )
        citywide_se = (
            shapes_with_boroughs.groupby('cluster_id')['point_id']
            .nunique()
            .sem()
        )

        # Total clusters and stats per borough (single-borough only)
        borough_stats = (
            single_borough_shapes.groupby('boro_name')
            .apply(
                lambda group: pd.Series({
                    'total_clusters': group['cluster_id'].nunique(),
                    'mean_parks_per_cluster': group.groupby('cluster_id')['point_id'].nunique().mean(),
                    'se_parks_per_cluster': group.groupby('cluster_id')['point_id'].nunique().sem()
                })
            )
        )

        # Stats for multi-borough clusters
        multi_borough_stats = pd.Series({
            'total_clusters': multi_borough_shapes['cluster_id'].nunique(),
            'mean_parks_per_cluster': multi_borough_shapes.groupby('cluster_id')['point_id'].nunique().mean(),
            'se_parks_per_cluster': multi_borough_shapes.groupby('cluster_id')['point_id'].nunique().sem()
        })

        # Write results for the current distance cutoff to the output file
        f.write(f"Results for {csv_file}:\n")
        f.write(f"Total clusters citywide (including multi-borough): {total_clusters}\n")
        f.write(f"Mean parks per cluster citywide (including multi-borough): {citywide_mean:.2f}\n")
        f.write(f"SE parks per cluster citywide (including multi-borough): {citywide_se:.2f}\n")
        f.write("Single-borough statistics:\n")
        f.write(borough_stats.to_string())
        f.write("\n\n")
        f.write("Multi-borough cluster statistics:\n")
        f.write(multi_borough_stats.to_string())
        f.write("\n\n")

print("Processing complete. Results written to", output_file)

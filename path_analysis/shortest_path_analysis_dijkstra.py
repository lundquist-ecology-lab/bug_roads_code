import geopandas as gpd
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def format_stats(stats, title):
    output = f"\n{title}\n{'=' * len(title)}\n"
    output += f"Number of Paths: {stats['count']:,.0f}\n"
    output += f"Mean Distance: {stats['mean']:,.2f} meters\n"
    output += f"Standard Error: {stats['se']:,.2f} meters\n"
    output += f"Median Distance: {stats['median']:,.2f} meters\n"
    output += f"Standard Deviation: {stats['std']:,.2f} meters\n"
    output += f"Minimum Distance: {stats['min']:,.2f} meters\n"
    output += f"Maximum Distance: {stats['max']:,.2f} meters\n"
    output += f"25th Percentile: {stats['q25']:,.2f} meters\n"
    output += f"75th Percentile: {stats['q75']:,.2f} meters\n"
    return output

def get_summary_stats(data):
    return pd.Series({
        'count': len(data),
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
        'se': data.std() / np.sqrt(len(data)) if len(data) > 0 else 0,
        'min': data.min(),
        'max': data.max(),
        'q25': data.quantile(0.25),
        'q75': data.quantile(0.75)
    })

def analyze_path_distances():
    logging.info("Loading path results...")
    paths_gdf = gpd.read_file("../outputs/dijkstra_paths.gpkg", layer="results_point_id_fixed")

    logging.info("Filtering paths to <= 1000m...")
    paths_gdf = paths_gdf[paths_gdf["path_length"] <= 1000]
    logging.info(f"Number of paths <= 1000m: {len(paths_gdf)}")

    logging.info("Loading borough boundaries...")
    boroughs = gpd.read_file("../data/borough_boundaries_26918.gpkg")

    logging.info("Loading parks data...")
    park_gdf = gpd.read_file("../data/open_and_park_single_26918.gpkg")

    if park_gdf.crs != paths_gdf.crs:
        park_gdf = park_gdf.to_crs(paths_gdf.crs)
    if boroughs.crs != park_gdf.crs:
        boroughs = boroughs.to_crs(park_gdf.crs)

    logging.info("Assigning parks to boroughs...")
    park_with_boroughs = gpd.sjoin(
        park_gdf, boroughs[["geometry", "boro_name"]],
        how="left", predicate="intersects"
    ).drop_duplicates(subset="point_id")
    park_with_boroughs = park_with_boroughs.loc[:, ~park_with_boroughs.columns.duplicated()]
    park_borough_dict = park_with_boroughs.set_index("point_id")["boro_name"].to_dict()

    logging.info("Mapping boroughs to paths...")
    paths_gdf["park1_borough"] = paths_gdf["park1_point_id"].map(park_borough_dict)
    paths_gdf["park2_borough"] = paths_gdf["park2_point_id"].map(park_borough_dict)

    output_gpkg = "../results/filtered_paths_with_boroughs.gpkg"
    logging.info(f"Saving result to {output_gpkg}...")
    paths_gdf.to_file(output_gpkg, layer="paths", driver="GPKG")

    # ==== STATISTICS OUTPUT ====
    output_dir = Path("../results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    stats_path = output_dir / "path_distance_analysis_1000m.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"Path Distance Analysis Results (Paths <= 1000m)\n")
        f.write(f"Generated on: {timestamp}\n")
        f.write("=" * 50 + "\n\n")

        logging.info("Calculating citywide statistics...")
        citywide_stats = get_summary_stats(paths_gdf["path_length"])
        f.write(format_stats(citywide_stats, "Citywide Statistics"))

        logging.info("Calculating within-borough statistics...")
        f.write("\nWithin-Borough Statistics\n" + "=" * 23 + "\n")
        within = paths_gdf[paths_gdf["park1_borough"] == paths_gdf["park2_borough"]]
        boroughs_list = sorted([b for b in paths_gdf["park1_borough"].unique() if pd.notna(b)])

        for boro in boroughs_list:
            boro_data = within[within["park1_borough"] == boro]
            if len(boro_data) > 0:
                f.write(format_stats(get_summary_stats(boro_data["path_length"]), boro))

        logging.info("Calculating between-borough statistics...")
        f.write("\nBetween-Borough Statistics\n" + "=" * 25 + "\n")
        between = paths_gdf[paths_gdf["park1_borough"] != paths_gdf["park2_borough"]]

        for i, b1 in enumerate(boroughs_list):
            for b2 in boroughs_list[i + 1:]:
                boro_pair = between[
                    ((between["park1_borough"] == b1) & (between["park2_borough"] == b2)) |
                    ((between["park1_borough"] == b2) & (between["park2_borough"] == b1))
                ]
                if len(boro_pair) > 0:
                    f.write(format_stats(get_summary_stats(boro_pair["path_length"]), f"{b1} <-> {b2}"))

    logging.info(f"Analysis completed successfully. Results written to {stats_path}")

if __name__ == "__main__":
    analyze_path_distances()


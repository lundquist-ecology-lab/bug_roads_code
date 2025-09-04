import geopandas as gpd
import pandas as pd
from pathlib import Path

def assign_boroughs(paths_gdf, parks_gdf):
    """Assign boroughs to paths using park1 and park2 point IDs from parks data."""
    # Drop duplicates and create lookup table
    park_borough_map = (
        parks_gdf.drop_duplicates(subset="point_id")
        .set_index("point_id")["borough"]
        .to_dict()
    )

    paths_gdf["park1_borough"] = paths_gdf["park1_point_id"].map(park_borough_map)
    paths_gdf["park2_borough"] = paths_gdf["park2_point_id"].map(park_borough_map)
    return paths_gdf

def analyze_building_path_crossings(paths_gdf, parks_gdf):
    """Analyze building-intersecting paths that cross parks."""
    total_paths = len(paths_gdf)
    total_building_paths = (~paths_gdf["Is_Zero_Building_Path"]).sum()

    building_paths = paths_gdf[~paths_gdf["Is_Zero_Building_Path"]]
    parks_sindex = parks_gdf.sindex
    crossing_data = []
    path_park_counts = {}

    for idx, path in building_paths.iterrows():
        bounds = path.geometry.bounds
        candidates = parks_gdf.iloc[list(parks_sindex.intersection(bounds))]
        intersecting = candidates[candidates.intersects(path.geometry)]

        if not intersecting.empty:
            path_park_counts[idx] = len(intersecting)
            for park_idx, park in intersecting.iterrows():
                inter = path.geometry.intersection(park.geometry)
                if not inter.is_empty:
                    crossing_data.append({
                        "path_id": idx,
                        "park_id": park_idx,
                        "intersection_length_m": inter.length,
                        "park_portion": inter.length / path.geometry.length,
                        "park1_borough": path["park1_borough"],
                        "park2_borough": path["park2_borough"],
                        "total_path_length": path.geometry.length
                    })

    df = pd.DataFrame(crossing_data)

    def stat_agg(df, col):
        return {
            "min": df[col].min() if not df.empty else 0,
            "max": df[col].max() if not df.empty else 0,
            "mean": df[col].mean() if not df.empty else 0
        }

    overall_stats = {
        "total_paths": total_paths,
        "building_intersecting_paths": total_building_paths,
        "paths_with_park_crossings": len(path_park_counts),
        "parks_crossed": stat_agg(pd.DataFrame(path_park_counts.values(), columns=["count"]), "count"),
        "crossing_lengths": stat_agg(df, "intersection_length_m"),
        "park_portion": stat_agg(df, "park_portion"),
        "most_crossed_parks": df["park_id"].value_counts().head(10).to_dict()
    }

    return overall_stats, df

def save_results(stats, df, output_path):
    with open(output_path, "w") as f:
        f.write("OVERALL STATISTICS\n==================\n")
        f.write(f"Building-Intersecting Paths: {stats['building_intersecting_paths']} / {stats['total_paths']}\n")
        f.write(f"Paths Crossing Parks: {stats['paths_with_park_crossings']}\n")
        f.write(f"Parks Crossed per Path: min={stats['parks_crossed']['min']}, max={stats['parks_crossed']['max']}, mean={stats['parks_crossed']['mean']:.2f}\n")
        f.write(f"Crossing Lengths (m): min={stats['crossing_lengths']['min']:.2f}, max={stats['crossing_lengths']['max']:.2f}, mean={stats['crossing_lengths']['mean']:.2f}\n")
        f.write(f"Park Portion of Path: min={stats['park_portion']['min']:.2%}, max={stats['park_portion']['max']:.2%}, mean={stats['park_portion']['mean']:.2%}\n\n")
        f.write("Most Frequently Crossed Parks:\n")
        for k, v in stats["most_crossed_parks"].items():
            f.write(f"Park ID {k}: {v} crossings\n")

    df.to_csv(output_path.parent / "euclidean_building_path_park_crossings_details.csv", index=False)

def main():
    base_dir = Path(__file__).parent
    building_path_gpkg = base_dir.parent / "outputs" / "euclidean_path_building_analysis.gpkg"
    parks_gpkg = base_dir.parent.parent / "data" / "nyc_parks_centroids.gpkg"

    print("Loading data...")
    paths_gdf = gpd.read_file(building_path_gpkg)
    parks_gdf = gpd.read_file(parks_gpkg)

    print("Assigning boroughs...")
    paths_gdf = assign_boroughs(paths_gdf, parks_gdf)

    print("Running analysis...")
    stats, df = analyze_building_path_crossings(paths_gdf, parks_gdf)

    output_path = base_dir.parent / "outputs" / "euclidean_building_path_park_crossing_analysis.txt"
    print("Saving results...")
    save_results(stats, df, output_path)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()


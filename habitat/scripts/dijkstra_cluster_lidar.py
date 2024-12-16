import pandas as pd
from pathlib import Path

# Load cluster data
def load_cluster_data(cluster_csv_path):
    return pd.read_csv(cluster_csv_path)

# Merge vegetation CSV files and compute total area for clusters
def process_clusters(veg_files, cluster_files, output_path, target_percentage):
    # Load all vegetation data
    vegetation_data = []
    for veg_file in veg_files:
        df = pd.read_csv(veg_file)
        vegetation_data.append(df)
    vegetation_df = pd.concat(vegetation_data, ignore_index=True)

    # Load and process each cluster file
    all_results = []
    for cluster_file in cluster_files:
        cluster_data = load_cluster_data(cluster_file)
        radius = int(Path(cluster_file).stem.split("_")[-1][:-1])
        
        merged_df = cluster_data.merge(vegetation_df, on=['point_id'], how='inner')
        cluster_summary = merged_df.groupby('cluster_id')['area_m2'].sum().reset_index()
        
        # Calculate threshold
        circle_area_m2 = 3.14159 * (radius ** 2)
        threshold_area_m2 = circle_area_m2 * (target_percentage / 100)

        cluster_summary['radius_m'] = radius
        cluster_summary['meets_threshold'] = cluster_summary['area_m2'] >= threshold_area_m2

        all_results.append(cluster_summary)

    # Combine all results and save
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Summary statistics
    summary = combined_results.groupby('radius_m')['meets_threshold'].mean() * 100
    for radius, percentage in summary.items():
        print(f"{percentage:.2f}% of clusters meet the vegetative cover threshold for radius {radius}m")

# Main function
def main():
    veg_dir = Path("../outputs/lidar_results_parks_121024")
    cluster_dir = Path("../../clusters/output/cluster_ranges")
    output_path = "../outputs/vegetation_cluster_results.csv"
    target_percentage = 11.6

    veg_files = list(veg_dir.glob("*_vegetation_*.csv"))
    cluster_files = list(cluster_dir.glob("park_clusters_*.csv"))

    process_clusters(veg_files, cluster_files, output_path, target_percentage)

if __name__ == "__main__":
    main()

import pandas as pd
from pathlib import Path

# Load cluster data
def load_cluster_data(cluster_csv_path):
    return pd.read_csv(cluster_csv_path)

# Merge vegetation CSV files and compute total area for clusters
def process_clusters(veg_files, cluster_files, output_path, target_percentage):
    # Check if vegetation files exist
    if not veg_files:
        raise ValueError(f"No vegetation files found. Check the directory path and file pattern.")
    
    print(f"Found {len(veg_files)} vegetation files")
    
    # Load all vegetation data and aggregate by point_id
    all_vegetation_data = []
    
    for veg_file in veg_files:
        try:
            df = pd.read_csv(veg_file)
            
            # Find the vegetation area column
            veg_area_col = None
            for col in df.columns:
                if 'vegetation_total_area_m2' in col:
                    veg_area_col = col
                    break
            
            if veg_area_col is None:
                print(f"Warning: No vegetation area column found in {veg_file.name}")
                continue
            
            # Keep only the relevant columns
            veg_data = df[['point_id', veg_area_col]].copy()
            veg_data = veg_data.rename(columns={veg_area_col: 'vegetation_area_m2'})
            
            # Remove rows with missing point_ids
            veg_data = veg_data.dropna(subset=['point_id'])
            
            all_vegetation_data.append(veg_data)
            print(f"Loaded {len(veg_data)} valid rows from {veg_file.name}")
            
        except Exception as e:
            print(f"Error loading {veg_file}: {e}")
            continue
    
    if not all_vegetation_data:
        raise ValueError("No vegetation data could be loaded from the files.")
    
    # Combine all vegetation data
    combined_veg = pd.concat(all_vegetation_data, ignore_index=True)
    print(f"Total vegetation records before aggregation: {len(combined_veg)}")
    
    # Aggregate vegetation area by point_id (sum all vegetation types per point)
    vegetation_df = combined_veg.groupby('point_id')['vegetation_area_m2'].sum().reset_index()
    print(f"Unique points after aggregation: {len(vegetation_df)}")
    
    # Show some statistics
    veg_stats = vegetation_df['vegetation_area_m2'].describe()
    print(f"Vegetation area statistics:")
    print(f"  Mean: {veg_stats['mean']:.2f} m²")
    print(f"  Median: {veg_stats['50%']:.2f} m²")
    print(f"  Max: {veg_stats['max']:.2f} m²")
    print(f"  Points with vegetation > 0: {(vegetation_df['vegetation_area_m2'] > 0).sum()}")

    # Check if cluster files exist
    if not cluster_files:
        raise ValueError(f"No cluster files found. Check the directory path and file pattern.")
    
    print(f"Found {len(cluster_files)} cluster files")

    # Load and process each cluster file
    all_results = []
    for cluster_file in cluster_files:
        try:
            cluster_data = load_cluster_data(cluster_file)
            radius = int(Path(cluster_file).stem.split("_")[-1][:-1])
            print(f"\n=== Processing {cluster_file.name} (radius: {radius}m) ===")
            
            merged_df = cluster_data.merge(vegetation_df, on=['point_id'], how='inner')
            print(f"Cluster data: {len(cluster_data)} rows")
            print(f"After merge: {len(merged_df)} rows")
            
            if len(merged_df) == 0:
                print(f"Warning: No matching point_ids found for {cluster_file.name}")
                continue
            
            # Group by cluster and sum vegetation areas
            cluster_summary = merged_df.groupby('cluster_id')['vegetation_area_m2'].sum().reset_index()
            cluster_summary.rename(columns={'vegetation_area_m2': 'area_m2'}, inplace=True)
            
            # Calculate threshold
            circle_area_m2 = 3.14159 * (radius ** 2)
            threshold_area_m2 = circle_area_m2 * (target_percentage / 100)
            
            print(f"Circle area: {circle_area_m2:.2f} m²")
            print(f"Threshold area ({target_percentage}%): {threshold_area_m2:.2f} m²")

            cluster_summary['radius_m'] = radius
            cluster_summary['meets_threshold'] = cluster_summary['area_m2'] >= threshold_area_m2
            
            # Show some cluster statistics
            cluster_area_stats = cluster_summary['area_m2'].describe()
            print(f"Cluster vegetation area statistics:")
            print(f"  Mean: {cluster_area_stats['mean']:.2f} m²")
            print(f"  Median: {cluster_area_stats['50%']:.2f} m²")
            print(f"  Max: {cluster_area_stats['max']:.2f} m²")
            print(f"  Clusters meeting threshold: {cluster_summary['meets_threshold'].sum()}")
            print(f"  Total clusters: {len(cluster_summary)}")

            all_results.append(cluster_summary)
            
            # Show a few sample clusters
            print("Sample clusters:")
            sample_clusters = cluster_summary.head(5)
            for _, cluster in sample_clusters.iterrows():
                meets = "YES" if cluster['meets_threshold'] else "NO"
                percentage = (cluster['area_m2'] / circle_area_m2) * 100
                print(f"  Cluster {cluster['cluster_id']}: {cluster['area_m2']:.2f} m² ({percentage:.1f}% coverage, meets threshold: {meets})")
            
        except Exception as e:
            print(f"Error processing {cluster_file}: {e}")
            continue

    if not all_results:
        raise ValueError("No cluster data could be processed successfully.")

    # Combine all results and save
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Summary statistics
    print("\n=== FINAL SUMMARY ===")
    summary = combined_results.groupby('radius_m')['meets_threshold'].mean() * 100
    for radius, percentage in summary.items():
        total_clusters = len(combined_results[combined_results['radius_m'] == radius])
        meeting_threshold = combined_results[(combined_results['radius_m'] == radius) & (combined_results['meets_threshold'])].shape[0]
        circle_area = 3.14159 * (radius ** 2)
        threshold_area = circle_area * (target_percentage / 100)
        print(f"Radius {radius}m: {percentage:.2f}% ({meeting_threshold}/{total_clusters}) clusters meet {target_percentage}% threshold ({threshold_area:.0f} m²)")

# Main function
def main():
    veg_dir = Path("../outputs/park_lidar_results")
    cluster_dir = Path("../../clusters/outputs/euclidean_clusters")
    output_path = "../outputs/vegetation_euclidean_cluster_fixed_results.csv"
    target_percentage = 11.6

    # Check if directories exist
    if not veg_dir.exists():
        print(f"Error: Vegetation directory does not exist: {veg_dir}")
        return
    
    if not cluster_dir.exists():
        print(f"Error: Cluster directory does not exist: {cluster_dir}")
        return

    veg_files = list(veg_dir.glob("*_vegetation_*.csv"))
    cluster_files = list(cluster_dir.glob("euclidean_clusters_*.csv"))

    try:
        process_clusters(veg_files, cluster_files, output_path, target_percentage)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

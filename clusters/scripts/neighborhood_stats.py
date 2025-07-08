import pandas as pd
import numpy as np
from pathlib import Path
import logging

class ParkStatsAnalyzer:
    def __init__(self, results_dir):
        """
        Initialize the park statistics analyzer.
        
        Args:
            results_dir: Path to directory containing the cluster results CSV files
        """
        self.logger = logging.getLogger(__name__)
        self.results_dir = Path(results_dir)
        
    def calculate_cluster_stats(self, df, group_col='cluster_id'):
        """Calculate comprehensive statistics for clusters."""
        # Get cluster sizes
        cluster_sizes = df.groupby(group_col).size()
        cluster_sizes = cluster_sizes[cluster_sizes > 1]  # Exclude singletons
        
        if len(cluster_sizes) == 0:
            return {
                'total_clusters': 0,
                'avg_parks_per_cluster': 0,
                'se_parks_per_cluster': 0,
                'total_parks_in_clusters': 0,
                'median_parks_per_cluster': 0
            }
        
        return {
            'total_clusters': len(cluster_sizes),
            'avg_parks_per_cluster': cluster_sizes.mean(),
            'se_parks_per_cluster': cluster_sizes.std() / np.sqrt(len(cluster_sizes)),
            'total_parks_in_clusters': cluster_sizes.sum(),
            'median_parks_per_cluster': cluster_sizes.median()
        }
    
    def analyze_threshold_results(self, threshold):
        """Analyze results for a specific distance threshold."""
        try:
            # Load the cluster results for this threshold
            df = pd.read_csv(self.results_dir / f'park_clusters_{threshold}m.csv')
            
            # Calculate citywide statistics
            citywide_stats = self.calculate_cluster_stats(df)
            
            # Calculate borough-specific statistics
            borough_stats = {}
            for borough in df['borough'].dropna().unique():
                borough_df = df[df['borough'] == borough]
                borough_stats[borough] = self.calculate_cluster_stats(borough_df)
            
            # Calculate multi-borough cluster statistics
            multiborough_clusters = df.groupby('cluster_id')['borough'].nunique()
            multiborough_df = df[df['cluster_id'].isin(
                multiborough_clusters[multiborough_clusters > 1].index
            )]
            multiborough_stats = self.calculate_cluster_stats(multiborough_df)
            
            # Add average boroughs per multi-borough cluster
            if len(multiborough_clusters[multiborough_clusters > 1]) > 0:
                multiborough_stats['avg_boroughs_per_cluster'] = multiborough_clusters[
                    multiborough_clusters > 1
                ].mean()
                multiborough_stats['se_boroughs_per_cluster'] = multiborough_clusters[
                    multiborough_clusters > 1
                ].std() / np.sqrt(len(multiborough_clusters[multiborough_clusters > 1]))
            else:
                multiborough_stats['avg_boroughs_per_cluster'] = 0
                multiborough_stats['se_boroughs_per_cluster'] = 0
            
            return {
                'citywide': citywide_stats,
                'boroughs': borough_stats,
                'multiborough': multiborough_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing threshold {threshold}m: {e}")
            raise
    
    def analyze_all_thresholds(self, thresholds, output_path):
        """Analyze and save summary statistics for all thresholds."""
        results = {}
        for threshold in thresholds:
            self.logger.info(f"Analyzing statistics for {threshold}m threshold...")
            results[threshold] = self.analyze_threshold_results(threshold)
        
        self.save_summary(results, output_path)
    
    def save_summary(self, results, output_path):
        """Save comprehensive statistical summary."""
        with open(output_path, 'w') as f:
            f.write("Park Cluster Statistical Summary\n")
            f.write("===============================\n\n")
            
            for threshold, stats in results.items():
                f.write(f"\nResults for {threshold}m threshold:\n")
                f.write("-" * 30 + "\n\n")
                
                # Citywide stats
                f.write("Citywide Statistics:\n")
                citywide = stats['citywide']
                f.write(f"- Total clusters: {citywide['total_clusters']}\n")
                f.write(f"- Average parks per cluster: {citywide['avg_parks_per_cluster']:.1f} "
                       f"(SE: ±{citywide['se_parks_per_cluster']:.2f})\n")
                f.write(f"- Median parks per cluster: {citywide['median_parks_per_cluster']:.1f}\n")
                f.write(f"- Total parks in clusters: {citywide['total_parks_in_clusters']}\n\n")
                
                # Borough stats
                f.write("Borough Statistics:\n")
                for borough, borough_stats in stats['boroughs'].items():
                    f.write(f"\n{borough}:\n")
                    f.write(f"- Total clusters: {borough_stats['total_clusters']}\n")
                    f.write(f"- Average parks per cluster: {borough_stats['avg_parks_per_cluster']:.1f} "
                           f"(SE: ±{borough_stats['se_parks_per_cluster']:.2f})\n")
                    f.write(f"- Median parks per cluster: {borough_stats['median_parks_per_cluster']:.1f}\n")
                    f.write(f"- Total parks in clusters: {borough_stats['total_parks_in_clusters']}\n")
                
                # Multi-borough stats
                f.write("\nMulti-borough Clusters:\n")
                multi = stats['multiborough']
                f.write(f"- Total clusters: {multi['total_clusters']}\n")
                f.write(f"- Average parks per cluster: {multi['avg_parks_per_cluster']:.1f} "
                       f"(SE: ±{multi['se_parks_per_cluster']:.2f})\n")
                f.write(f"- Median parks per cluster: {multi['median_parks_per_cluster']:.1f}\n")
                f.write(f"- Total parks: {multi['total_parks_in_clusters']}\n")
                f.write(f"- Average boroughs per cluster: {multi['avg_boroughs_per_cluster']:.1f} "
                       f"(SE: ±{multi['se_boroughs_per_cluster']:.2f})\n")
                
                f.write("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Set up paths
    results_dir = Path('../outputs')
    output_path = '../outputs/cluster_statistics_summary.txt'
    
    # Define thresholds to analyze
    thresholds = [100, 250, 500, 1000]
    
    # Run statistical analysis
    analyzer = ParkStatsAnalyzer(results_dir=results_dir)
    analyzer.analyze_all_thresholds(thresholds, output_path)
import geopandas as gpd
import networkx as nx
import pandas as pd
from pathlib import Path
import logging
from collections import defaultdict
import numpy as np
from tqdm.auto import tqdm

class ParkClusterAnalyzer:
    def __init__(self, results_path, boroughs_path):
        """
        Initialize the park cluster analyzer.
        
        Args:
            results_path: Path to the results.gpkg file from PathFinder
            boroughs_path: Path to the borough boundaries file
        """
        self.logger = logging.getLogger(__name__)
        self.results_path = Path(results_path)
        self.boroughs_path = Path(boroughs_path)
        self.park_id_mapping = {}
        self.park_geometries = {}
        self.paths_gdf = None
        self.boroughs = None
        
    def load_data(self):
        """Load the path results and borough data."""
        self.logger.info("Loading data...")
        
        try:
            # Load the results from PathFinder
            self.paths_gdf = gpd.read_file(self.results_path)
            
            # Load borough boundaries
            self.boroughs = gpd.read_file(self.boroughs_path)
            
            # Create a GeoDataFrame with unique park points
            unique_parks = []
            for _, row in self.paths_gdf.iterrows():
                # Add park1
                unique_parks.append({
                    'unique_id': row['park1_unique_id'],
                    'point_id': row['park1_point_id'],
                    'geometry': row['park1_geom']
                })
                # Add park2
                unique_parks.append({
                    'unique_id': row['park2_unique_id'],
                    'point_id': row['park2_point_id'],
                    'geometry': row['park2_geom']
                })
            
            # Convert to GeoDataFrame and drop duplicates
            parks_gdf = gpd.GeoDataFrame(unique_parks)
            parks_gdf = parks_gdf.drop_duplicates(subset=['unique_id'])
            
            # Spatial join with boroughs
            parks_with_boroughs = gpd.sjoin(
                parks_gdf, 
                self.boroughs[['boro_name', 'geometry']], 
                how='left', 
                predicate='within'
            )
            
            # Create park to borough mapping
            self.park_to_borough = pd.Series(
                parks_with_boroughs['boro_name'].values,
                index=parks_with_boroughs['unique_id']
            )
            
            # Update ID mappings
            for _, row in parks_gdf.iterrows():
                self.park_id_mapping[row['unique_id']] = row['point_id']
            
            self.logger.info(f"Loaded {len(parks_gdf)} unique parks across {len(self.boroughs)} boroughs")
                
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def create_graph(self, distance_threshold):
        """Create graph for given distance threshold."""
        graph = nx.Graph()
        
        # Add all parks as nodes
        unique_parks = set(self.park_id_mapping.keys())
        graph.add_nodes_from(unique_parks)
        
        # Add edges for parks within threshold
        edge_count = 0
        for _, row in self.paths_gdf.iterrows():
            if row['path_length'] <= distance_threshold:
                graph.add_edge(
                    row['park1_unique_id'],
                    row['park2_unique_id'],
                    weight=row['path_length']
                )
                edge_count += 1
        
        self.logger.info(f"Created graph with {len(unique_parks)} nodes and {edge_count} edges")
        return graph

    def find_clusters(self, graph):
        """Find clusters using the Girvan-Newman algorithm."""
        try:
            self.logger.info("Finding clusters using Girvan-Newman algorithm...")
            
            # Create a copy of the graph since G-N algorithm modifies it
            graph_copy = graph.copy()
            
            # Run Girvan-Newman
            communities = list(nx.community.girvan_newman(graph_copy))
            
            # We want the clustering with highest modularity
            best_modularity = -1
            best_clustering = None
            
            for clustering in communities:
                # Convert clustering to dictionary format
                cluster_dict = {}
                for cluster_idx, cluster in enumerate(clustering):
                    for node in cluster:
                        cluster_dict[node] = cluster_idx
                        
                # Calculate modularity
                modularity = nx.community.modularity(graph, clustering)
                
                if modularity > best_modularity:
                    best_modularity = modularity
                    best_clustering = cluster_dict
            
            self.logger.info(f"Found optimal clustering with modularity: {best_modularity:.3f}")
            return best_clustering
            
        except Exception as e:
            self.logger.error(f"Error in cluster analysis: {e}")
            raise

    def analyze_clusters_by_borough(self, clusters):
        """Analyze clusters by borough and identify cross-borough clusters."""
        # Convert clusters to DataFrame
        cluster_df = pd.DataFrame.from_dict(clusters, orient='index', columns=['cluster_id'])
        cluster_df['borough'] = cluster_df.index.map(self.park_to_borough)
        
        # Get boroughs for each cluster
        cluster_boroughs = defaultdict(set)
        for park_id, cluster_id in clusters.items():
            borough = self.park_to_borough.get(park_id)
            if borough is not None:
                cluster_boroughs[cluster_id].add(borough)
        
        # Identify multi-borough clusters
        multiborough_clusters = {
            cluster_id: boroughs 
            for cluster_id, boroughs in cluster_boroughs.items() 
            if len(boroughs) > 1
        }
        
        # Calculate statistics
        stats = {
            'citywide': self._calculate_cluster_stats(cluster_df, exclude_singletons=True),
            'boroughs': {},
            'multiborough': self._calculate_multiborough_stats(
                cluster_df, multiborough_clusters, exclude_singletons=True
            )
        }
        
        # Calculate borough-specific stats
        for borough in self.park_to_borough.unique():
            if pd.isna(borough):
                continue
            borough_clusters = cluster_df[cluster_df['borough'] == borough]
            stats['boroughs'][borough] = self._calculate_cluster_stats(
                borough_clusters, exclude_singletons=True
            )
            
        return stats

    def _calculate_cluster_stats(self, cluster_df, exclude_singletons=True):
        """Calculate basic cluster statistics."""
        # Count parks per cluster
        cluster_sizes = cluster_df.groupby('cluster_id').size()
        
        if exclude_singletons:
            cluster_sizes = cluster_sizes[cluster_sizes > 1]
        
        if len(cluster_sizes) == 0:
            return {
                'total_clusters': 0,
                'avg_parks_per_cluster': 0,
                'total_parks_in_clusters': 0
            }
        
        return {
            'total_clusters': len(cluster_sizes),
            'avg_parks_per_cluster': cluster_sizes.mean(),
            'total_parks_in_clusters': cluster_sizes.sum()
        }

    def _calculate_multiborough_stats(self, cluster_df, multiborough_clusters, exclude_singletons=True):
        """Calculate statistics for multi-borough clusters."""
        if not multiborough_clusters:
            return {
                'total_clusters': 0,
                'avg_parks_per_cluster': 0,
                'total_parks': 0,
                'avg_boroughs_per_cluster': 0
            }
            
        multi_cluster_sizes = cluster_df[
            cluster_df['cluster_id'].isin(multiborough_clusters.keys())
        ].groupby('cluster_id').size()
        
        if exclude_singletons:
            multi_cluster_sizes = multi_cluster_sizes[multi_cluster_sizes > 1]
            
        return {
            'total_clusters': len(multi_cluster_sizes),
            'avg_parks_per_cluster': multi_cluster_sizes.mean(),
            'total_parks': multi_cluster_sizes.sum(),
            'avg_boroughs_per_cluster': np.mean([
                len(boroughs) for boroughs in multiborough_clusters.values()
            ])
        }

    def analyze_multiple_thresholds(self, thresholds, output_path):
        """Analyze clusters for multiple distance thresholds."""
        self.load_data()
        
        results = {}
        for threshold in thresholds:
            self.logger.info(f"\nAnalyzing clusters for {threshold}m threshold...")
            
            # Create graph and find clusters
            graph = self.create_graph(threshold)
            clusters = self.find_clusters(graph)
            
            # Analyze clusters
            stats = self.analyze_clusters_by_borough(clusters)
            results[threshold] = stats
            
        # Save comprehensive summary
        self.save_summary(results, output_path)
        
    def save_summary(self, results, output_path):
        """Save comprehensive analysis summary."""
        with open(output_path, 'w') as f:
            f.write("Park Cluster Analysis Summary\n")
            f.write("============================\n\n")
            
            for threshold, stats in results.items():
                f.write(f"\nResults for {threshold}m threshold:\n")
                f.write("-" * 30 + "\n\n")
                
                # Citywide stats
                f.write("Citywide Statistics:\n")
                citywide = stats['citywide']
                f.write(f"- Total clusters: {citywide['total_clusters']}\n")
                f.write(f"- Average parks per cluster: {citywide['avg_parks_per_cluster']:.1f}\n")
                f.write(f"- Total parks in clusters: {citywide['total_parks_in_clusters']}\n\n")
                
                # Borough stats
                f.write("Borough Statistics:\n")
                for borough, borough_stats in stats['boroughs'].items():
                    f.write(f"\n{borough}:\n")
                    f.write(f"- Total clusters: {borough_stats['total_clusters']}\n")
                    f.write(f"- Average parks per cluster: {borough_stats['avg_parks_per_cluster']:.1f}\n")
                    f.write(f"- Total parks in clusters: {borough_stats['total_parks_in_clusters']}\n")
                
                # Multi-borough stats
                f.write("\nMulti-borough Clusters:\n")
                multi = stats['multiborough']
                f.write(f"- Total clusters: {multi['total_clusters']}\n")
                f.write(f"- Average parks per cluster: {multi['avg_parks_per_cluster']:.1f}\n")
                f.write(f"- Total parks: {multi['total_parks']}\n")
                f.write(f"- Average boroughs per cluster: {multi['avg_boroughs_per_cluster']:.1f}\n")
                
                f.write("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/logs/cluster_analysis.log'),
            logging.StreamHandler()
        ]
    )
    
    # Set up file paths
    results_path = '../../pathfinding/outputs/results.gpkg'
    boroughs_path = '../../data/borough_boundaries_26918.gpkg'
    output_path = '../outputs/cluster_analysis_summary.txt'
    
    # Define thresholds to analyze
    thresholds = [100, 250, 500, 1000]
    
    print(f"\nAnalyzing clusters for thresholds: {thresholds} meters")
    print(f"Results will be saved to: {output_path}")
    
    # Run analysis
    analyzer = ParkClusterAnalyzer(
        results_path=results_path,
        boroughs_path=boroughs_path
    )
    
    analyzer.analyze_multiple_thresholds(thresholds, output_path)
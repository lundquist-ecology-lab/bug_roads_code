import geopandas as gpd
import networkx as nx
import pandas as pd
from pathlib import Path
import logging
from collections import defaultdict
import numpy as np
from tqdm.auto import tqdm
from scipy import stats

class ParkClusterAnalyzer:
    def __init__(self, results_path, boroughs_path, parks_path=None):
        """
        Initialize the park cluster analyzer.
        
        Args:
            results_path: Path to the results.gpkg file from PathFinder
            boroughs_path: Path to the borough boundaries file
            parks_path: Optional path to parks data file (if park geometries stored separately)
        """
        self.logger = logging.getLogger(__name__)
        self.results_path = Path(results_path)
        self.boroughs_path = Path(boroughs_path)
        self.parks_path = Path(parks_path) if parks_path else None
        self.paths_gdf = None
        self.boroughs = None
        self.parks_gdf = None
        self.park_to_borough = None
        
    def load_data(self):
        """Load the path results and borough data."""
        self.logger.info("Loading data...")
        
        try:
            # Load the results from PathFinder
            self.paths_gdf = gpd.read_file(self.results_path)
            
            # Debug: Print available columns
            self.logger.info(f"Available columns: {self.paths_gdf.columns.tolist()}")
            self.logger.info(f"Data shape: {self.paths_gdf.shape}")
            
            # Load borough boundaries
            self.boroughs = gpd.read_file(self.boroughs_path)
            
            # Extract unique parks and their representative locations
            self._extract_parks_from_paths()
            
            # Assign parks to boroughs
            self._assign_parks_to_boroughs()
                
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def _extract_parks_from_paths(self):
        """Extract park locations from path start/end points using point_ids."""
        self.logger.info("Extracting park points from path geometries...")
        
        # Collect coordinates for each unique park ID
        park_coords_collection = defaultdict(list)
        
        for _, row in self.paths_gdf.iterrows():
            path_geom = row['geometry']
            
            if hasattr(path_geom, 'coords'):
                # LineString geometry
                coords = list(path_geom.coords)
                start_point = coords[0]
                end_point = coords[-1]
                
                # Collect coordinates for each park using point_id
                park1_id = row['park1_point_id']
                park2_id = row['park2_point_id']
                
                park_coords_collection[park1_id].append(start_point)
                park_coords_collection[park2_id].append(end_point)
        
        # Create park points using centroid of all coordinate occurrences
        from shapely.geometry import Point
        
        parks_data = []
        for point_id, coords_list in park_coords_collection.items():
            # Calculate centroid of all coordinate occurrences for this park
            coords_array = np.array(coords_list)
            centroid_coords = coords_array.mean(axis=0)
            
            parks_data.append({
                'point_id': point_id,
                'geometry': Point(centroid_coords),
                'coord_count': len(coords_list)  # For debugging
            })
        
        # Create GeoDataFrame
        self.parks_gdf = gpd.GeoDataFrame(parks_data, crs=self.paths_gdf.crs)
        self.logger.info(f"Extracted {len(self.parks_gdf)} unique parks from {len(self.paths_gdf)} path records")
        
        # Log statistics about coordinate variations
        coord_counts = [p['coord_count'] for p in parks_data]
        self.logger.info(f"Average paths per park: {np.mean(coord_counts):.1f}, Max: {max(coord_counts)}")
    
    def _assign_parks_to_boroughs(self):
        """Assign parks to boroughs using spatial join."""
        if self.boroughs is None:
            self.logger.warning("No borough data available")
            return
            
        # Ensure CRS matches between parks and boroughs
        if self.parks_gdf.crs != self.boroughs.crs:
            self.logger.info(f"Converting parks CRS from {self.parks_gdf.crs} to {self.boroughs.crs}")
            self.parks_gdf = self.parks_gdf.to_crs(self.boroughs.crs)
        
        # Spatial join with boroughs
        parks_with_boroughs = gpd.sjoin(
            self.parks_gdf, 
            self.boroughs[['boro_name', 'geometry']], 
            how='left', 
            predicate='within'
        )
        
        # Debug: Check for duplicates after spatial join
        self.logger.info(f"Parks before spatial join: {len(self.parks_gdf)}")
        self.logger.info(f"Parks after spatial join: {len(parks_with_boroughs)}")
        
        # Handle potential duplicates from boundary cases
        if parks_with_boroughs['point_id'].duplicated().any():
            num_duplicates = parks_with_boroughs['point_id'].duplicated().sum()
            self.logger.warning(f"Found {num_duplicates} duplicate park assignments - keeping first assignment")
            parks_with_boroughs = parks_with_boroughs.drop_duplicates(subset=['point_id'], keep='first')
        
        # Create park to borough mapping
        self.park_to_borough = parks_with_boroughs.set_index('point_id')['boro_name']
        
        # Log borough assignment stats
        borough_counts = self.park_to_borough.value_counts()
        self.logger.info(f"Parks per borough: {borough_counts.to_dict()}")
        
        unassigned = self.park_to_borough.isna().sum()
        if unassigned > 0:
            self.logger.warning(f"{unassigned} parks could not be assigned to a borough")

    def create_graph(self, distance_threshold):
        """Create graph for given distance threshold using point_ids directly."""
        graph = nx.Graph()
        
        # Get all unique parks from the paths data
        unique_parks = set()
        unique_parks.update(self.paths_gdf['park1_point_id'])
        unique_parks.update(self.paths_gdf['park2_point_id'])
        
        # Add all parks as nodes
        graph.add_nodes_from(unique_parks)
        
        # Add edges for parks within threshold
        edge_count = 0
        for _, row in self.paths_gdf.iterrows():
            if row['path_length'] <= distance_threshold:
                graph.add_edge(
                    row['park1_point_id'],
                    row['park2_point_id'],
                    weight=row['path_length']
                )
                edge_count += 1
        
        self.logger.info(f"Created graph with {len(unique_parks)} nodes and {edge_count} edges")
        return graph

    def find_clusters(self, graph):
        """Find clusters using connected components or Girvan-Newman algorithm."""
        try:
            self.logger.info("Finding clusters...")
            
            if graph.number_of_edges() == 0:
                self.logger.warning("Graph has no edges - no clusters can be formed")
                return {}
            
            # For large graphs, use connected components instead of Girvan-Newman
            if graph.number_of_nodes() > 500:
                self.logger.info("Using connected components for large graph")
                components = list(nx.connected_components(graph))
                
                # Convert to cluster dictionary
                clusters = {}
                for cluster_id, component in enumerate(components):
                    for node in component:
                        clusters[node] = cluster_id
                        
                self.logger.info(f"Found {len(components)} connected components")
                return clusters
            
            else:
                self.logger.info("Using Girvan-Newman algorithm for detailed clustering")
                return self._girvan_newman_clustering(graph)
            
        except Exception as e:
            self.logger.error(f"Error in cluster analysis: {e}")
            # Fall back to connected components
            self.logger.info("Falling back to connected components")
            components = list(nx.connected_components(graph))
            clusters = {}
            for cluster_id, component in enumerate(components):
                for node in component:
                    clusters[node] = cluster_id
            return clusters
    
    def _girvan_newman_clustering(self, graph):
        """Apply Girvan-Newman clustering algorithm."""
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

    def _calculate_mean_and_se(self, values):
        """Calculate mean and standard error for a list of values."""
        if len(values) == 0:
            return 0, 0
        elif len(values) == 1:
            return values[0], 0
        else:
            mean_val = np.mean(values)
            se_val = stats.sem(values)  # Standard error of the mean
            return mean_val, se_val

    def analyze_clusters_by_borough(self, clusters):
        """Analyze clusters by borough and identify cross-borough clusters."""
        if not clusters:
            return {
                'citywide': {'total_clusters': 0, 'avg_parks_per_cluster': 0, 'se_parks_per_cluster': 0, 
                           'cluster_size_formatted': '0 ± 0 (0)', 'total_parks_in_clusters': 0},
                'boroughs': {},
                'multiborough': {'total_clusters': 0, 'avg_parks_per_cluster': 0, 'se_parks_per_cluster': 0,
                               'cluster_size_formatted': '0 ± 0 (0)', 'total_parks': 0, 'avg_boroughs_per_cluster': 0}
            }
        
        # Convert clusters to DataFrame - clusters already use point_ids as keys
        cluster_df = pd.DataFrame.from_dict(clusters, orient='index', columns=['cluster_id'])
        cluster_df.index.name = 'point_id'
        
        # Map to boroughs using point_ids directly
        if self.park_to_borough is not None:
            cluster_df['borough'] = cluster_df.index.map(self.park_to_borough)
        else:
            cluster_df['borough'] = None
        
        # Get boroughs for each cluster
        cluster_boroughs = defaultdict(set)
        for point_id, cluster_id in clusters.items():
            borough = self.park_to_borough.get(point_id) if self.park_to_borough is not None else None
            if borough is not None and pd.notna(borough):
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
        if self.park_to_borough is not None:
            for borough in self.park_to_borough.unique():
                if pd.isna(borough):
                    continue
                borough_clusters = cluster_df[cluster_df['borough'] == borough]
                stats['boroughs'][borough] = self._calculate_cluster_stats(
                    borough_clusters, exclude_singletons=True
                )
            
        return stats

    def _calculate_cluster_stats(self, cluster_df, exclude_singletons=True):
        """Calculate basic cluster statistics with mean and SE."""
        # Count parks per cluster
        cluster_sizes = cluster_df.groupby('cluster_id').size()
        
        if exclude_singletons:
            cluster_sizes = cluster_sizes[cluster_sizes > 1]
        
        if len(cluster_sizes) == 0:
            return {
                'total_clusters': 0,
                'avg_parks_per_cluster': 0,
                'se_parks_per_cluster': 0,
                'cluster_size_formatted': '0 ± 0 (0)',
                'total_parks_in_clusters': 0
            }
        
        # Calculate mean and SE
        cluster_size_values = cluster_sizes.values
        mean_size, se_size = self._calculate_mean_and_se(cluster_size_values)
        total_clusters = len(cluster_sizes)
        
        return {
            'total_clusters': total_clusters,
            'avg_parks_per_cluster': mean_size,
            'se_parks_per_cluster': se_size,
            'cluster_size_formatted': f"{mean_size:.1f} ± {se_size:.2f} ({total_clusters})",
            'total_parks_in_clusters': cluster_sizes.sum()
        }

    def _calculate_multiborough_stats(self, cluster_df, multiborough_clusters, exclude_singletons=True):
        """Calculate statistics for multi-borough clusters with mean and SE."""
        if not multiborough_clusters:
            return {
                'total_clusters': 0,
                'avg_parks_per_cluster': 0,
                'se_parks_per_cluster': 0,
                'cluster_size_formatted': '0 ± 0 (0)',
                'total_parks': 0,
                'avg_boroughs_per_cluster': 0
            }
            
        multi_cluster_sizes = cluster_df[
            cluster_df['cluster_id'].isin(multiborough_clusters.keys())
        ].groupby('cluster_id').size()
        
        if exclude_singletons:
            multi_cluster_sizes = multi_cluster_sizes[multi_cluster_sizes > 1]
            
        if len(multi_cluster_sizes) == 0:
            return {
                'total_clusters': 0,
                'avg_parks_per_cluster': 0,
                'se_parks_per_cluster': 0,
                'cluster_size_formatted': '0 ± 0 (0)',
                'total_parks': 0,
                'avg_boroughs_per_cluster': 0
            }
        
        # Calculate mean and SE for cluster sizes
        cluster_size_values = multi_cluster_sizes.values
        mean_size, se_size = self._calculate_mean_and_se(cluster_size_values)
        total_clusters = len(multi_cluster_sizes)
            
        return {
            'total_clusters': total_clusters,
            'avg_parks_per_cluster': mean_size,
            'se_parks_per_cluster': se_size,
            'cluster_size_formatted': f"{mean_size:.1f} ± {se_size:.2f} ({total_clusters})",
            'total_parks': multi_cluster_sizes.sum(),
            'avg_boroughs_per_cluster': np.mean([
                len(boroughs) for boroughs in multiborough_clusters.values()
            ])
        }

    def analyze_multiple_thresholds(self, thresholds, output_dir="../outputs"):
        """Analyze clusters for multiple distance thresholds."""
        self.load_data()
        
        # Ensure output directories exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        clusters_dir = output_path / "dijkstra_clusters"
        clusters_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        for threshold in thresholds:
            self.logger.info(f"\nAnalyzing clusters for {threshold}m threshold...")
            
            # Create graph and find clusters
            graph = self.create_graph(threshold)
            clusters = self.find_clusters(graph)
            
            # Save individual cluster file
            if clusters:
                # Create output dataframe with borough information
                cluster_data = []
                for point_id, cluster_id in clusters.items():
                    borough = self.park_to_borough.get(point_id) if self.park_to_borough is not None else None
                    cluster_data.append({
                        'point_id': point_id,
                        'cluster_id': cluster_id,
                        'borough': borough
                    })
                
                cluster_df = pd.DataFrame(cluster_data)
                cluster_file = clusters_dir / f"dijkstra_clusters_{threshold}m.csv"
                cluster_df.to_csv(cluster_file, index=False)
                self.logger.info(f"Cluster assignments for {threshold}m saved to {cluster_file}")
            
            # Analyze clusters
            stats = self.analyze_clusters_by_borough(clusters)
            results[threshold] = stats
            
        # Save comprehensive summary
        summary_file = output_path / "dijkstra_analysis_summary.txt"
        self.save_summary(results, summary_file)
        
    def save_summary(self, results, output_path):
        """Save comprehensive analysis summary with mean ± SE format."""
        with open(output_path, 'w') as f:
            f.write("Park Cluster Analysis Summary (Dijkstra Paths)\n")
            f.write("==============================================\n\n")
            
            for threshold, stats in results.items():
                f.write(f"\nResults for {threshold}m threshold:\n")
                f.write("-" * 30 + "\n\n")
                
                # Citywide stats
                f.write("Citywide Statistics:\n")
                citywide = stats['citywide']
                f.write(f"- Total clusters: {citywide['total_clusters']}\n")
                f.write(f"- Cluster size: {citywide['cluster_size_formatted']}\n")
                f.write(f"- Total parks in clusters: {citywide['total_parks_in_clusters']}\n\n")
                
                # Borough stats
                if stats['boroughs']:
                    f.write("Borough Statistics:\n")
                    for borough, borough_stats in stats['boroughs'].items():
                        f.write(f"\n{borough}:\n")
                        f.write(f"- Total clusters: {borough_stats['total_clusters']}\n")
                        f.write(f"- Cluster size: {borough_stats['cluster_size_formatted']}\n")
                        f.write(f"- Total parks in clusters: {borough_stats['total_parks_in_clusters']}\n")
                else:
                    f.write("Borough Statistics: No borough data available\n")
                
                # Multi-borough stats
                f.write("\nMulti-borough Clusters:\n")
                multi = stats['multiborough']
                f.write(f"- Total clusters: {multi['total_clusters']}\n")
                f.write(f"- Cluster size: {multi['cluster_size_formatted']}\n")
                f.write(f"- Total parks: {multi['total_parks']}\n")
                f.write(f"- Average boroughs per cluster: {multi['avg_boroughs_per_cluster']:.1f}\n")
                
                f.write("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('../logs/cluster_analysis.log'),
            logging.StreamHandler()
        ]
    )
    
    # Set up file paths
    results_path = '../../pathfinding/outputs/results.gpkg'
    boroughs_path = '../../data/borough_boundaries_26918.gpkg'
    parks_path = None  # Not needed with this approach
    
    # Define thresholds to analyze
    thresholds = [100, 250, 500, 1000]
    
    print(f"\nAnalyzing clusters for thresholds: {thresholds} meters")
    
    # Run analysis
    analyzer = ParkClusterAnalyzer(
        results_path=results_path,
        boroughs_path=boroughs_path,
        parks_path=parks_path
    )
    
    analyzer.analyze_multiple_thresholds(thresholds, "../outputs")

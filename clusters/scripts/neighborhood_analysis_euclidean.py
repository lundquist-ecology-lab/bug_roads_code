import geopandas as gpd
import networkx as nx
from pathlib import Path
import logging
from collections import defaultdict
import pandas as pd
import numpy as np
from networkx.algorithms.community import girvan_newman
from scipy import stats

class EuclideanClusterAnalyzer:
    def __init__(self, results_path, boroughs_path=None):
        """
        Initialize the Euclidean cluster analyzer.

        Args:
            results_path: Path to the results.gpkg file containing path data
            boroughs_path: Optional path to borough boundaries for analysis
        """
        self.logger = logging.getLogger(__name__)
        self.results_path = Path(results_path)
        self.boroughs_path = Path(boroughs_path) if boroughs_path else None
        self.parks_gdf = None
        self.paths_gdf = None
        self.boroughs = None
        self.park_to_borough = None

    def load_data(self):
        """Load the path data and extract park points."""
        try:
            self.logger.info("Loading path data...")
            self.paths_gdf = gpd.read_file(self.results_path)
            self.logger.info(f"Loaded {len(self.paths_gdf)} path records.")
            self.logger.info(f"Available columns: {self.paths_gdf.columns.tolist()}")
            
            # Extract unique park points from path data
            self._extract_parks_from_paths()
            
            # Load borough data if provided
            if self.boroughs_path and self.boroughs_path.exists():
                self.boroughs = gpd.read_file(self.boroughs_path)
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
                
                # Collect coordinates for each park using unique_d
                park1_id = row['park1_point_id']
                park2_id = row['park2_point_id']
                
                park_coords_collection[park1_id].append(start_point)
                park_coords_collection[park2_id].append(end_point)
        
        # Debug: Check what we're getting
        self.logger.info(f"Found {len(park_coords_collection)} unique park IDs from paths")
        
        # Get unique parks from paths data directly for verification
        unique_parks_from_paths = set()
        unique_parks_from_paths.update(self.paths_gdf['park1_point_id'])
        unique_parks_from_paths.update(self.paths_gdf['park2_point_id'])
        self.logger.info(f"Unique parks from path columns: {len(unique_parks_from_paths)}")
        
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
                'coord_count': len(coords_list)
            })
        
        # Create GeoDataFrame
        self.parks_gdf = gpd.GeoDataFrame(parks_data, crs=self.paths_gdf.crs)
        self.logger.info(f"Created parks GDF with {len(self.parks_gdf)} parks")
        self.logger.info(f"Extracted {len(self.parks_gdf)} unique parks from {len(self.paths_gdf)} path records")
        
        # Verify no duplicates
        duplicates = self.parks_gdf['point_id'].duplicated().sum()
        if duplicates > 0:
            self.logger.error(f"ERROR: Found {duplicates} duplicate point_ids in parks_gdf!")
        
        # Log statistics
        coord_counts = [p['coord_count'] for p in parks_data]
        self.logger.info(f"Average paths per park: {np.mean(coord_counts):.1f}, Max: {max(coord_counts)}")
        
        # Debug: Check for any weird point_ids
        sample_ids = list(park_coords_collection.keys())[:10]
        self.logger.info(f"Sample point_ids: {sample_ids}")

    def _assign_parks_to_boroughs(self):
        """Assign parks to boroughs using spatial join."""
        if self.boroughs is None:
            self.logger.warning("No borough data available")
            return
            
        # Ensure CRS matches
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
        
        # Handle potential duplicates from boundary cases
        if parks_with_boroughs['point_id'].duplicated().any():
            self.logger.warning("Found parks on borough boundaries, keeping first assignment")
            parks_with_boroughs = parks_with_boroughs.drop_duplicates(subset=['point_id'], keep='first')
        
        # Create borough mapping
        self.park_to_borough = parks_with_boroughs.set_index('point_id')['boro_name']
        
        # Log borough assignment stats
        borough_counts = self.park_to_borough.value_counts()
        self.logger.info(f"Parks per borough: {borough_counts.to_dict()}")
        
        unassigned = self.park_to_borough.isna().sum()
        if unassigned > 0:
            self.logger.warning(f"{unassigned} parks could not be assigned to a borough")

    def create_graph(self, distance_threshold):
        """Create a graph based on Euclidean distances between park points within the threshold."""
        graph = nx.Graph()

        # Add all parks as nodes using point_id as the node identifier
        for idx, park in self.parks_gdf.iterrows():
            graph.add_node(
                park['point_id'],  # Use point_id as node identifier
                geometry=park.geometry,
                borough=self.park_to_borough.get(park['point_id']) if self.park_to_borough is not None else None
            )

        # Get geometries and point_ids for distance calculations
        geometries = self.parks_gdf.geometry.to_numpy()
        point_ids = self.parks_gdf['point_id'].to_numpy()

        # Calculate distances and add edges for parks within the threshold
        edge_count = 0
        for i, geom1 in enumerate(geometries):
            for j, geom2 in enumerate(geometries):
                if i >= j:
                    continue
                dist = geom1.distance(geom2)  # Euclidean distance between points
                if dist <= distance_threshold:
                    graph.add_edge(point_ids[i], point_ids[j], weight=dist)
                    edge_count += 1

        self.logger.info(f"Created graph with {len(graph.nodes)} nodes and {edge_count} edges for {distance_threshold}m threshold.")
        return graph

    def find_clusters_girvan_newman(self, graph):
        """Find clusters using the Girvan-Newman algorithm or connected components for large graphs."""
        try:
            if graph.number_of_edges() == 0:
                self.logger.warning("Graph has no edges - no clusters can be formed")
                return {}
            
            self.logger.info(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            
            # For large graphs, use connected components instead of Girvan-Newman
            if graph.number_of_nodes() > 500:
                self.logger.info("Using connected components for large graph")
                components = list(nx.connected_components(graph))
                
                cluster_mapping = {}
                for cluster_id, component in enumerate(components):
                    for node in component:
                        cluster_mapping[node] = cluster_id
                        
                self.logger.info(f"Found {len(components)} connected components")
                
                # Debug: Log component sizes
                component_sizes = [len(comp) for comp in components]
                self.logger.info(f"Component size stats: min={min(component_sizes)}, max={max(component_sizes)}, mean={np.mean(component_sizes):.1f}")
                multi_node_components = [size for size in component_sizes if size > 1]
                self.logger.info(f"Multi-node components: {len(multi_node_components)} out of {len(components)} total")
                
                return cluster_mapping
            
            else:
                self.logger.info("Using Girvan-Newman algorithm")
                communities_generator = girvan_newman(graph)
                top_level_communities = next(communities_generator)
                
                cluster_mapping = {}
                for cluster_id, community in enumerate(top_level_communities):
                    for node in community:
                        cluster_mapping[node] = cluster_id
                        
                self.logger.info(f"Found {len(top_level_communities)} clusters using Girvan-Newman.")
                return cluster_mapping
                
        except Exception as e:
            self.logger.error(f"Error during clustering: {e}")
            # Fall back to connected components
            self.logger.info("Falling back to connected components")
            components = list(nx.connected_components(graph))
            cluster_mapping = {}
            for cluster_id, component in enumerate(components):
                for node in component:
                    cluster_mapping[node] = cluster_id
            return cluster_mapping

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

    def analyze_clusters_by_borough(self, clusters, graph):
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
        
        self.logger.debug(f"Total clusters before filtering: {len(cluster_sizes)}")
        self.logger.debug(f"Cluster size distribution: {cluster_sizes.value_counts().sort_index().to_dict()}")
        
        if exclude_singletons:
            cluster_sizes = cluster_sizes[cluster_sizes > 1]
        
        self.logger.debug(f"Multi-park clusters after filtering: {len(cluster_sizes)}")
        
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
        
        self.logger.debug(f"Final stats: {total_clusters} clusters, mean size: {mean_size:.1f}")
        
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

    def analyze_multiple_thresholds(self, thresholds, output_dir="../outputs/euclidean_clusters"):
        """Analyze clusters for multiple distance thresholds."""
        self.load_data()
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Store results for summary
        all_results = {}
        
        for threshold in thresholds:
            self.logger.info(f"\nAnalyzing clusters for {threshold}m threshold...")

            graph = self.create_graph(distance_threshold=threshold)
            clusters = self.find_clusters_girvan_newman(graph)
            
            self.logger.info(f"Total clusters found: {len(set(clusters.values())) if clusters else 0}")
            self.logger.info(f"Total parks in clusters: {len(clusters)}")
            
            # Sanity check
            if len(clusters) > len(self.parks_gdf):
                self.logger.error(f"ERROR: More clusters ({len(clusters)}) than parks ({len(self.parks_gdf)})!")
            
            unique_cluster_ids = len(set(clusters.values())) if clusters else 0
            if unique_cluster_ids > len(self.parks_gdf):
                self.logger.error(f"ERROR: More unique cluster IDs ({unique_cluster_ids}) than parks ({len(self.parks_gdf)})!")

            # Analyze by borough if possible
            borough_stats = self.analyze_clusters_by_borough(clusters, graph)

            # Save cluster results for this threshold
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
                output_csv_path = output_path / f"euclidean_clusters_{threshold}m.csv"
                cluster_df.to_csv(output_csv_path, index=False)
                self.logger.info(f"Cluster assignments for {threshold}m saved to {output_csv_path}")
                
                # Store results for summary
                all_results[threshold] = {
                    'total_clusters': len(set(clusters.values())),
                    'total_parks': len(clusters),
                    'borough_stats': borough_stats
                }
            else:
                self.logger.warning(f"No clusters found for {threshold}m threshold")
                all_results[threshold] = {
                    'total_clusters': 0,
                    'total_parks': 0,
                    'borough_stats': {
                        'citywide': {'total_clusters': 0, 'avg_parks_per_cluster': 0, 'se_parks_per_cluster': 0, 
                                   'cluster_size_formatted': '0 ± 0 (0)', 'total_parks_in_clusters': 0},
                        'boroughs': {},
                        'multiborough': {'total_clusters': 0, 'avg_parks_per_cluster': 0, 'se_parks_per_cluster': 0,
                                       'cluster_size_formatted': '0 ± 0 (0)', 'total_parks': 0, 'avg_boroughs_per_cluster': 0}
                    }
                }
        
        # Save summary
        self._save_summary(all_results, output_path / "euclidean_analysis_summary.txt")

    def _save_summary(self, results, output_path):
        """Save comprehensive analysis summary with mean ± SE format."""
        with open(output_path, 'w') as f:
            f.write("Euclidean Distance Cluster Analysis Summary\n")
            f.write("==========================================\n\n")
            
            for threshold, stats in results.items():
                f.write(f"\nResults for {threshold}m threshold:\n")
                f.write("-" * 30 + "\n\n")
                
                # Citywide stats
                f.write("Citywide Statistics:\n")
                citywide = stats['borough_stats']['citywide']
                f.write(f"- Total clusters: {citywide['total_clusters']}\n")
                f.write(f"- Cluster size: {citywide['cluster_size_formatted']}\n")
                f.write(f"- Total parks in clusters: {citywide['total_parks_in_clusters']}\n\n")
                
                # Borough stats
                if stats['borough_stats']['boroughs']:
                    f.write("Borough Statistics:\n")
                    for borough, borough_stats in stats['borough_stats']['boroughs'].items():
                        f.write(f"\n{borough}:\n")
                        f.write(f"- Total clusters: {borough_stats['total_clusters']}\n")
                        f.write(f"- Cluster size: {borough_stats['cluster_size_formatted']}\n")
                        f.write(f"- Total parks in clusters: {borough_stats['total_parks_in_clusters']}\n")
                else:
                    f.write("Borough Statistics: No borough data available\n")
                
                # Multi-borough stats
                f.write("\nMulti-borough Clusters:\n")
                multi = stats['borough_stats']['multiborough']
                f.write(f"- Total clusters: {multi['total_clusters']}\n")
                f.write(f"- Cluster size: {multi['cluster_size_formatted']}\n")
                f.write(f"- Total parks: {multi['total_parks']}\n")
                f.write(f"- Average boroughs per cluster: {multi['avg_boroughs_per_cluster']:.1f}\n")
                
                f.write("\n" + "=" * 50 + "\n")
        
        self.logger.info(f"Analysis summary saved to {output_path}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('../logs/euclidean_cluster_analysis.log'),
            logging.StreamHandler()
        ]
    )

    # File paths
    results_path = '../../pathfinding/outputs/results.gpkg'
    boroughs_path = '../../data/borough_boundaries_26918.gpkg'  # Optional

    # Distance thresholds in meters - smaller values for Euclidean analysis
    thresholds = [100, 250, 500, 1000]

    print(f"\nAnalyzing Euclidean clusters for thresholds: {thresholds} meters")

    # Run analysis
    analyzer = EuclideanClusterAnalyzer(
        results_path=results_path,
        boroughs_path=boroughs_path
    )
    analyzer.analyze_multiple_thresholds(thresholds)

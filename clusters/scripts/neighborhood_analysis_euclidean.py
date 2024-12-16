import geopandas as gpd
import networkx as nx
from pathlib import Path
import logging
from scipy.spatial import distance
import pandas as pd
from networkx.algorithms.community import girvan_newman

class EuclideanClusterAnalyzer:
    def __init__(self, results_path):
        """
        Initialize the Euclidean cluster analyzer.

        Args:
            results_path: Path to the results.gpkg file containing park points
        """
        self.logger = logging.getLogger(__name__)
        self.results_path = Path(results_path)
        self.parks_gdf = None

    def load_data(self):
        """Load the park points data."""
        try:
            self.logger.info("Loading park points data...")
            self.parks_gdf = gpd.read_file(self.results_path)
            self.logger.info(f"Loaded {len(self.parks_gdf)} park points.")
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def create_graph(self, distance_threshold):
        """Create a graph based on Euclidean distances between polygon edges within the threshold."""
        graph = nx.Graph()

        # Add all parks as nodes
        for idx, park in self.parks_gdf.iterrows():
            graph.add_node(park['point_id'], unique_id=park['unique_id'], geometry=park.geometry)

        # Get geometries and IDs
        geometries = self.parks_gdf.geometry.to_numpy()
        ids = self.parks_gdf['point_id'].to_numpy()

        # Calculate distances and add edges for parks within the threshold
        for i, geom1 in enumerate(geometries):
            for j, geom2 in enumerate(geometries):
                if i >= j:
                    continue
                dist = geom1.distance(geom2)  # Shapely's method for distance between geometries
                if dist <= distance_threshold:
                    graph.add_edge(ids[i], ids[j], weight=dist)

        self.logger.info(f"Created graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
        return graph


    def find_clusters_girvan_newman(self, graph):
        """Find clusters using the Girvan-Newman algorithm."""
        try:
            communities_generator = girvan_newman(graph)
            top_level_communities = next(communities_generator)
            cluster_mapping = {}
            for cluster_id, community in enumerate(top_level_communities):
                for node in community:
                    cluster_mapping[node] = cluster_id
            self.logger.info(f"Found {len(top_level_communities)} clusters using Girvan-Newman.")
            return cluster_mapping
        except Exception as e:
            self.logger.error(f"Error during Girvan-Newman clustering: {e}")
            raise

    def analyze_multiple_thresholds(self, thresholds):
        """Analyze clusters for multiple distance thresholds."""
        for threshold in thresholds:
            self.logger.info(f"Analyzing clusters for {threshold}m threshold...")

            graph = self.create_graph(distance_threshold=threshold)
            clusters = self.find_clusters_girvan_newman(graph)

            # Save cluster results for this threshold
            cluster_df = pd.DataFrame(
                {
                    'point_id': list(clusters.keys()),
                    'cluster_id': list(clusters.values()),
                    'unique_id': [graph.nodes[node]['unique_id'] for node in clusters.keys()]
                }
            )
            output_csv_path = f"../outputs/euclidean_clusters/euclidean_clusters_{threshold}m.csv"
            cluster_df.to_csv(output_csv_path, index=False)
            self.logger.info(f"Results for {threshold}m saved to {output_csv_path}.")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/logs/euclidean_cluster_analysis.log'),
            logging.StreamHandler()
        ]
    )

    # File paths
    results_path = '../../data/open_and_park_single_26918.gpkg'

    # Distance thresholds in meters
    thresholds = [100, 250, 500, 1000]

    # Run analysis
    analyzer = EuclideanClusterAnalyzer(results_path=results_path)
    analyzer.load_data()
    analyzer.analyze_multiple_thresholds(thresholds)

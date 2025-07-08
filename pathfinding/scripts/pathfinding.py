import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import Point, LineString
from scipy.spatial import cKDTree
from datetime import datetime, timedelta
import pickle
from multiprocessing import Manager, Pool, shared_memory
import math
from tqdm.auto import tqdm
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../log/pathfinding.log'),
        logging.StreamHandler()
    ]
)

class PathFinder:
    def __init__(self, buffer_size=100):
        self.BUFFER_SIZE = buffer_size
        self.logger = logging.getLogger(__name__)
        
    def initialize_shared_data(self):
        """Load and prepare data for sharing across processes"""
        self.logger.info("Loading network data...")
        
        # Load graph and create shared memory for coordinates
        with open('../graph/global_network_graph.pkl', 'rb') as f:
            self.graph = pickle.load(f)
            
        coords = np.load('../graph/global_network_coords.npy')
        self.shm = shared_memory.SharedMemory(create=True, size=coords.nbytes)
        self.shared_coords = np.ndarray(coords.shape, dtype=coords.dtype, buffer=self.shm.buf)
        np.copyto(self.shared_coords, coords)
        
        # Create KD-tree
        with open('../graph/global_network_kdtree.pkl', 'rb') as f:
            self.kd_tree = pickle.load(f)
            
        self.logger.info("Network data loaded successfully")

    def cleanup_shared_memory(self):
        """Clean up shared memory resources"""
        try:
            self.shm.close()
            self.shm.unlink()
        except Exception as e:
            self.logger.error(f"Error cleaning up shared memory: {e}")

    def find_nearby_park_pairs(self, parks, max_distance=1000):
        """Find pairs of parks within the specified distance using a generator."""
        self.logger.info("Finding nearby park pairs...")
        parks_sindex = parks.sindex
        
        for idx1, park1 in parks.iterrows():
            search_area = park1.geometry.buffer(max_distance)
            possible_matches_idx = list(parks_sindex.intersection(search_area.bounds))
            
            for idx2 in possible_matches_idx:
                if idx1 < idx2:  # Avoid duplicates
                    park2 = parks.iloc[idx2]
                    if park1.geometry.distance(park2.geometry) <= max_distance:
                        yield {
                            'park1_unique_id': park1['unique_id'],
                            'park1_point_id': park1['point_id'],
                            'park2_unique_id': park2['unique_id'],
                            'park2_point_id': park2['point_id'],
                            'geometry1': park1.geometry,
                            'geometry2': park2.geometry
                        }
    @staticmethod
    def find_closest_park_boundary_points(geometry1, geometry2):
        """Find the closest points between two park boundaries - efficient version"""
        from shapely.ops import nearest_points
        
        # Use Shapely's optimized nearest_points function
        try:
            # This is much more efficient than nested loops
            point1, point2 = nearest_points(geometry1, geometry2)
            return point1, point2
        except Exception:
            # Fallback to centroids if geometries are invalid
            return geometry1.centroid, geometry2.centroid
        
    @staticmethod
    def find_nearest_network_point_to_specific_point(point, kd_tree, coords):
        """Find nearest network node to a specific point"""
        point_coords = np.array([point.x, point.y])
        dist, idx = kd_tree.query(point_coords)
        return tuple(coords[idx])

    def process_single_pair(self, pair, kd_tree, coords, graph):
        """Process a single pair of parks to find the shortest path"""
        try:
            # Find closest boundary points between the two parks
            park1_boundary_point, park2_boundary_point = PathFinder.find_closest_park_boundary_points(
                pair['geometry1'], pair['geometry2']
            )
            
            # Map those specific points to network nodes
            start_point = PathFinder.find_nearest_network_point_to_specific_point(
                park1_boundary_point, kd_tree, coords
            )
            end_point = PathFinder.find_nearest_network_point_to_specific_point(
                park2_boundary_point, kd_tree, coords
            )
            
            # Validate that both points exist in the graph
            if start_point not in graph or end_point not in graph:
                self.logger.warning(f"Start or end point not in graph: {start_point}, {end_point}")
                return None
                
            try:
                path = nx.shortest_path(graph, start_point, end_point, weight='weight')
            except nx.NetworkXNoPath:
                self.logger.debug(f"No path found between points: {start_point}, {end_point}")
                return None
                
            # Validate path has at least 2 points
            if len(path) < 2:
                self.logger.warning(f"Path has insufficient points: {len(path)}")
                return None
                
            # Create points list with validation
            path_points = []
            for node in path:
                try:
                    if isinstance(node, (tuple, list)) and len(node) == 2:
                        path_points.append(Point(node))
                    else:
                        self.logger.warning(f"Invalid node format in path: {node}")
                        return None
                except Exception as e:
                    self.logger.error(f"Error creating point from node {node}: {e}")
                    return None
                    
            # Ensure we have enough points for a LineString
            if len(path_points) < 2:
                self.logger.warning("Not enough valid points to create LineString")
                return None
                
            try:
                path_geometry = LineString(path_points)
                path_length = nx.shortest_path_length(graph, start_point, end_point, weight='weight')
                
                # Calculate euclidean distance between the closest boundary points
                euclidean_distance = park1_boundary_point.distance(park2_boundary_point)
                
            except Exception as e:
                self.logger.error(f"Error creating LineString or calculating path length: {e}")
                return None
            
            return {
                'park1_unique_id': pair['park1_unique_id'],
                'park1_point_id': pair['park1_point_id'],
                'park2_unique_id': pair['park2_unique_id'],
                'park2_point_id': pair['park2_point_id'],
                'path_length': path_length,
                'euclidean_distance': euclidean_distance,
                'geometry': path_geometry
            }
        except Exception as e:
            self.logger.error(f"Error processing park pair: {e}")
            return None

    def process_batch(self, batch_data):
        """Process a batch of park pairs"""
        batch_num, pairs_batch, results_queue = batch_data
        batch_results = []
        
        try:
            for pair in pairs_batch:
                result = self.process_single_pair(pair, self.kd_tree, self.shared_coords, self.graph)
                if result:
                    batch_results.append(result)
                    
                if len(batch_results) >= self.BUFFER_SIZE:
                    results_queue.put(batch_results)
                    batch_results = []
                    
            if batch_results:
                results_queue.put(batch_results)
                
        except Exception as e:
            self.logger.error(f"Error processing batch {batch_num}: {e}")
            
        return batch_num

    def save_results(self, results, output_file):
        """Save results to a GeoPackage file, appending to existing data."""
        try:
            gdf = gpd.GeoDataFrame(results, crs="EPSG:26918")
            
            # Check if the file exists
            if Path(output_file).exists():
                gdf.to_file(output_file, driver='GPKG', mode='a')
            else:
                gdf.to_file(output_file, driver='GPKG')
            
            self.logger.info(f"Saved {len(results)} results to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")


    def run(self, start_batch=1, num_processes=4):
        """Main execution method with batch start and append functionality."""
        start_time = datetime.now()
        self.logger.info(f"Starting pathfinding process at {start_time}, starting from batch {start_batch}")
        
        try:
            # Initialize shared resources
            self.initialize_shared_data()
            
            # Load parks and find pairs
            parks = gpd.read_file('../../data/open_and_park_single_26918.gpkg')
            park_pairs = list(self.find_nearby_park_pairs(parks))
            
            # Determine total batches
            num_batches = math.ceil(len(park_pairs) / self.BUFFER_SIZE)
            self.logger.info(f"Total number of batches: {num_batches}")
            
            # Validate start_batch
            if start_batch < 1 or start_batch > num_batches:
                self.logger.error(f"Invalid start_batch: {start_batch}. Must be between 1 and {num_batches}.")
                return
            
            # Prepare batches
            manager = Manager()
            results_queue = manager.Queue()
            batches = []
            for i in range(start_batch - 1, num_batches):
                start_idx = i * self.BUFFER_SIZE
                end_idx = min((i + 1) * self.BUFFER_SIZE, len(park_pairs))
                batches.append((i + 1, park_pairs[start_idx:end_idx], results_queue))
            
            # Process batches
            with Pool(processes=num_processes) as pool:
                with tqdm(total=len(batches), desc="Processing batches") as pbar:
                    for batch_num in pool.imap_unordered(self.process_batch, batches):
                        pbar.update(1)
                        
                        # Save results from queue for completed batches
                        while not results_queue.empty():
                            batch_results = results_queue.get()
                            self.save_results(batch_results, '../outputs/results.gpkg')
                
            end_time = datetime.now()
            self.logger.info(f"Process completed at: {end_time}")
            self.logger.info(f"Total time: {end_time - start_time}")
        
        except Exception as e:
            self.logger.error(f"Error in main process: {e}")
            raise
        
        finally:
            self.cleanup_shared_memory()


if __name__ == "__main__":
    pathfinder = PathFinder(buffer_size=100)
    pathfinder.run(start_batch=1, num_processes=6)

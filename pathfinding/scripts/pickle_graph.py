import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.strtree import STRtree
from scipy.spatial import cKDTree
from tqdm import tqdm
import pickle
from multiprocessing import Pool, cpu_count
import tempfile


def extract_vertices(geom):
    """Extract vertices from a polygon or multipolygon geometry."""
    if isinstance(geom, Polygon):
        return list(geom.exterior.coords)
    elif isinstance(geom, MultiPolygon):
        all_coords = []
        for poly in geom.geoms:
            if isinstance(poly, Polygon) and not poly.is_empty:
                all_coords.extend(list(poly.exterior.coords))
        return all_coords
    else:
        return []


def process_vertex_chunk(chunk_data):
    """Process a chunk of vertices to add edges to the graph."""
    chunk, all_vertices, kd_tree, max_edge_length, building_tree, building_geometries = chunk_data
    local_edges = []
    for i, v1 in enumerate(chunk):
        idxs = kd_tree.query_ball_point(v1, max_edge_length)
        for j in idxs:
            if j > i:  # Avoid adding duplicate edges
                v2 = all_vertices[j]
                line = LineString([v1, v2])
                intersecting_idxs = building_tree.query(line)
                valid_edge = True
                for idx_bld in intersecting_idxs:
                    bld = building_geometries.iloc[idx_bld]
                    intersection = line.intersection(bld)
                    if not intersection.is_empty and intersection.geom_type not in ['Point', 'MultiPoint']:
                        valid_edge = False
                        break
                if valid_edge:
                    dist = Point(v1).distance(Point(v2))
                    local_edges.append((v1, v2, dist))
    
    # Write edges to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    with open(temp_file.name, 'wb') as f:
        pickle.dump(local_edges, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return temp_file.name

def build_global_network(grid, unioned_buildings_gdf, max_edge_length=50):
    print("Using unioned building geometries for intersection checks.")
    building_geometries = unioned_buildings_gdf.geometry
    print(f"Number of building geometry features: {len(building_geometries)}")

    print("Building STRtree for buildings...")
    building_tree = STRtree(building_geometries)

    print("Extracting grid vertices...")
    all_vertices = []
    for poly in grid.geometry:
        if poly.is_empty:
            continue
        verts = extract_vertices(poly)
        all_vertices.extend(verts)

    all_vertices = list(set(all_vertices))
    print(f"Number of unique vertices extracted from grid: {len(all_vertices)}")

    print("Building KD-tree for vertices...")
    coords = np.array(all_vertices)
    kd_tree = cKDTree(coords)

    print("Dividing vertices into chunks for parallel processing...")
    num_cores = min(cpu_count(), 12)  # Use up to 12 cores
    chunk_size = len(all_vertices) // num_cores
    vertex_chunks = [all_vertices[i:i + chunk_size] for i in range(0, len(all_vertices), chunk_size)]

    print("Building global network graph in parallel...")
    pool = Pool(processes=num_cores)
    chunk_data = [(chunk, all_vertices, kd_tree, max_edge_length, building_tree, building_geometries)
                  for chunk in vertex_chunks]

    temp_files = list(tqdm(pool.imap(process_vertex_chunk, chunk_data), total=len(chunk_data)))
    pool.close()
    pool.join()

    print("Merging results from temporary files...")
    G = nx.Graph()
    for v in all_vertices:
        G.add_node(v)

    for temp_file in temp_files:
        with open(temp_file, 'rb') as f:
            edges = pickle.load(f)
            for v1, v2, dist in edges:
                G.add_edge(v1, v2, weight=dist)
        # Delete the temporary file to save disk space
        os.unlink(temp_file)

    return G, coords, kd_tree



def main():
    parks_path = '../../data/open_and_park_single_26918.gpkg'
    grid_path = '../../data/50m_grid_clipped_with_water.gpkg'
    unioned_buildings_path = '../../data/new_york_city_buildings_union_26918.gpkg'


    print("Loading data...")
    parks = gpd.read_file(parks_path)
    grid = gpd.read_file(grid_path)
    unioned_buildings_gdf = gpd.read_file(unioned_buildings_path)

    # Ensure CRS alignment
    if not (parks.crs == grid.crs == unioned_buildings_gdf.crs):
        print("Aligning CRS...")
        common_crs = parks.crs
        if common_crs is None:
            print("Error: Unable to determine common CRS. Please set the CRS of the input files.")
            sys.exit(1)
        grid = grid.to_crs(common_crs)
        unioned_buildings_gdf = unioned_buildings_gdf.to_crs(grid.crs)

    print(f"Number of grid features (polygons): {len(grid)}")
    print(f"Number of unioned building features (polygons): {len(unioned_buildings_gdf)}")

    G, coords, kd_tree = build_global_network(grid, unioned_buildings_gdf, max_edge_length=50)

    print("Saving graph and data structures...")
    with open('../graph/global_network_graph.pkl', 'wb') as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    np.save('../graph/global_network_coords.npy', coords)

    with open('../graph/global_network_kdtree.pkl', 'wb') as f:
        pickle.dump(kd_tree, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Global network saved successfully.")


if __name__ == "__main__":
    main()

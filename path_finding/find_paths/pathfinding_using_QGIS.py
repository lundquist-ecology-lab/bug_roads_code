import os
import sys
import time
import csv
from tqdm import tqdm
from shapely.geometry import Point
from shapely.strtree import STRtree
import tempfile

# Adjust these paths to match the locations on your system. Ensure the version number is correct.
qgis_install_path = "C:/Program Files/QGIS 3.36.0"  # Adjust as needed
sys.path.extend([
    f"{qgis_install_path}/apps/qgis/python",
    f"{qgis_install_path}/apps/Python39/lib/site-packages",
    f"{qgis_install_path}/apps/qgis/python/plugins",  # Make sure this is correct
])

from qgis.core import (
    QgsApplication,
    QgsProject,
    QgsPointXY,
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsFeatureRequest,
)

# Initialize the QGIS application
QgsApplication.setPrefixPath(f"{qgis_install_path}", True)
qgs = QgsApplication([], False)
qgs.initQgis()

# Now we can safely import processing
import processing
from processing.core.Processing import Processing
Processing.initialize()

# Setup the project and layers
project = QgsProject.instance()
project.read('E:/research/bug_roads/shortest_distance/bug_roads.qgz')
parks_layer = project.mapLayersByName('nyc_parks_centroids')[0]
network_layer = project.mapLayersByName('new_york_city_area_bug_road')[0]  # Replace with your street network layer

# Print layer information for debugging
# print(f"Parks layer: {parks_layer.name()}, Feature count: {parks_layer.featureCount()}")
# print(f"Network layer: {network_layer.name()}, Feature count: {network_layer.featureCount()}")

# Measure execution time
start_time = time.time()

# Set the distance threshold (in feet)
distance_threshold = 3300

# Create a spatial index for the parks
parks = [Point(feature.geometry().asPoint().x(), feature.geometry().asPoint().y()) for feature in parks_layer.getFeatures()]
park_tree = STRtree(parks)

def calculate_shortest_distances(start_park_feature):
    start_park_point = QgsPointXY(start_park_feature.geometry().asPoint())
    start_park_point_text = f"{start_park_point.x()}, {start_park_point.y()}"
    
    # Find the indices of the end parks within the distance threshold using the spatial index
    nearby_park_indices = park_tree.query(Point(start_park_point.x(), start_park_point.y()).buffer(distance_threshold))
    
    # Create a temporary layer for the nearby end parks
    end_parks_layer = QgsVectorLayer("Point?crs=EPSG:2263", "end_parks_layer", "memory")
    end_parks_layer.startEditing()
    for park_index in nearby_park_indices:
        end_park_point = parks[park_index]
        end_park_feature = QgsFeature()
        end_park_feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(end_park_point.x, end_park_point.y)))
        end_parks_layer.addFeature(end_park_feature)
    end_parks_layer.commitChanges()
    
    # Calculate the shortest paths
    shortest_path_params = {
        'INPUT': network_layer,
        'START_POINT': start_park_point_text,
        'END_POINTS': end_parks_layer,
        'STRATEGY': 0,
        'TRAVEL_COST': distance_threshold,
        'OUTPUT': 'memory:'
    }
    shortest_paths = processing.run("native:shortestpathpointtolayer", shortest_path_params)['OUTPUT']
    
    results = []
    for shortest_path_feature in shortest_paths.getFeatures():
        geometry = shortest_path_feature.geometry()
        if not geometry.isNull():
            end_park_point = geometry.asPolyline()[-1]
            distance = shortest_path_feature.attribute('cost')
            if distance is not None:
                results.append({
                    'start_park': start_park_point_text,
                    'end_park': f"{end_park_point.x()}, {end_park_point.y()}",
                    'distance': distance
                })
        # else:
            # print(f"Null geometry encountered for start park {start_park_point_text}")  # Debug statement
    
    # print(f"Results for start park {start_park_point_text}: {results}")  # Debug statement
    return results

# Set the FID range for the start parks to analyze
start_fid = 1  # Replace with the starting FID
end_fid = 419  # Replace with the ending FID

# Create a feature request to filter the parks layer based on the FID range
request = QgsFeatureRequest().setFilterExpression(f'"fid" >= {start_fid} AND "fid" <= {end_fid}')

# Open the final CSV file in append mode
output_file = 'nyc_park_distance_1.csv'
fieldnames = ['start_park', 'end_park', 'distance']

# Generate a unique prefix for the temporary file based on the FID range
temp_file_prefix = f"temp_{start_fid}_{end_fid}_"

# Open a temporary file for writing with a unique prefix
with tempfile.NamedTemporaryFile(mode='w', newline='', delete=False, prefix=temp_file_prefix) as temp_file:
    temp_writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
    temp_writer.writeheader()

    # Calculate the shortest distances for the specified range of start parks
    for start_park_feature in tqdm(parks_layer.getFeatures(request), total=end_fid - start_fid + 1, unit="park"):
        park_results = calculate_shortest_distances(start_park_feature)
        
        # Write the results to the temporary file immediately
        temp_writer.writerows(park_results)

    # After the loop, append the temporary file to the final CSV file
    with open(output_file, 'a', newline='') as csvfile:
        csvfile.write(temp_file.read())

    # Clean up the temporary file
    temp_file.close()
    os.unlink(temp_file.name)

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time} seconds")
print(f"Results saved to: {output_file}")

# Cleanup and exit
qgs.exitQgis()
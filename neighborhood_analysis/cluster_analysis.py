import sys
import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_distinct_colors(n):
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [(int((i >> 16) & 255), int((i >> 8) & 255), int(i & 255)) for i in range(0, max_value, interval)]
    return [(r/255, g/255, b/255) for r, g, b in colors]

def main():
    distances = [100, 250, 500, 1000]
    for distance_threshold in distances:
        process_distance(distance_threshold)
    
    create_summary_statistics(distances)
    create_plots(distances)
    create_parks_plots(distances)

def process_distance(distance_threshold):
    df = pd.read_csv("filtered_NYC.csv")
    centroids_df = pd.read_csv("nyc_parks_centroids.csv")

    # Create a 'geometry' column from the 'X' and 'Y' columns
    centroids_df['geometry'] = gpd.points_from_xy(centroids_df['X'], centroids_df['Y'], crs='EPSG:2263')

    # Create a directed graph and perform community detection
    G = create_graph_from_df(df, distance_threshold, centroids_df)
    community_assignment = detect_communities(G)

    # Create the DataFrame of community assignments
    community_df = pd.DataFrame(list(community_assignment.items()), columns=['fid', 'community'])

    # Merge with original DataFrame to include park names, boroughs, and FID types
    merged_df = pd.merge(community_df, df[['start_fid', 'end_fid']], left_on='fid', right_on='start_fid', how='left')
    merged_df = pd.merge(merged_df, df[['start_fid', 'end_fid']], left_on='fid', right_on='end_fid', how='left', suffixes=('_start', '_end'))
    merged_df['fid_type'] = 'start,end'
    merged_df = pd.merge(merged_df, centroids_df[['fid', 'name311', 'borough', 'geometry']], on='fid', how='left')

    # Group by community and aggregate park names, boroughs, FIDs, and FID types
    grouped_df = merged_df.groupby('community').agg({
        'fid': lambda x: ', '.join(x.dropna().astype(int).astype(str).unique()),
        'name311': lambda x: ', '.join(x.dropna().unique()),
        'borough': lambda x: ', '.join(x.dropna().unique()),
        'fid_type': 'first',
        'geometry': lambda x: list(x)
    }).reset_index()

    # Check if the cluster spans multiple boroughs
    grouped_df['multiborough'] = grouped_df['borough'].apply(lambda x: len(x.split(', ')) > 1)

    # Save the DataFrame of community assignments
    output_filename = f'nyc_clusters_at_{distance_threshold}.csv'
    grouped_df[['community', 'fid', 'name311', 'borough', 'fid_type', 'multiborough']].to_csv(output_filename, index=False)

def create_graph_from_df(df, distance_threshold, centroids_df):
    G = nx.Graph()

    for _, row in df.iterrows():
        source = row['start_fid']
        target = row['end_fid']
        distance = row['distance_meters']
        if distance <= distance_threshold:
            source_name311 = centroids_df[centroids_df['fid'] == source]['name311'].values[0]
            target_name311 = centroids_df[centroids_df['fid'] == target]['name311'].values[0]
            if source_name311 != target_name311:  # Exclude paths between the same park
                G.add_edge(source, target, weight=distance)
                G.add_edge(target, source, weight=distance)  # Add the reverse edge

    return G

def detect_communities(G):
    communities_generator = nx.community.girvan_newman(G)
    top_level_communities = next(communities_generator)
    community_assignment = {}
    for i, community in enumerate(top_level_communities):
        for node in community:
            community_assignment[node] = i

    return community_assignment

def create_summary_statistics(distances):
    summary_stats = []
    for distance_threshold in distances:
        df = pd.read_csv(f'/clusters/nyc_clusters_at_{distance_threshold}.csv')

        total_stats = compute_stats(df)
        total_stats['distance'] = distance_threshold
        total_stats['category'] = 'Total'
        summary_stats.append(total_stats)

        for borough in ['M', 'Q', 'X', 'B', 'R']:
            print(f"Processing borough {borough} for distance {distance_threshold}")
            borough_df = df[df['borough'].str.contains(borough)]
            if not borough_df.empty:
                borough_stats = compute_stats(borough_df)
                borough_stats['distance'] = distance_threshold
                borough_stats['category'] = f'Borough: {borough}'
                summary_stats.append(borough_stats)
            else:
                # Add placeholder for borough with no data
                borough_stats = {
                    'num_communities': 0,
                    'num_parks': 0,
                    'min': 0,
                    'max': 0,
                    'mean': 0,
                    'se': 0,
                    'distance': distance_threshold,
                    'category': f'Borough: {borough}'
                }
                summary_stats.append(borough_stats)
        
        multiborough_df = df[df['multiborough']]
        if not multiborough_df.empty:
            multiborough_stats = compute_stats(multiborough_df)
            multiborough_stats['distance'] = distance_threshold
            multiborough_stats['category'] = 'Multiborough'
            summary_stats.append(multiborough_stats)

    summary_stats_df = pd.DataFrame(summary_stats)
    summary_stats_df.to_csv('/clusters/nyc_cluster_summary_statistics.csv', index=False)

def compute_stats(df):
    community_sizes = df['fid'].str.split(', ').apply(len)
    stats = {
        'num_communities': df['community'].nunique(),
        'num_parks': community_sizes.sum(),
        'min': community_sizes.min(),
        'max': community_sizes.max(),
        'mean': community_sizes.mean(),
        'se': community_sizes.std() / np.sqrt(len(community_sizes))
    }
    return stats

def create_plots(distances):
    summary_stats_df = pd.read_csv('/clusters/nyc_cluster_summary_statistics.csv')

    # Map for full names of boroughs
    borough_names = {
        'M': 'Manhattan',
        'Q': 'Queens',
        'X': 'Bronx',
        'B': 'Brooklyn',
        'R': 'Staten Island',
        'Total': 'City-wide'
    }

    # Order of lines
    categories_order = ['X', 'M', 'B', 'Q', 'R', 'Total']

    fig, ax = plt.subplots()

    for category_code in categories_order:
        if category_code == 'Total':
            category_df = summary_stats_df[summary_stats_df['category'] == 'Total']
            category_name = 'City-wide'
        else:
            category_df = summary_stats_df[summary_stats_df['category'] == f'Borough: {category_code}']
            category_name = borough_names.get(category_code, category_code)
        
        if not category_df.empty:
            ax.plot(
                category_df['distance'],
                category_df['num_communities'],
                'o-',
                label=category_name
            )

    ax.set_xlabel('Distance Threshold (m)')
    ax.set_ylabel('Number of Communities')
    ax.set_title('Number of Communities with Distance Thresholds')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Remove the grid
    ax.grid(False)

    # Ensure the path exists and is correctly formatted
    plot_path = os.path.join(os.getcwd(), '/figures/number_of_communities_plot.png')
    try:
        print(f"Attempting to save plot to: {plot_path}")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.show()
        print(f"Plot successfully saved to {plot_path}")
    except Exception as e:
        print(f"Failed to save the plot: {e}")

def create_parks_plots(distances):
    summary_stats_df = pd.read_csv('/clusters/nyc_cluster_summary_statistics.csv')

    # Map for full names of boroughs
    borough_names = {
        'M': 'Manhattan',
        'Q': 'Queens',
        'X': 'Bronx',
        'B': 'Brooklyn',
        'R': 'Staten Island',
        'Total': 'City-wide'
    }

    # Order of lines
    categories_order = ['X', 'M', 'B', 'Q', 'R', 'Total']

    fig, ax = plt.subplots()

    for category_code in categories_order:
        if category_code == 'Total':
            category_df = summary_stats_df[summary_stats_df['category'] == 'Total']
            category_name = 'City-wide'
        else:
            category_df = summary_stats_df[summary_stats_df['category'] == f'Borough: {category_code}']
            category_name = borough_names.get(category_code, category_code)
        
        if not category_df.empty:
            ax.errorbar(
                category_df['distance'],
                category_df['mean'],
                yerr=category_df['se'],
                fmt='o-',
                label=category_name,
                capsize=5
            )

    ax.set_xlabel('Distance Threshold (m)')
    ax.set_ylabel('Number of Parks within Communities')
    ax.set_title('Number of Parks within Communities with Distance Thresholds')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Remove the grid
    ax.grid(False)

    # Ensure the path exists and is correctly formatted
    plot_path = os.path.join(os.getcwd(), '/figures/number_of_parks_within_communities_plot.png')
    try:
        print(f"Attempting to save plot to: {plot_path}")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.show()
        print(f"Plot successfully saved to {plot_path}")
    except Exception as e:
        print(f"Failed to save the plot: {e}")

if __name__ == "__main__":
    main()

# scripts/intersection_analysis.py

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


# Step 1: Calculate the bounding box
def calculate_bounding_box(df, expand_by=30):
    lat_min, lat_max = df['Latitude'].min(), df['Latitude'].max()
    lon_min, lon_max = df['Longitude'].min(), df['Longitude'].max()

    # Expand the bounding box to ensure capturing more intersections
    lat_min -= expand_by
    lat_max += expand_by
    lon_min -= expand_by
    lon_max += expand_by

    return lat_min, lat_max, lon_min, lon_max

# Step 2: Filter intersections within the bounding box
def filter_intersections_within_bbox(intersections_df, lat_min, lat_max, lon_min, lon_max, EARTH_RADIUS = 6371.0 ):
    filtered_intersections_df = intersections_df[
        (intersections_df['Y'] >= lat_min) & (intersections_df['Y'] <= lat_max) &
        (intersections_df['X'] >= lon_min) & (intersections_df['X'] <= lon_max)
    ]
    return filtered_intersections_df

# Step 3: Find nearest intersections using KDTree
def find_nearest_intersections(df, intersections_df, distance_threshold_km=0.01, EARTH_RADIUS = 6371.0):
    # Convert distance threshold to radians
    distance_threshold_rad = distance_threshold_km / EARTH_RADIUS

    # Convert coordinates to radians
    intersections_coords = intersections_df[['Y', 'X']].values
    intersections_coords_rad = np.radians(intersections_coords)
    df_coords = df[['Latitude', 'Longitude']].values
    df_coords_rad = np.radians(df_coords)

    # Create KDTree and query the nearest intersection
    kdtree = cKDTree(intersections_coords_rad)

    # Create a copy of the dataframe and initialize the intersection column
    df = df.copy()
    df['intersection'] = None
    intersections_list = []

    for i, coord in enumerate(df_coords_rad):
        distances, indices = kdtree.query(coord, k=1, distance_upper_bound=distance_threshold_rad)
        if distances != float('inf'):  # If a point is found within the distance threshold
            intersection_info = intersections_df.iloc[indices]
            intersection_name = intersection_info['NODE_NAME']
            df.iloc[i, df.columns.get_loc('intersection')] = intersection_name
            intersections_list.append(intersection_info)

    # Create a new DataFrame for the intersections found
    intersections_df_new = pd.DataFrame(intersections_list).drop_duplicates()

    return df, intersections_df_new

def extract_intersection_range(df):
    first_index = df['intersection'].first_valid_index()
    last_index = df['intersection'].last_valid_index()
    if first_index is not None and last_index is not None:
        return df.loc[first_index:last_index]
    else:
        return df


# Main function that integrates the above steps
def process_intersections(df_resampled_filtered, intersections_df, distance_threshold_km=0.01):
    # Step 1: Calculate bounding box
    lat_min, lat_max, lon_min, lon_max = calculate_bounding_box(df_resampled_filtered)

    # Step 2: Filter intersections within the bounding box
    filtered_intersections_df = filter_intersections_within_bbox(intersections_df, lat_min, lat_max, lon_min, lon_max)

    # Step 3: Find nearest intersections and add intersection information to df
    df_with_intersections, intersections_df_new = find_nearest_intersections(
        df_resampled_filtered, filtered_intersections_df, distance_threshold_km
    )

    # Step 4: Extract the range between the first and last valid intersection indices
    df_with_valid_intersections = extract_intersection_range(df_with_intersections)

    return df_with_valid_intersections, intersections_df_new



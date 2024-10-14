# proximity_analysis.py

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from src.haversine import haversine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))




def mark_points_within_distance(df, intersections_df, distance_threshold=0.03):
    """
    Marks points within a certain distance of intersections and returns the filtered dataframe.
    
    Args:
    df (pd.DataFrame): The dataframe containing GPS points.
    intersections_df (pd.DataFrame): The dataframe containing intersection points.
    distance_threshold (float): The threshold distance in kilometers.
    
    Returns:
    pd.DataFrame: The filtered dataframe containing points within the threshold distance of an intersection.
    """
    # Initialize new column to identify if a point is within a certain distance of an intersection
    df['within_30m'] = False

    # Loop through intersections_df and mark points within the distance threshold
    for _, intersection in intersections_df.iterrows():
        intersection_lat = intersection['Y']
        intersection_lon = intersection['X']
        
        # Calculate distances from each point in df to the current intersection
        distances = df.apply(
            lambda row: haversine(row['Longitude'], row['Latitude'], intersection_lon, intersection_lat), axis=1
        )
        
        # Mark points within the threshold
        df['within_30m'] = df['within_30m'] | (distances <= distance_threshold)

    # Filter the data to only include points within the distance threshold
    df_within_threshold = df[df['within_30m']].copy()

    return df_within_threshold






def calculate_nearest_intersections(df, intersections_df):
    df['nearest_intersection'] = None
    df['nearest_distance'] = np.inf

    for _, intersection in intersections_df.iterrows():
        intersection_lat = intersection['Y']
        intersection_lon = intersection['X']
        intersection_id = intersection['NODE_NAME']

        # Calculate distances from each point in df to the current intersection
        distances = df.apply(
            lambda row: haversine(row['Longitude'], row['Latitude'], intersection_lon, intersection_lat), axis=1
        )

        # Update the nearest intersection if the distance is smaller
        closer_points = distances < df['nearest_distance']
        df.loc[closer_points, 'nearest_distance'] = distances[closer_points]
        df.loc[closer_points, 'nearest_intersection'] = intersection_id

    return df


def calculate_distance_to_nearest(df, intersections_df):
    """Calculates distance to the nearest intersection for each point."""
    df['distance_to_intersection'] = df.apply(
        lambda row: min([haversine(row['Longitude'], row['Latitude'], inter_lon, inter_lat)
                         for inter_lon, inter_lat in zip(intersections_df['X'], intersections_df['Y'])]),
        axis=1
    )
    return df
    
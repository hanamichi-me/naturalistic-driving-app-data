# scripts/directional_distance.py
import sys
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.haversine import haversine

# Step 2: Compute directional distance to the nearest intersection
def compute_directional_distances(df, intersections_df_new):
    df['directional_distance_to_intersection'] = np.nan

    for i in range(1, len(df)):
        prev_row = df.iloc[i - 1]
        current_row = df.iloc[i]

        if current_row['nearest_intersection'] == prev_row['nearest_intersection']:
            intersection_id = current_row['nearest_intersection']
            intersection = intersections_df_new[intersections_df_new['NODE_NAME'] == intersection_id].iloc[0]

            df.loc[df.index[i], 'directional_distance_to_intersection'] = compute_directional_distance(
                current_row, prev_row, intersection_lon=intersection['X'], intersection_lat=intersection['Y']
            )

    return df

# Compute directional distance between two points
def compute_directional_distance(row, prev_row, intersection_lon, intersection_lat):
    current_distance = haversine(row['Longitude'], row['Latitude'], intersection_lon, intersection_lat)
    prev_distance = haversine(prev_row['Longitude'], prev_row['Latitude'], intersection_lon, intersection_lat)

    # If the previous distance is greater, the vehicle is approaching the intersection
    if prev_distance > current_distance:
        return -current_distance
    else:
        return current_distance
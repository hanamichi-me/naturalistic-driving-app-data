# data_processing.py
# data_processing.py: This file contains functions to load, preprocess, resample, and slice trip data, as well as calculate distances and speeds using the Haversine formula.

import os
import pandas as pd
import numpy as np
from datetime import datetime
# from src.feature_extraction import add_new_features
# from src.kalman_filter import process_trip_data



# # Loads the dataset from a file and assigns column names
# def load_data(file_path, column_names):
#     df = pd.read_csv(file_path, header=None, names=column_names)
#     return df

# Preprocesses the data: converts columns to numeric types, handles missing values, and sets the 'Time' column as the index
def preprocess_data(df, config, logger):
    df['AccX'] = pd.to_numeric(df['AccX'], errors='coerce').fillna(0.0)
    df['AccX'] = df['AccX'].astype(float)
    df = df.dropna(subset=['AccX'])

    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df.dropna(subset=['Latitude', 'Longitude'])
    df['label'] = None
    df['Time'] = pd.to_datetime(df['Time'], format='mixed', errors='coerce')
    df.set_index('Time', inplace=True)


    # 保存数据到 CSV
    filename = config['data']['filename']
    output_path = os.path.join(config['data_load']['dataset_csv'], filename)
    df.to_csv(output_path, index=True)

    logger.info(f'Preprocessed dataset saved to {output_path}')
    logger.info(f'Display preprocessed dataset from {output_path}')


    return df

def save_dataframe(df, path):
    """Saves the dataframe to a specified path"""
    df.to_csv(path, index=False)
    print(f"DataFrame has been saved to {path}")



def assign_segments(df, intersection_column='nearest_intersection'):
    """
    Assign segments to each row in the DataFrame based on changes in the intersection.

    Args:
        df (DataFrame): The input DataFrame with nearest intersection information.
        intersection_column (str): The column name that contains intersection information.

    Returns:
        DataFrame: Updated DataFrame with an additional 'segment' column.
    """
    df = df.copy()
    df['segment'] = None
    current_segment = 1
    prev_intersection = None
    for i in range(len(df)):
        current_intersection = df.iloc[i][intersection_column]
        # If we encounter a new intersection, increment the segment ID
        if current_intersection != prev_intersection and pd.notna(current_intersection):
            current_segment += 1
        df.loc[df.index[i], 'segment'] = current_segment
        prev_intersection = current_intersection
    return df


def extract_segments_between_intersections(df, intersection_column='intersection'):
    """
    Extract data points between the first and last intersections.

    Args:
        df (DataFrame): The input DataFrame with intersection information.
        intersection_column (str): The column name that contains intersection information.

    Returns:
        DataFrame: DataFrame with data points between the first and last valid intersection indices.
    """
    first_index = df[intersection_column].first_valid_index()
    last_index = df[intersection_column].last_valid_index()

    if first_index is not None and last_index is not None:
        return df.loc[first_index:last_index]
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no valid intersection is found


def remove_proximity_points(df, distance_column='nearest_distance', distance_threshold=0):
    """
    Remove data points that are within a certain distance threshold of intersections.

    Args:
        df (DataFrame): The input DataFrame.
        distance_column (str): The column name that contains distance to intersection.
        distance_threshold (float): Threshold distance below which data points should be removed.

    Returns:
        DataFrame: DataFrame without data points that are too close to intersections.
    """
    return df[df[distance_column] > distance_threshold]

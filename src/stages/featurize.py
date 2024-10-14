
import sys
import argparse
import pandas as pd
import numpy as np
from typing import Text
# from src.feature_extraction import haversine
import yaml
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.logs import get_logger 
from src.kalman_filter import process_trip_data
from src.visualization import load_map_configurations
from src.intersection_analysis import process_intersections
from src.proximity_analysis import mark_points_within_distance, calculate_nearest_intersections, calculate_distance_to_nearest
from src.haversine import haversine
from src.directional_distance import compute_directional_distances, compute_directional_distance
from datetime import datetime




def featurize(config_path: Text) -> None:
    """
    Load data, apply resampling and feature calculation, and save the result.

    Args:
        config_path (Text): Path to the configuration file (YAML).
    """
    # Load configuration file
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('FEATURIZE', log_level=config['base']['log_level'])
    # logger.info('Load raw data')
    filename = config['data']['filename']
    dataset_csv = config['data_load']['dataset_csv']
    file_path = os.path.join(dataset_csv, filename)

    df = pd.read_csv(file_path, index_col='Time', parse_dates=['Time'])

    # Apply preprocessing or any necessary steps if required
    df_resampled = resample_and_calculate(df, config, logger)
    

    return df_resampled, config, logger

# Resamples the data at 1-second intervals and calculates the distance and speed between consecutive points
def resample_and_calculate(df, config, logger):

    logger = get_logger('FEATURIZE', log_level=config['base']['log_level'])
    logger.info('Load raw data')
    df_resampled = df.resample('1s').mean()
    

    df_resampled['prev_Latitude'] = df_resampled['Latitude'].shift(1)
    df_resampled['prev_Longitude'] = df_resampled['Longitude'].shift(1)
    
    df_resampled['distance_km'] = df_resampled.apply(lambda row: haversine(
        row['Longitude'], row['Latitude'], row['prev_Longitude'], row['prev_Latitude']), axis=1)
    
    df_resampled['time_diff'] = df_resampled.index.to_series().diff().dt.total_seconds()
    
    df_resampled['speed_kmh'] = (df_resampled['distance_km'] / (df_resampled['time_diff'] / 3600)).fillna(0)
    
    df_resampled.drop(columns=['prev_Latitude', 'prev_Longitude'], inplace=True)
    # logger.info('function resample_and_calculate return df_resampled')

    
    return df_resampled

def slice_and_save_data(df_resampled: pd.DataFrame, config: dict, logger )  -> pd.DataFrame:
    """
    Slice the resampled data based on start and end rows, and save the filtered data to a CSV file.

    Args:
        df_resampled (pd.DataFrame): The resampled dataframe to slice and save.
        config (dict): The configuration dictionary containing file paths and slice parameters.
    
    Returns:
        pd.DataFrame: The sliced dataframe.
    """
    # Get slicing parameters from config with default values
    start_row = config['slice_params'].get('start_row', 1)
    end_row = config['slice_params'].get('end_row', len(df_resampled) - 2)

    # Slice the resampled data
    df_resampled_filtered = df_resampled.iloc[start_row:end_row]

    # Define new main directory for saving
    main_dir1 = config['paths']['main_dir1']
    sub_dir = config['paths']['sub_dir']
    filename = config['data']['filename']

    # Create file path and directories if they don't exist
    os.makedirs(os.path.join(main_dir1, sub_dir), exist_ok=True)

    # Save the filtered data to a CSV file
    output_path = os.path.join(main_dir1, sub_dir, filename)
    os.makedirs(os.path.join(main_dir1, sub_dir), exist_ok=True)
    
    df_resampled_filtered.to_csv(output_path, index=False)
    logger.info(f'Filtered data saved to {output_path}')


    return df_resampled_filtered



# Step 3: Create new features for analysis
def add_new_features(df):
    # Calculate acceleration magnitude
    df['acceleration_magnitude'] = df.apply(
        lambda row: np.sqrt(row['AccX']**2 + row['AccY']**2 + row['AccZ']**2), axis=1
    )

    # Calculate gyroscope magnitude
    df['gyroscope_magnitude'] = df.apply(
        lambda row: np.sqrt(row['GyX']**2 + row['GyY']**2 + row['GyZ']**2), axis=1
    )

    # Calculate acceleration rate of change (derivative)
    df['acceleration'] = df['speed_kmh'].diff().fillna(0) / (df['time_diff'].fillna(1))

    # Clip acceleration to remove extreme values caused by data sampling noise
    df['acceleration'] = df['acceleration'].clip(lower=-20, upper=20)

    return df
    

def load_trip_data(main_dir, main_dir2, merged_trips,logger):
    """
    Load, process, and save trip data from a given directory.
    
    Args:
        main_dir (str): Path to the directory containing the trip data files.
        main_dir2 (str): Path to the directory where the data should be saved.
        merged_trips (str): Name of the output CSV file.
        
    Returns:
        None
    """
    all_trips = []
    
    # Iterate through all .txt files in the directory
    for root, dirs, files in os.walk(main_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                trip_id = os.path.basename(file)
                df['trip_id'] = trip_id
                
                # Add new features and process trip data
                df = add_new_features(df)
                df_processed = process_trip_data(df)
                
                all_trips.append(df_processed)
    
    # Concatenate all processed trips
    df_resampled_filtered = pd.concat(all_trips, ignore_index=True)
    
    # Create the output directory if it doesn't exist
    os.makedirs(main_dir2, exist_ok=True)
    
    # Save the processed data to a CSV file
    output_path1 = os.path.join(main_dir2, merged_trips)
    df_resampled_filtered.to_csv(output_path1, index=False)
    logger.info(f'df_resampled_filtered Dataset saved to {output_path1}')
    
    return df_resampled_filtered

def process_and_save_intersections(df_resampled_filtered, intersections_df, config, main_dir2,logger):
    """
    Process and save intersections using the filtered resampled trip data.
    
    Args:
        df_resampled_filtered (pd.DataFrame): The resampled and filtered trip data.
        config (dict): Configuration dictionary loaded from params.yaml.
        main_dir2 (str): Directory to save the processed intersection data.
        
    Returns:
        None
    """

    # Process intersections
    df_resampled_filtered, intersections_df_new = process_intersections(df_resampled_filtered, intersections_df)
    
    # Save the updated intersections dataframe to CSV
    output_path = os.path.join(main_dir2, config['data']['intersections_csv'])
    intersections_df_new.to_csv(output_path, index=False)
    logger.info(f'intersections_df_new Dataset saved to {output_path}')

    output_path1 = os.path.join(main_dir2, config['data']['df_resampled_filtered'])
    df_resampled_filtered.to_csv(output_path1, index=False)
    logger.info(f'df_resampled_filtered Dataset saved to {output_path1}')
    
    
    return df_resampled_filtered, intersections_df_new


def process_data(df, intersections_df, output_path, main_dir2, logger):
    """
    Process the data by calculating nearest intersections and directional distances, 
    and save the resulting DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The trip data to be processed.
        intersections_df (pd.DataFrame): The intersections data.
        config (dict): Configuration dictionary loaded from params.yaml.
        main_dir2 (str): Directory to save the processed data.
        
    Returns:
        None
    """

    # Step 1: Process data - calculate nearest intersections and directional distances
    df_processed = calculate_nearest_intersections(df, intersections_df)
    df_processed = compute_directional_distances(df_processed, intersections_df)
    # Step 2: Save the processed DataFrame to a CSV file
    os.makedirs(main_dir2, exist_ok=True)  # Ensure the directory exists

    df_processed.to_csv(output_path, index=True)
    logger.info(f'df_within_30m_csv Dataset saved to {output_path}')

    return df_processed

if __name__ == '__main__':
    # Argument parser for command-line execution
    parser = argparse.ArgumentParser(description="Resample and calculate features for trip data.")
    parser.add_argument('--config', dest='config', required=True, help="Path to configuration file")
    args = parser.parse_args()

    # Run featurization process
    df_resampled, config, logger = featurize(config_path=args.config)

    df_resampled_filtered = slice_and_save_data(df_resampled, config,logger)

    df_resampled_filtered = load_trip_data(config['paths']['main_dir1'], config['paths']['main_dir2'], config['data']['merged_trips'],logger)

    (intersections_df, intersections_no_signal_df, traffic_signal_df, _, _, _, _, _, _, _) = load_map_configurations(config)
    
    # !important!!!!!!
    # choose what datasets you want to analyse: intersections_df/intersections_no_signal_df/traffic_signal_df
    df_resampled_filtered, intersections_df_new = process_and_save_intersections(df_resampled_filtered, traffic_signal_df, config, config['paths']['main_dir2'], logger)

    df_within_30m = mark_points_within_distance(df_resampled_filtered, intersections_df_new, distance_threshold=0.03)
    df_within_30m = calculate_distance_to_nearest(df_within_30m, intersections_df_new)
    df_within_30m = process_data(df_within_30m, intersections_df_new, os.path.join(config['paths']['main_dir2'], config['data']['df_within_30m_csv']), config['paths']['main_dir2'],logger)
    



# data_split.py

import sys
import os

# 添加项目的根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Text
from src.utils.logs import get_logger  # 假设你已经有一个日志工具
from features_and_targets1 import features_and_targets
import joblib  # For saving the scaler

def set_random_seed(seed_value: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed_value)

def load_data(file_path: Text, column_names: list) -> pd.DataFrame:
    """Load data from a CSV file and set the column names."""
    return pd.read_csv(file_path, names=column_names)

def data_split(df, features, target, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets, and scale the features.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        features (list): List of feature column names.
        target (str): The target column name.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.

    Returns:
        X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, scaler
    """
    X = df[features]
    y = df[target]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # X_train = pd.DataFrame(X_train, columns=features)
    # X_test = pd.DataFrame(X_test, columns=features)
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train.columns)

    return X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, scaler


def save_splits(X_test, X_train_scaled, X_test_scaled, y_train, y_test, config):
    """
    Save the train/test splits to CSV files.

    Args:
        X_train_scaled (ndarray): Scaled training features.
        X_test_scaled (ndarray): Scaled testing features.
        y_train (ndarray): Training labels.
        y_test (ndarray): Testing labels.
        output_dir (str): Directory where to save the CSV files.
    """
    os.makedirs(config['paths']['split_data_dir'], exist_ok=True)

    # Convert to DataFrame for saving
    pd.DataFrame(X_test).to_csv(os.path.join(config['paths']['split_data_dir'], config['data']['X_test']), index=False)
    pd.DataFrame(X_train_scaled).to_csv(os.path.join(config['paths']['split_data_dir'], config['data']['X_train_scaled']), index=False)
    pd.DataFrame(X_test_scaled).to_csv(os.path.join(config['paths']['split_data_dir'], config['data']['X_test_scaled']), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(config['paths']['split_data_dir'], config['data']['y_train']), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(config['paths']['split_data_dir'], config['data']['y_test']), index=False)


def save_scaler(scaler, output_dir, config):
    """
    Save the trained scaler to a file using joblib.

    Args:
        scaler (StandardScaler): The fitted scaler.
        output_dir (str): Directory where to save the scaler.
    """
    scaler_path = os.path.join(output_dir, config['data']['scaler'])
    joblib.dump(scaler, scaler_path)
    print(f'Scaler saved to {scaler_path}')
    

def data_split_pipeline(config_path: Text, feature_set_index: int) -> None:
    """
    Pipeline to load data, split it into train/test sets, and scale the features according to configuration file.

    Args:
        config_path {Text}: Path to the configuration file.
    """
    # Load configuration file
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('DATA_SPLIT', log_level=config['base']['log_level'])
    
    # Set random seed
    seed_value = config['base']['random_stage']
    set_random_seed(seed_value)
    logger.info(f'Set random seed: {seed_value}')

    # Build file path

    file_path = os.path.join(config['paths']['main_dir2'],  config['data']['df_within_30m_csv'])

    logger.info(f'Loading data from {file_path}')
    df = pd.read_csv(file_path)
    selected_set = features_and_targets[feature_set_index]
    features = selected_set['features']
    target = selected_set['target']
    logger.info(f'Using features: {features} and target: {target}')
    
    logger.info(f'Splitting data into train/test sets')
    X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, scaler = data_split(df, features, target)
    logger.info(f'Data splitting completed.')
    
    # Save the splits
    output_dir = config['paths']['split_data_dir']
    save_splits(X_test, X_train_scaled, X_test_scaled, y_train, y_test, config)
    logger.info(f'Train/test splits saved to {output_dir}.')

    # Save the scaler
    save_scaler(scaler, output_dir, config)
    logger.info(f'Scaler saved to {output_dir}.')


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help="Path to configuration file")
    parser.add_argument('--feature_set_index', dest='feature_set_index', type=int, required=True, help="Index of the feature set to use")
    args = parser.parse_args()

    # Run data split pipeline with selected feature set
    data_split_pipeline(config_path=args.config, feature_set_index=args.feature_set_index)


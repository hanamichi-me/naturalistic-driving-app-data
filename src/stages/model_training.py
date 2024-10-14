# src/model_training.py

import sys
import os

# 添加项目的根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from typing import Text
from src.utils.logs import get_logger  # 假设你已经有一个日志工具
import joblib  # 用于保存模型

def load_split_data(config):
    """
    Load the saved split train/test data from CSV files.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test: Loaded data arrays.
    """
    X_train_scaled = pd.read_csv(os.path.join(config['paths']['split_data_dir'], config['data']['X_train_scaled'])) 
    X_test_scaled = pd.read_csv(os.path.join(config['paths']['split_data_dir'], config['data']['X_test_scaled']) )
    y_train = pd.read_csv(os.path.join(config['paths']['split_data_dir'], config['data']['y_train']))
    y_test = pd.read_csv(os.path.join(config['paths']['split_data_dir'], config['data']['y_test']))


    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate_model(X_train, y_train, X_test, y_test, n_estimators=100, random_state=42, logger=None):
    """
    Train a Random Forest Regressor and evaluate it.

    Args:
        X_train (array): Scaled training features.
        y_train (array): Training labels.
        X_test (array): Scaled testing features.
        y_test (array): Testing labels.
        n_estimators (int): Number of estimators for the Random Forest.
        random_state (int): Random seed for reproducibility.
        logger: Logger for logging the training process.

    Returns:
        model: Trained Random Forest model.
    """
    if logger:
        logger.info(f"Training Random Forest with {n_estimators} estimators and random state {random_state}")

    # 将 y_train 和 y_test 转换为一维数组
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()


    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if logger:
        logger.info(f"Model evaluation completed. MSE: {mse}, R^2: {r2}")
    else:
        print(f"Mean Squared Error: {mse}")
        print(f"R^2 Score: {r2}")

    return model

def model_training_pipeline(config_path: Text, feature_set_index: int) -> None:
    """
    Pipeline to train and evaluate a Random Forest model according to configuration file.

    Args:
        config_path {Text}: Path to the configuration file.
        feature_set_index (int): Index of the feature set to use.
    """
    # Load configuration file
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('MODEL_TRAINING', log_level=config['base']['log_level'])

    logger.info("Starting model training pipeline.")

    # Load and split data from saved CSVs
    logger.info("Loading and splitting data from saved CSVs.")
    X_train_scaled, X_test_scaled, y_train, y_test = load_split_data(config)
    logger.info("Data loaded and split successfully.")
    

    # Train and evaluate the model
    n_estimators = config['model'].get('n_estimators', 100)
    random_state = config['base']['random_stage']

    logger.info("Training and evaluating the model.")
    model = train_and_evaluate_model(X_train_scaled, y_train, X_test_scaled, y_test, n_estimators=n_estimators, random_state=random_state, logger=logger)

    # Save the model to a file if specified in the config
    if 'model_output_path' in config['paths']:
        model_output_path = config['paths']['model_output_path']
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        print(os.path.join(model_output_path, config['data']['random_forest_model']))
        joblib.dump(model, os.path.join(model_output_path, config['data']['random_forest_model']))
        logger.info(f"Model saved to {os.path.join(model_output_path, config['data']['random_forest_model'])}")


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help="Path to configuration file")
    parser.add_argument('--feature_set_index', dest='feature_set_index', type=int, required=True, help="Index of the feature set to use")
    args = parser.parse_args()

    # Run the model training pipeline
    model_training_pipeline(config_path=args.config, feature_set_index=args.feature_set_index)

# visualization.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import yaml
import joblib  
from src.utils.logs import get_logger 
from typing import Text
import argparse
from features_and_targets1 import features_and_targets
from src.stages.data_split import data_split_pipeline
from src.stages.model_training import model_training_pipeline
# from src.stages.visualization import plot_prediction_vs_distance


def plot_prediction_vs_distance(model, scaler, X_test, target_name, feature_name='directional_distance_to_intersection', 
                                distance_range=(-0.03, 0.03), steps=100, ax=None, save_path=None, logger=None):
    """
    Plot the prediction of the model against the distance to intersection.
    
    Args:
        model: Trained model used for predictions.
        scaler: The scaler used to scale the input data.
        X_test: The test dataset.
        target_name: Name of the target variable.
        feature_name: Name of the feature (default is 'directional_distance_to_intersection').
        distance_range: The range of distances for prediction (default is (-0.03, 0.03)).
        steps: Number of steps for the plot (default is 100).
        ax: Matplotlib axis to plot on (optional).
        save_path: Path to save the plot (optional).
        logger: Logger for logging information (optional).
    
    Returns:
        None
    """
    if logger:
        logger.info(f"Starting plot for {target_name} vs {feature_name}")
    
    if ax is None:
        fig, ax = plt.subplots()

    unique_distances = np.linspace(distance_range[0], distance_range[1], steps)
    y_plot_mean = []
    X_plot_list = []

    try:
        for distance in unique_distances:
            X_temp = X_test.copy()
            # 更新 feature_name 的值为当前的 distance
            if feature_name in X_temp.columns:
                X_temp[feature_name] = distance
            else:
                X_temp = X_temp.assign(**{feature_name: distance})
            # X_temp[feature_name] = distance

            X_plot_list.append(X_temp)

        # 合并所有的生成数据
        X_plot = pd.concat(X_plot_list)

        # 数据标准化
        X_plot_scaled = scaler.transform(X_plot)

        X_plot_scaled = pd.DataFrame(X_plot_scaled, columns=X_plot.columns)

        # 进行预测
        y_plot = model.predict(X_plot_scaled)
    
        # 计算每个 unique_distance 的预测均值
        for i in range(len(unique_distances)):
            start_index = i * len(X_test)
            end_index = (i + 1) * len(X_test)
            y_plot_mean.append(np.mean(y_plot[start_index:end_index]))

        # 平滑处理
        y_plot_mean_smooth = gaussian_filter1d(y_plot_mean, sigma=2)

        # 绘制平滑曲线和原始预测
        ax.plot(unique_distances * 1000, y_plot_mean_smooth, label=f'Smoothed Predicted {target_name}', color='blue')
        ax.plot(unique_distances * 1000, y_plot_mean, alpha=1, linestyle=':', label=f'Original Predicted {target_name}', color='darkorange', linewidth=2)
        ax.set_xlabel('Distance to Intersection (meters)')
        ax.set_ylabel(f'Predicted {target_name}')
        ax.set_title(f'Predicted {target_name} vs Distance to Intersection')
        ax.axvline(x=0, color='r', linestyle='--', label='Intersection')
        ax.legend()

        # 保存图像（如果有路径）
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            if logger:
                logger.info(f"Plot saved to {save_path}")
        
        if logger:
            logger.info(f"Plotting {target_name} vs {feature_name} completed successfully.")

        return ax
        
    except Exception as e:
        if logger:
            logger.error(f"Error occurred while plotting: {e}")
        raise e


def visualization_pipeline(config_path: Text, feature_set_index: int, save_path=None, feature_name='directional_distance_to_intersection', distance_range=(-0.03, 0.03), steps=100, ax=None) -> None:
    """
    Pipeline to generate a plot of model predictions vs distance to intersections.
    
    Args:
        config_path (Text): Path to the configuration file.
    
    Returns:
        None
    """
    # Load configuration file
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    # Initialize logger
    logger = get_logger('PLOT_PREDICTION', log_level=config['base']['log_level'])

    logger.info("Starting visualization pipeline.")

    # Load model and scaler
    model = joblib.load(os.path.join(config['paths']['model_output_path'], config['data']['random_forest_model']))
    scaler = joblib.load(os.path.join(config['paths']['split_data_dir'], 'scaler.pkl'))

    # Load test data
    X_test = pd.read_csv(os.path.join(config['paths']['split_data_dir'], config['data']['X_test']))

    selected_set = features_and_targets[feature_set_index]
    features = selected_set['features']
    target = selected_set['target']



    # Plot prediction vs distance
    ax = plot_prediction_vs_distance(
        model=model, 
        scaler=scaler, 
        X_test=X_test, 
        target_name=target,  # Replace with your actual target name
        feature_name=feature_name, 
        distance_range=distance_range, 
        steps=steps, 
        ax=ax,
        save_path=save_path, 
        logger=logger
    )

    return ax

def complete_pipeline(config_path, feature_set_index, ax=None, save_path=None):
    """
    Complete pipeline that performs data splitting, model training, and visualization.
    
    Args:
        config_path (str): Path to the configuration file.
        feature_set_index (int): Index of the feature set to use.
        ax (Matplotlib Axis, optional): Axis object to draw the plot on. Defaults to None.
        save_path (str, optional): Path to save the plot image. Defaults to None.
    
    Returns:
        ax: The axis with the plot.
    """
    # 1. 数据拆分
    print()
    print(f"Running data split for feature set index {feature_set_index}")
    data_split_pipeline(config_path=config_path, feature_set_index=feature_set_index)
    
    # 2. 模型训练
    print()
    print(f"Running model training for feature set index {feature_set_index}")
    model_training_pipeline(config_path=config_path, feature_set_index=feature_set_index)

    # 3. 加载模型和scaler
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    model = joblib.load(os.path.join(config['paths']['model_output_path'], config['data']['random_forest_model']))
    scaler = joblib.load(os.path.join(config['paths']['split_data_dir'], 'scaler.pkl'))
    
    # 4. 加载测试数据
    X_test = pd.read_csv(os.path.join(config['paths']['split_data_dir'], config['data']['X_test']))

    selected_set = features_and_targets[feature_set_index]
    target = selected_set['target']

        # 5. 可视化
    print()
    print(f"Generating visualization for feature set index {feature_set_index}")
    logger = get_logger('DATA_PLOT', log_level=config['base']['log_level'])
    logger.info("Starting plotting plot_prediction_vs_distance.")
    

    ax = plot_prediction_vs_distance(model, scaler, X_test, target, ax=ax, save_path=save_path, logger=logger)

    return ax



if __name__ == '__main__':
    # Argument parsing for command-line execution
    parser = argparse.ArgumentParser(description="Plot prediction vs distance to intersection.")
    parser.add_argument('--config', dest='config', required=True, help="Path to configuration file")
    parser.add_argument('--feature_set_index', dest='feature_set_index', type=int, required=True, help="Index of the feature set to use")
    parser.add_argument('--save_path', dest='save_path', required=False, help="Optional path to save the plot")
    args = parser.parse_args()

    # Run the visualization pipeline
    ax = visualization_pipeline(
        config_path=args.config,
        feature_set_index=args.feature_set_index,
        save_path=args.save_path
    )
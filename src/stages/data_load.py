# data_load.py

import sys
import os

# 添加项目的根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import pandas as pd
import yaml
from typing import Text
import numpy as np
from src.data_processing import preprocess_data
# from src.feature_extraction import resample_and_calculate
from src.utils.logs import get_logger  # 假设你已经有一个日志工具


def set_random_seed(seed_value: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed_value)

def load_data(file_path: Text, column_names: list) -> pd.DataFrame:
    """Load data from a CSV file and set the column names.
    Args:
        file_path {Text}: path to the data file
        column_names {list}: list of column names for the dataset
    Returns:
        pd.DataFrame: loaded dataset
    """
    return pd.read_csv(file_path, names=column_names)

def data_load(config_path: Text) -> None:
    """Load raw data according to configuration file.
    Args:
        config_path {Text}: path to configuration file
    """
    # 读取配置文件
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    # 初始化日志记录器
    logger = get_logger('DATA_LOAD', log_level=config['base']['log_level'])

    # 设置随机种子
    seed_value = config['base']['random_stage']
    set_random_seed(seed_value)
    logger.info(f'Set random seed: {seed_value}')

    # 构建文件路径
    main_dir = config['paths']['main_dir']
    sub_dir = config['paths']['sub_dir']
    filename = config['data']['filename']
    file_path = os.path.join(main_dir, sub_dir, filename)

    # 加载数据列名
    column_names = config['data_columns']['column_names']

    # 加载数据
    logger.info(f'Loading original data from {file_path}')
    df = load_data(file_path, column_names)
    logger.info(f'display original data from {file_path}')
    

    return df, config, logger


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help="Path to configuration file")
    args = parser.parse_args()

    # 加载数据
    df, config, logger = data_load(config_path=args.config)
    df = preprocess_data(df,config, logger)




















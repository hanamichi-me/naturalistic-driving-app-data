# scripts/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def calculate_cumulative_distance(df):
    """
    Calculate cumulative distance for each segment.

    Args:
        df (DataFrame): Input DataFrame.

    Returns:
        DataFrame: Updated DataFrame with cumulative distance for each segment.
    """
    df['cumulative_distance'] = np.nan

    for segment in df['segment'].unique():
        segment_data = df[df['segment'] == segment]
        cumulative_distances = segment_data['distance_km'].cumsum() * 1000  # 将公里转换为米
        df.loc[segment_data.index, 'cumulative_distance'] = cumulative_distances

    return df
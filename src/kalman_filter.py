# kalman_filter.py

import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor
# from src.feature_extraction import add_new_features

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def update(self, measurement, rate, dt):
        self.priori_estimate = self.posteri_estimate + rate * dt
        self.priori_error_estimate = self.posteri_error_estimate + self.process_variance
        blending_factor = self.priori_error_estimate / (self.priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = self.priori_estimate + blending_factor * (measurement - self.priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * self.priori_error_estimate
        return self.posteri_estimate


def process_trip_data(df):
    def kalman_filter_euler_angles(accel_data, gyro_data, dt_series, process_variance=1e-5, measurement_variance=1e-3):
        pitch_kf = KalmanFilter(process_variance, measurement_variance)
        roll_kf = KalmanFilter(process_variance, measurement_variance)
        yaw_kf = KalmanFilter(process_variance, measurement_variance)

        pitch = np.zeros(len(accel_data))
        roll = np.zeros(len(accel_data))
        yaw = np.zeros(len(accel_data))

        pitch[0] = np.arctan2(-accel_data[0, 0], np.sqrt(accel_data[0, 1]**2 + accel_data[0, 2]**2))
        roll[0] = np.arctan2(accel_data[0, 1], accel_data[0, 2])
        yaw[0] = np.arctan2(accel_data[0, 1], accel_data[0, 0])

        for i in range(1, len(dt_series)):
            pitch_accel = np.arctan2(-accel_data[i, 0], np.sqrt(accel_data[i, 1]**2 + accel_data[i, 2]**2))
            roll_accel = np.arctan2(accel_data[i, 1], accel_data[i, 2])
            yaw_accel = np.arctan2(accel_data[i, 1], accel_data[i, 0])

            pitch_rate = gyro_data[i, 0]
            roll_rate = gyro_data[i, 1]
            yaw_rate = gyro_data[i, 2]

            current_dt = dt_series.iloc[i]

            pitch[i] = pitch_kf.update(pitch_accel, pitch_rate, current_dt)
            roll[i] = roll_kf.update(roll_accel, roll_rate, current_dt)
            yaw[i] = yaw_kf.update(yaw_accel, yaw_rate, current_dt)

        return pitch, roll, yaw

    def calculate_rotation_matrix(roll, pitch, yaw):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(pitch), -np.sin(pitch)],
                        [0, np.sin(pitch), np.cos(pitch)]])
        
        R_y = np.array([[np.cos(roll), 0, np.sin(roll)],
                        [0, 1, 0],
                        [-np.sin(roll), 0, np.cos(roll)]])
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    accel_data = df[['AccX', 'AccY', 'AccZ']].values
    gyro_data = df[['GyX', 'GyY', 'GyZ']].values

    pitch_angles, roll_angles, yaw_angles = kalman_filter_euler_angles(accel_data, gyro_data, df['time_diff'].reset_index(drop=True))

    df['Pitch_Angle'] = pitch_angles
    df['Roll_Angle'] = roll_angles
    df['Yaw_Angle'] = yaw_angles

    reoriented_acc = []
    for i in range(len(df)):
        roll = df['Roll_Angle'].iloc[i]
        pitch = df['Pitch_Angle'].iloc[i]
        yaw = df['Yaw_Angle'].iloc[i]
        R = calculate_rotation_matrix(roll, pitch, yaw)
        acc = np.array([df['AccX'].iloc[i], df['AccY'].iloc[i], df['AccZ'].iloc[i]])
        reoriented_acc.append(np.dot(R, acc))

    reoriented_acc = np.array(reoriented_acc)
    df['Reoriented_AccX'] = reoriented_acc[:, 0]
    df['Reoriented_AccY'] = reoriented_acc[:, 1]
    df['Reoriented_AccZ'] = reoriented_acc[:, 2]

    return df
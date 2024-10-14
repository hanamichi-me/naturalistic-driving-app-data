# scripts/model_training.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def train_and_evaluate_model(X_train, y_train, X_test, y_test, n_estimators=100, random_state=42):
    """
    Train a Random Forest Regressor and evaluate it.

    Args:
        X_train (array): Scaled training features.
        y_train (array): Training labels.
        X_test (array): Scaled testing features.
        y_test (array): Testing labels.
        n_estimators (int): Number of estimators for the Random Forest.
        random_state (int): Random seed for reproducibility.

    Returns:
        model: Trained Random Forest model.
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    return model
    

def predict_segment_values(df, features, model, scaler, target_name):
    """
    Predict target values for each segment using the trained model.

    Args:
        df (DataFrame): Input DataFrame with segments.
        features (list): List of features to use for prediction.
        model (RandomForestRegressor): Trained model.
        scaler (StandardScaler): Scaler used for feature standardization.
        target_name (str): The name of the target variable to be predicted (e.g., 'speed_kmh', 'acceleration').

    Returns:
        DataFrame: Updated DataFrame with actual and predicted target values for each segment.
    """
    all_actual_values = []
    all_predicted_values = []
    all_distances = []

    for segment in df['segment'].unique():
        segment_data = df[df['segment'] == segment]

        cumulative_distance = segment_data['cumulative_distance']
        actual_value = segment_data[target_name]

        segment_X = segment_data[features]
        segment_X_scaled = scaler.transform(segment_X)
        segment_X_scaled = pd.DataFrame(segment_X_scaled, columns=features)
        predicted_value = model.predict(segment_X_scaled)

        all_actual_values.extend(actual_value)
        all_predicted_values.extend(predicted_value)
        all_distances.extend(cumulative_distance)

    combined_df = pd.DataFrame({
        'cumulative_distance': all_distances,
        f'actual_{target_name}': all_actual_values,
        f'predicted_{target_name}': all_predicted_values
    })

    return combined_df


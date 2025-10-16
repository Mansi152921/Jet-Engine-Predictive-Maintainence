# src/models/predict_model.py

import os
import logging
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def predict_on_test_data(processed_datapath, models_path, model_name="PredictiveMaintenanceRUL", model_stage="None"):
    """
    Loads the best model from the registry and evaluates it on the test set.
    """
    logger = logging.getLogger(__name__)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    logger.info(f"Loading model '{model_name}' from stage '{model_stage}'...")

    # --- Load the Registered Model ---
    # The URI format loads the latest version of the model from a specific stage
    logged_model_uri = f'models:/{model_name}/latest'
    loaded_model = mlflow.pyfunc.load_model(logged_model_uri)
    logger.info("Model loaded successfully.")

    # --- Load and Process Test Data ---
    test_df = pd.read_csv(os.path.join(processed_datapath, 'test_FD001.csv'))
    rul_df = pd.read_csv(os.path.join(processed_datapath, 'RUL_FD001.csv'))
    logger.info("Loaded processed test and RUL data.")

    # --- Feature Engineering on Test Data (Must match training) ---
    window_size = 5
    feature_cols = ['setting_1', 'setting_2', 'setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    for col in feature_cols:
        test_df[f'{col}_mean'] = test_df.groupby('unit_number')[col].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
        test_df[f'{col}_std'] = test_df.groupby('unit_number')[col].rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True)
    
    test_df.fillna(0, inplace=True)
    logger.info("Engineered features for test data.")
    
    # The ground truth RUL is the last RUL value for each engine in the test set
    truth_rul = test_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    truth_rul = pd.merge(truth_rul, rul_df, left_index=True, right_index=True)
    truth_rul['RUL'] = truth_rul['time_in_cycles'] + truth_rul['RUL']
    y_true = truth_rul['RUL']

    # We need to predict only on the last cycle for each engine
    X_test = test_df.groupby('unit_number').last().reset_index()
    X_test = X_test.drop(columns=['unit_number', 'time_in_cycles'] + feature_cols)

    # --- Make Predictions ---
    logger.info(f"Making predictions on {X_test.shape[0]} engines...")
    y_pred = loaded_model.predict(X_test)

    # --- Evaluate Performance ---
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    logger.info("--- Test Set Evaluation ---")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE:  {mae:.4f}")
    logger.info(f"R2 Score:   {r2:.4f}")
    logger.info("---------------------------")
    
    return {"rmse": rmse, "mae": mae, "r2": r2}

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    processed_datapath = os.path.join(project_dir, 'data', 'processed')
    models_path = os.path.join(project_dir, 'models')

    predict_on_test_data(processed_datapath, models_path)
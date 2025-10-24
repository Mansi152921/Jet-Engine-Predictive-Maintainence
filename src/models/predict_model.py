# src/models/predict_model.py

import os
import logging
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- ADD THIS IMPORT ---
from pyspark.sql import SparkSession
# ---------------------

def predict_on_test_data(processed_datapath, uc_model_name):
    """
    Loads the best model from the UC registry, evaluates it, and
    saves the predictions to a Delta table.
    """
    logger = logging.getLogger(__name__)
    
    mlflow.set_registry_uri("databricks-uc")
    logger.info(f"Loading latest version of model '{uc_model_name}' from Unity Catalog...")

    client = MlflowClient()
    
    try:
        all_versions = client.search_model_versions(f"name='{uc_model_name}'")
        latest_version_obj = sorted(all_versions, key=lambda v: int(v.version), reverse=True)[0]
        latest_version = latest_version_obj.version
        logger.info(f"Found latest version: {latest_version}")
    except IndexError:
        logger.error(f"No model versions found for '{uc_model_name}'. Did the training job run?")
        raise
    
    logged_model_uri = f'models:/{uc_model_name}/{latest_version}'
    loaded_model = mlflow.pyfunc.load_model(logged_model_uri)
    logger.info(f"Model version {latest_version} loaded successfully.")

    # --- Load and Process Test Data from Volume ---
    test_df = pd.read_csv(os.path.join(processed_datapath, 'test_FD001.csv'))
    rul_df = pd.read_csv(os.path.join(processed_datapath, 'RUL_FD001.csv'))
    logger.info("Loaded processed test and RUL data from Volume.")

    # --- Feature Engineering on Test Data (Must match training) ---
    window_size = 5
    feature_cols = ['setting_1', 'setting_2', 'setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    for col in feature_cols:
        test_df[f'{col}_mean'] = test_df.groupby('unit_number')[col].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
        test_df[f'{col}_std'] = test_df.groupby('unit_number')[col].rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True)
    
    test_df.fillna(0, inplace=True)
    logger.info("Engineered features for test data.")
    
    truth_rul = test_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    truth_rul = pd.merge(truth_rul, rul_df, left_index=True, right_index=True)
    truth_rul['RUL'] = truth_rul['time_in_cycles'] + truth_rul['RUL']
    y_true = truth_rul['RUL']

    # --- START OF PREDICTION CHANGES ---
    # We keep the full test set to get 'unit_number' for the final table
    X_test_full_df = test_df.groupby('unit_number').last().reset_index()
    
    # We drop the columns just for the prediction step
    X_test_features_df = X_test_full_df.drop(columns=['unit_number', 'time_in_cycles'] + feature_cols)

    # --- Make Predictions ---
    logger.info(f"Making predictions on {X_test_features_df.shape[0]} engines...")
    y_pred = loaded_model.predict(X_test_features_df)
    # --- END OF PREDICTION CHANGES ---

    # --- Evaluate Performance ---
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    logger.info("--- Test Set Evaluation ---")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE:  {mae:.4f}")
    logger.info(f"R2 Score:   {r2:.4f}")
    logger.info("---------------------------")
    
    # --- START NEW SECTION: Save Predictions ---
    logger.info("Saving predictions to a Delta table...")

    # 1. Create a results DataFrame
    results_df = pd.DataFrame({
        'unit_number': X_test_full_df['unit_number'],
        'prediction': y_pred,
        'ground_truth_rul': y_true
    })

    # 2. Convert to Spark DataFrame
    spark = SparkSession.builder.getOrCreate()
    spark_results_df = spark.createDataFrame(results_df)
    
    # 3. Define the table name (e.g., in the same schema as the model)
    table_name = f"{uc_model_name}_predictions" 
    
    # 4. Save as a Delta table
    spark_results_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
    
    logger.info(f"Successfully saved predictions to '{table_name}'")
    # --- END NEW SECTION ---
    
    return {"rmse": rmse, "mae": mae, "r2": r2}

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # --- Databricks-specific Change ---
    # Define your Unity Catalog paths here.
    # !! UPDATE THESE PLACEHOLDERS to match your environment !!
    CATALOG_NAME = "jet_engine_catalog"
    SCHEMA_NAME = "dev_schema"
    VOLUME_NAME = "models_volume" # The name of your UC Volume

    # Define the absolute paths within your Volume
    processed_datapath = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/{VOLUME_NAME}/data/processed"
    
    # Define the 3-level name for your model in the UC Registry
    uc_model_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.PredictiveMaintenanceRUL"

    # Run the main function
    predict_on_test_data(processed_datapath, uc_model_name)
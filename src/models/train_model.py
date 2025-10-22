# src/models/train_model.py

import os
import logging
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

# Import all the models we want to train
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

from sklearn.metrics import mean_squared_error, r2_score

# --- Databricks-specific Change ---
# We now pass the 3-level Unity Catalog model name to the function
def train_multiple_models(final_datapath, uc_model_name):
    """
    Trains multiple models, compares them using MLflow, and registers the best one
    to the Unity Catalog Model Registry.
    """
    logger = logging.getLogger(__name__)

    # --- Databricks-specific Change ---
    # REMOVED: mlflow.set_tracking_uri(...) - Databricks handles this automatically.
    
    # --- Databricks-specific Change ---
    # Set the registry to Unity Catalog (recommended)
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    
    logger.info('Starting model training and selection process...')

    experiment_path = f"/Shared/{uc_model_name}"
    mlflow.set_experiment(experiment_name=experiment_path)

    # --- Load Final Data from UC Volume ---
    # The path is now a direct Volume path, not a relative OS path
    X_train_path = os.path.join(final_datapath, 'X_train.csv')
    y_train_path = os.path.join(final_datapath, 'y_train.csv')
    
    logger.info(f"Loading data from: {X_train_path}")
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    logger.info(f'Loaded training data with shape X: {X_train.shape}, y: {y_train.shape}')
    
    # --- Define Models to Train ---
    models = {
        "LinearRegression": LinearRegression(),
        "SVM": SVR(), # Support Vector Machine for Regression
"RandomForest": RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42),
        "LightGBM": lgb.LGBMRegressor(random_state=42)
    }

    # --- Start a Parent MLflow Run ---
    with mlflow.start_run(run_name="Model Comparison Experiment") as parent_run:
        logger.info(f"Parent Run ID: {parent_run.info.run_id}")
        mlflow.log_param("model_candidates", list(models.keys()))

        # --- Train Each Model in a Nested Run ---
        for model_name, model in models.items():
            with mlflow.start_run(run_name=f"Train_{model_name}", nested=True) as child_run:
                logger.info(f"--- Training {model_name} ---")
                
                mlflow.sklearn.autolog()
                
                model.fit(X_train, y_train)

                predictions = model.predict(X_train)
                rmse = np.sqrt(mean_squared_error(y_train, predictions))
                r2 = r2_score(y_train, predictions)

                logger.info(f"Finished training {model_name}. RMSE: {rmse:.4f}, R2: {r2:.4f}")

    # --- Find and Register the Best Model ---
    logger.info("--- Searching for the best model ---")
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=parent_run.info.experiment_id,
        filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'"
    )

    best_run = None
    best_rmse = float('inf')

    for run in runs:
        run_metrics = run.data.metrics
        metric_key = 'training_root_mean_squared_error' 
        if metric_key in run_metrics and run_metrics[metric_key] < best_rmse:
            best_rmse = run_metrics[metric_key]
            best_run = run

    if best_run:
        best_model_name = best_run.data.tags['mlflow.runName'].replace('Train_', '')
        logger.info(f"Best model is: {best_model_name} with RMSE: {best_rmse:.4f}")
        
        best_model_uri = f"runs:/{best_run.info.run_id}/model"
        
        # --- Databricks-specific Change ---
        # Register the model using the 3-level UC name
        model_version = mlflow.register_model(
            model_uri=best_model_uri,
            name=uc_model_name
        )
        logger.info(f"Registered best model as '{model_version.name}' version {model_version.version} to Unity Catalog")
    else:
        logger.warning("No best model found. Check MLflow runs.")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # --- Databricks-specific Change ---
    # Hardcode the paths to your Unity Catalog Volume.
    # !! UPDATE THESE PLACEHOLDERS !!
    CATALOG_NAME = "jet_engine_catalog"
    SCHEMA_NAME = "dev_schema"
    VOLUME_NAME = "models_volume" # The name of your UC Volume
    
    # Path inside your Volume where the data is stored
    DATA_SUB_PATH = "data/final" 
    
    # Define the final paths
    final_datapath = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/{VOLUME_NAME}/{DATA_SUB_PATH}"
    
    # Define the 3-level name for your model in the UC Registry
    uc_model_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.PredictiveMaintenanceRUL"

    # Call the main function with the Databricks paths
    train_multiple_models(final_datapath, uc_model_name)
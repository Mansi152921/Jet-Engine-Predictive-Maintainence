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

def train_multiple_models(final_datapath, models_path):
    """
    Trains multiple models, compares them using MLflow, and registers the best one.
    """
    logger = logging.getLogger(__name__)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    logger.info('Starting model training and selection process...')

    # --- Load Final Data ---
    X_train = pd.read_csv(os.path.join(final_datapath, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(final_datapath, 'y_train.csv')).values.ravel()
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

    # Find the run with the lowest RMSE
    for run in runs:
        run_metrics = run.data.metrics
        # Note: autologger for scikit-learn logs training RMSE as 'training_root_mean_squared_error'
        metric_key = 'training_root_mean_squared_error' 
        if metric_key in run_metrics and run_metrics[metric_key] < best_rmse:
            best_rmse = run_metrics[metric_key]
            best_run = run

    if best_run:
        best_model_name = best_run.data.tags['mlflow.runName'].replace('Train_', '')
        logger.info(f"Best model is: {best_model_name} with RMSE: {best_rmse:.4f}")
        
        best_model_uri = f"runs:/{best_run.info.run_id}/model"
        model_version = mlflow.register_model(
            model_uri=best_model_uri,
            name="PredictiveMaintenanceRUL"
        )
        logger.info(f"Registered best model as '{model_version.name}' version {model_version.version}")
    else:
        logger.warning("No best model found. Check MLflow runs.")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    final_datapath = os.path.join(project_dir, 'data', 'final')
    models_path = os.path.join(project_dir, 'models')

    train_multiple_models(final_datapath, models_path)
# src/models/train_model.py

import os
import logging
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def train_model(final_datapath, models_path):
    """
    Trains the LSTM model, evaluates it, and logs everything with MLflow.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    logger = logging.getLogger(__name__)
    logger.info('Starting model training...')

    # --- Load Final Data ---
    X_train_path = os.path.join(final_datapath, 'X_train.npy')
    y_train_path = os.path.join(final_datapath, 'y_train.npy')
    
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    logger.info(f'Loaded training data with shape X: {X_train.shape}, y: {y_train.shape}')
    
    # --- Define Model Architecture ---
    model = Sequential()
    model.add(LSTM(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        units=100,
        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(
        units=50,
        return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    logger.info('Model architecture defined and compiled.')
    print(model.summary())

    # --- MLflow Integration ---
    # Autologging is the easiest way to capture everything automatically.
    # It logs parameters, metrics from model.fit, and the final model artifact.
    mlflow.tensorflow.autolog()

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f'Starting MLflow run with ID: {run_id}')
        
        # --- Train the Model ---
        history = model.fit(X_train, y_train,
                            epochs=10, # Kept low for a quick test run; increase as needed
                            batch_size=200,
                            validation_split=0.10,
                            verbose=1)

        logger.info('Model training complete.')
        
        # --- Save the final model manually (optional, as autolog handles it) ---
        # model_save_path = os.path.join(models_path, 'final_model.h5')
        # model.save(model_save_path)
        # logger.info(f'Final model saved to {model_save_path}')

    return run_id


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Define project directory paths
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    final_datapath = os.path.join(project_dir, 'data', 'final')
    models_path = os.path.join(project_dir, 'models')

    train_model(final_datapath, models_path)
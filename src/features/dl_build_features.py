# src/features/build_features.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import logging
import joblib

def generate_sequences(df: pd.DataFrame, sequence_length: int, sequence_cols: list):
    """
    Generates sequences from the dataframe for time-series prediction.
    """
    data_array = df[sequence_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements - sequence_length + 1), range(sequence_length, num_elements + 1)):
        yield data_array[start:stop, :]

def generate_labels(df: pd.DataFrame, sequence_length: int, label: list):
    """
    Generates labels for the sequences.
    """
    data_array = df[label].values
    num_elements = data_array.shape[0]
    return data_array[sequence_length-1:num_elements, :]

def build_features(processed_datapath, output_datapath, models_path):
    """
    Loads processed data, engineers features, and saves the final feature set.
    """
    logger = logging.getLogger(__name__)
    logger.info('Starting feature engineering...')

    # --- Load Processed Data ---
    train_df = pd.read_csv(os.path.join(processed_datapath, 'train_FD001.csv'))
    test_df = pd.read_csv(os.path.join(processed_datapath, 'test_FD001.csv'))
    logger.info('Loaded processed train and test data.')

    # --- Feature Engineering: Calculate RUL for Training Data ---
    # Find the maximum cycle for each engine unit
    max_cycles = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycles']
    # Merge max cycles back into the training data
    train_df = pd.merge(train_df, max_cycles, on='unit_number', how='left')
    # Calculate RUL
    train_df['RUL'] = train_df['max_cycles'] - train_df['time_in_cycles']
    train_df.drop(columns=['max_cycles'], inplace=True)
    logger.info('Calculated RUL for training data.')

    # --- Feature Scaling ---
    # Define columns to be scaled (settings and sensors)
    feature_cols = ['setting_1', 'setting_2', 'setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    # Initialize and fit the scaler ONLY on the training data
    scaler = MinMaxScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    # Transform the test data using the SAME scaler
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    logger.info('Scaled features using MinMaxScaler.')

    # Save the scaler object for future use during inference
    os.makedirs(models_path, exist_ok=True)
    scaler_path = os.path.join(models_path, 'scaler.gz')
    joblib.dump(scaler, scaler_path)
    logger.info(f'Scaler saved to {scaler_path}')
    
    # --- Generate Sequences ---
    sequence_length = 50
    sequence_cols = feature_cols 
    
    # Generate sequences for training
    train_sequences = list(generate_sequences(train_df, sequence_length, sequence_cols))
    X_train = np.array(train_sequences)
    
    # Generate labels for training
    train_labels = list(generate_labels(train_df, sequence_length, ['RUL']))
    y_train = np.array(train_labels).reshape(-1, 1)

    logger.info(f'Generated training sequences with shape {X_train.shape}')
    logger.info(f'Generated training labels with shape {y_train.shape}')

    # Save the final feature-engineered data
    os.makedirs(output_datapath, exist_ok=True)
    np.save(os.path.join(output_datapath, 'X_train.npy'), X_train)
    np.save(os.path.join(output_datapath, 'y_train.npy'), y_train)
    
    # We will process the test data similarly during the prediction phase
    # For now, we only need the processed test_df for final evaluation
    test_df.to_csv(os.path.join(output_datapath, 'test_features.csv'), index=False)
    
    logger.info('Feature engineering complete. Final datasets saved.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Define project directory paths
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    processed_datapath = os.path.join(project_dir, 'data', 'processed')
    output_datapath = os.path.join(project_dir, 'data', 'final')
    models_path = os.path.join(project_dir, 'models')

    build_features(processed_datapath, output_datapath, models_path)
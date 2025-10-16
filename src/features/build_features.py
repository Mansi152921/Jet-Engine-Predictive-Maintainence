# src/features/build_features.py

import pandas as pd
import numpy as np
import os
import logging
import joblib
from sklearn.preprocessing import MinMaxScaler

def build_features(processed_datapath, output_datapath, models_path):
    """
    Loads processed data, engineers tabular features for ML models,
    and saves the final feature set.
    """
    logger = logging.getLogger(__name__)
    logger.info('Starting feature engineering for ML models...')

    # --- Load Processed Data ---
    train_df = pd.read_csv(os.path.join(processed_datapath, 'train_FD001.csv'))
    test_df = pd.read_csv(os.path.join(processed_datapath, 'test_FD001.csv'))
    rul_df = pd.read_csv(os.path.join(processed_datapath, 'RUL_FD001.csv'))
    logger.info('Loaded processed train, test, and RUL data.')

    # --- Feature Engineering: Calculate RUL for Training Data ---
    max_cycles = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycles']
    train_df = pd.merge(train_df, max_cycles, on='unit_number', how='left')
    train_df['RUL'] = train_df['max_cycles'] - train_df['time_in_cycles']
    train_df.drop(columns=['max_cycles'], inplace=True)
    logger.info('Calculated RUL for training data.')

    # --- Feature Engineering: Create Rolling Features ---
    window_size = 5
    feature_cols = ['setting_1', 'setting_2', 'setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    # Group by engine unit to calculate rolling stats correctly
    for col in feature_cols:
        train_df[f'{col}_mean'] = train_df.groupby('unit_number')[col].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
        train_df[f'{col}_std'] = train_df.groupby('unit_number')[col].rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True)
    
    train_df.fillna(0, inplace=True) # Fill NaNs created by std dev on single data points
    logger.info(f'Created rolling features with window size {window_size}')
    
    # --- Prepare Final Datasets ---
    # We don't need the original sensor readings anymore, just the stats
    y_train = train_df['RUL']
    X_train = train_df.drop(columns=['RUL', 'unit_number', 'time_in_cycles'] + feature_cols)
    
    # We will process the test data similarly during the prediction phase.
    
    # Save the final feature-engineered data
    os.makedirs(output_datapath, exist_ok=True)
    X_train.to_csv(os.path.join(output_datapath, 'X_train.csv'), index=False)
    y_train.to_csv(os.path.join(output_datapath, 'y_train.csv'), index=False)
    
    logger.info('Feature engineering complete. Final datasets saved as CSV.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    processed_datapath = os.path.join(project_dir, 'data', 'processed')
    output_datapath = os.path.join(project_dir, 'data', 'final')
    models_path = os.path.join(project_dir, 'models')

    build_features(processed_datapath, output_datapath, models_path)
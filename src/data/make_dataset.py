# src/data/make_dataset.py

import pandas as pd
import os
import logging

def make_dataset(input_filepath, output_filepath):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Starting data loading and processing...') 

    # Define column names based on the dataset's documentation
    index_names = ['unit_number', 'time_in_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = [f'sensor_{i}' for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    # --- Load Training Data ---
    train_path = os.path.join(input_filepath, 'train_FD001.txt')
    logger.info(f'Loading training data from {train_path}')
    train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=col_names)

    # --- Load Test Data ---
    test_path = os.path.join(input_filepath, 'test_FD001.txt')
    logger.info(f'Loading test data from {test_path}')
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=col_names)

    # --- Load RUL (Ground Truth) Data for the Test Set ---
    rul_path = os.path.join(input_filepath, 'RUL_FD001.txt')
    logger.info(f'Loading RUL data from {rul_path}')
    rul_df = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])

    # --- Save the processed dataframes to the output folder ---
    os.makedirs(output_filepath, exist_ok=True)
    
    train_output_path = os.path.join(output_filepath, 'train_FD001.csv')
    train_df.to_csv(train_output_path, index=False)
    logger.info(f'Saved processed training data to {train_output_path}')

    test_output_path = os.path.join(output_filepath, 'test_FD001.csv')
    test_df.to_csv(test_output_path, index=False)
    logger.info(f'Saved processed test data to {test_output_path}')
    
    rul_output_path = os.path.join(output_filepath, 'RUL_FD001.csv')
    rul_df.to_csv(rul_output_path, index=False)
    logger.info(f'Saved RUL data to {rul_output_path}')

    logger.info('Data processing complete.')


if __name__ == '__main__':
    # Set up logging
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Define project directory paths
    # Assumes the script is run from the root of the project directory
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    input_filepath = os.path.join(project_dir, 'data', 'raw')
    output_filepath = os.path.join(project_dir, 'data', 'processed')

    # Run the main function
    make_dataset(input_filepath, output_filepath) 
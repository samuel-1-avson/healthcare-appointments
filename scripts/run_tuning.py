"""
Script to run hyperparameter tuning for the No-Show Prediction model.
"""
import sys
import os
import logging
import pandas as pd
import yaml
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ml.preprocessing import NoShowPreprocessor
from src.ml.tuning import HyperparameterTuner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    logger.info("Starting hyperparameter tuning...")
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'ml_config.yaml')
    config = load_config(config_path)
    
    # Handle missing paths in config
    if 'paths' not in config:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        config['paths'] = {
            'features_data': os.path.join(project_root, 'data', 'processed', 'appointments_features.csv')
        }
    
    # Load data
    data_path = config['paths']['features_data']
    if not os.path.exists(data_path):
        # Fallback to absolute path if relative path fails
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'appointments_features.csv')
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Initialize preprocessor
    preprocessor = NoShowPreprocessor(config)
    
    # Prepare data (select features, split)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    
    # Fit and transform training data
    logger.info("Preprocessing training data...")
    X_train_transformed = preprocessor.fit_transform(X_train, y_train)
    
    # Initialize tuner
    tuner = HyperparameterTuner(config, preprocessor=preprocessor)
    
    # Run tuning for XGBoost
    logger.info("Tuning XGBoost model...")
    tuner.tune_model('xgboost', X_train_transformed, y_train)
    
    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'tuning')
    tuner.save_results(output_dir)
    
    logger.info("Tuning complete!")

if __name__ == "__main__":
    main()

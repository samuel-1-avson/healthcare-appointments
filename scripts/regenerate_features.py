
import sys
import pandas as pd
import yaml
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data_cleaner import DataCleaner
from src.feature_engineer import FeatureEngineer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    config_path = project_root / 'config' / 'ml_config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    logger.info("Starting feature regeneration...")
    
    # Load config
    config = load_config()
    
    # Update paths in config if they are relative
    # Assuming config structure based on usage in other files
    # We might need to manually set paths if they are not in the config file we read
    # Let's check if 'paths' key exists, if not we define them
    if 'paths' not in config:
        config['paths'] = {
            'raw_data': project_root / 'data' / 'raw' / 'appointments_raw.csv',
            'processed_data': project_root / 'data' / 'processed' / 'appointments_cleaned.csv',
            'features_data': project_root / 'data' / 'processed' / 'appointments_features.csv'
        }
    else:
        # Resolve paths relative to project root if they are strings
        for key, path_str in config['paths'].items():
            if isinstance(path_str, str):
                config['paths'][key] = project_root / path_str

    # Also check cleaning config
    if 'cleaning' not in config:
        config['cleaning'] = {
            'column_mapping': {
                'PatientId': 'patientid',
                'AppointmentID': 'appointmentid',
                'Gender': 'gender',
                'ScheduledDay': 'scheduledday',
                'AppointmentDay': 'appointmentday',
                'Age': 'age',
                'Neighbourhood': 'neighbourhood',
                'Scholarship': 'scholarship',
                'Hipertension': 'hypertension',
                'Diabetes': 'diabetes',
                'Alcoholism': 'alcoholism',
                'Handcap': 'handicap',
                'SMS_received': 'sms_received',
                'No-show': 'no_show'
            },
            'date_columns': ['scheduledday', 'appointmentday'],
            'min_age': 0,
            'max_age': 120
        }
        
    logger.info(f"Config keys: {config.keys()}")
    
    # And features config
    if 'features' not in config:
        logger.warning("'features' key missing in config, creating it.")
        config['features'] = {}
    
    # Ensure features config has the bins needed
    if 'age_bins' not in config['features']:
         config['features']['age_bins'] = [-1, 12, 18, 30, 50, 70, 120]
         config['features']['age_labels'] = ['Child', 'Teen', 'Young Adult', 'Middle Age', 'Senior', 'Elderly']
    
    if 'lead_time_bins' not in config['features']:
         config['features']['lead_time_bins'] = [-1, 0, 2, 7, 14, 30, 365]
         config['features']['lead_time_labels'] = ['Same Day', '1-2 days', '3-7 days', '8-14 days', '15-30 days', '30+ days']

    # Load Raw Data
    raw_data_path = config['paths']['raw_data']
    if not Path(raw_data_path).exists():
        # Try fallback
        raw_data_path = project_root / 'data' / 'raw' / 'appointments_raw.csv'
    
    if not Path(raw_data_path).exists():
        logger.error(f"Raw data not found at {raw_data_path}")
        return

    logger.info(f"Loading raw data from {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    
    # Clean Data
    cleaner = DataCleaner(config)
    df_clean = cleaner.clean_pipeline(df)
    
    # Engineer Features
    engineer = FeatureEngineer(config)
    df_features = engineer.engineer_all_features(df_clean)
    
    # Save
    output_path = config['paths']['features_data']
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    df_features.to_csv(output_path, index=False)
    logger.info(f"Saved generated features to {output_path}")
    logger.info("Regeneration complete.")

if __name__ == "__main__":
    main()

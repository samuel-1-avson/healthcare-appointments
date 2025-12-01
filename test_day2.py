# test_day2.py
"""Test script for Day 2 modules."""

import pandas as pd
from src.utils import load_config, setup_logging
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.feature_engineer import FeatureEngineer

def main():
    """Test the cleaning and feature engineering pipeline."""
    
    print("\n" + "="*60)
    print("TESTING DAY 2: CLEANING & FEATURE ENGINEERING")
    print("="*60)
    
    # Setup
    config = load_config("config/config.yaml")
    logger = setup_logging(config['logging']['level'])
    
    # Load data
    print("\n1. LOADING DATA...")
    loader = DataLoader(config)
    df_raw = loader.load(source="auto")
    print(f"   ✅ Loaded {len(df_raw):,} rows")
    
    # Clean data
    print("\n2. CLEANING DATA...")
    cleaner = DataCleaner(config)
    df_clean = cleaner.clean_pipeline(df_raw)
    print(f"   ✅ Cleaned to {len(df_clean):,} rows")
    
    # Engineer features
    print("\n3. ENGINEERING FEATURES...")
    engineer = FeatureEngineer(config)
    df_features = engineer.engineer_all_features(df_clean)
    print(f"   ✅ Created {len(engineer.get_features_created())} new features")
    print(f"   ✅ Final shape: {df_features.shape}")
    
    # Display sample of new features
    print("\n4. SAMPLE OF NEW FEATURES:")
    feature_cols = engineer.get_features_created()[:10]  # First 10 features
    if feature_cols:
        print(df_features[feature_cols].head())
    
    # Display cleaning report
    print("\n5. CLEANING REPORT:")
    report = cleaner.get_cleaning_report()
    for key, value in report.items():
        if not isinstance(value, list) and not isinstance(value, dict):
            print(f"   • {key}: {value}")
    
    print("\n✅ DAY 2 MODULES WORKING CORRECTLY!")
    print("="*60)
    
    return df_features


if __name__ == "__main__":
    df = main()
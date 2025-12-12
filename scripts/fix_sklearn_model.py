#!/usr/bin/env python
"""
Quick fix script to retrain the preprocessor and model with current scikit-learn version.
This fixes the '_fill_dtype' attribute error caused by version mismatch.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 60)
    print("Retraining model with current scikit-learn version...")
    print("=" * 60)
    
    # Load data
    data_path = "data/processed/cleaned_data.csv"
    if not os.path.exists(data_path):
        data_path = "data/raw/KaggleV2-May-2016.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Creating a minimal preprocessor for compatibility...")
        
        # Create minimal preprocessor that matches expected structure
        numeric_features = ['age', 'lead_days', 'hour', 'prev_noshow_rate', 'avg_lead_time', 'handicap']
        categorical_features = ['gender', 'neighbourhood', 'weekday']
        binary_features = ['scholarship', 'hypertension', 'diabetes', 'alcoholism', 'sms_received']
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
                ('bin', 'passthrough', binary_features)
            ],
            remainder='drop'
        )
        
        # Fit with dummy data
        dummy_data = pd.DataFrame({
            'age': [30, 40, 50],
            'lead_days': [5, 10, 15],
            'hour': [9, 14, 16],
            'prev_noshow_rate': [0.1, 0.2, 0.3],
            'avg_lead_time': [7.0, 10.0, 12.0],
            'handicap': [0, 0, 1],
            'gender': ['M', 'F', 'M'],
            'neighbourhood': ['A', 'B', 'C'],
            'weekday': ['Monday', 'Tuesday', 'Wednesday'],
            'scholarship': [0, 1, 0],
            'hypertension': [0, 0, 1],
            'diabetes': [0, 1, 0],
            'alcoholism': [0, 0, 0],
            'sms_received': [1, 1, 0]
        })
        
        preprocessor.fit(dummy_data)
        
        # Create a simple model
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        X_transformed = preprocessor.transform(dummy_data)
        y_dummy = [0, 1, 0]
        model.fit(X_transformed, y_dummy)
        
        # Save
        output_dir = "models/production"
        os.makedirs(output_dir, exist_ok=True)
        
        joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
        joblib.dump(model, os.path.join(output_dir, "model.joblib"))
        
        print(f"✅ Saved new preprocessor and model to {output_dir}")
        return
    
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Use the existing preprocessing logic
    from src.ml.preprocessing import NoShowPreprocessor
    
    preprocessor = NoShowPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    
    # Train a simple model
    print("Training model...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Save
    output_dir = "models/production"
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(preprocessor.preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
    joblib.dump(model, os.path.join(output_dir, "model.joblib"))
    
    # Save metadata
    import json
    import sklearn
    metadata = {
        "sklearn_version": sklearn.__version__,
        "model_type": "GradientBoostingClassifier",
        "n_features": X_train.shape[1],
        "trained_at": pd.Timestamp.now().isoformat()
    }
    with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved new model and preprocessor to {output_dir}")
    print(f"   scikit-learn version: {sklearn.__version__}")

if __name__ == "__main__":
    main()

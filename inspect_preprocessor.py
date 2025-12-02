import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def inspect_preprocessor():
    path = Path('models/production/preprocessor.joblib')
    if not path.exists():
        print(f"File not found: {path}")
        return

    print(f"Loading {path}...")
    data = joblib.load(path)
    
    if isinstance(data, dict):
        print("Artifact is a dictionary.")
        print(f"Keys: {data.keys()}")
        preprocessor = data.get('preprocessor')
        print(f"Valid Numeric: {len(data.get('valid_numeric', []))}")
        print(f"Valid Categorical: {len(data.get('valid_categorical', []))}")
        print(f"Valid Binary: {len(data.get('valid_binary', []))}")
    else:
        print("Artifact is not a dictionary.")
        preprocessor = data

    print(f"\nPreprocessor Type: {type(preprocessor)}")
    
    if hasattr(preprocessor, 'transformers_'):
        print("\nTransformers:")
        for name, trans, cols in preprocessor.transformers_:
            print(f"  - Name: {name}")
            print(f"    Type: {type(trans)}")
            print(f"    Columns: {cols}")
            
    if hasattr(preprocessor, 'feature_names_in_'):
        print(f"\nFeature Names In ({len(preprocessor.feature_names_in_)}):")
        print(preprocessor.feature_names_in_)
    else:
        print("\nNo feature_names_in_ attribute found.")

    if hasattr(preprocessor, 'n_features_in_'):
        print(f"\nN Features In: {preprocessor.n_features_in_}")

if __name__ == "__main__":
    inspect_preprocessor()

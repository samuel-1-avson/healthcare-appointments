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
        preprocessor = data.get('preprocessor')
        vn = data.get('valid_numeric', [])
        vc = data.get('valid_categorical', [])
        vb = data.get('valid_binary', [])
        print(f"Valid Numeric Count: {len(vn)}")
        print(f"Valid Categorical Count: {len(vc)}")
        print(f"Valid Binary Count: {len(vb)}")
        print(f"Total Metadata Features: {len(vn) + len(vc) + len(vb)}")
        print(f"Valid Numeric: {vn}")
        print(f"Valid Categorical: {vc}")
        print(f"Valid Binary: {vb}")
    else:
        print("Artifact is not a dictionary.")
        preprocessor = data

    print(f"\nPreprocessor Type: {type(preprocessor)}")
    
    if hasattr(preprocessor, 'n_features_in_'):
        print(f"\nN Features In (Expected by Preprocessor): {preprocessor.n_features_in_}")
        
    if hasattr(preprocessor, 'feature_names_in_'):
        print(f"\nFeature Names In ({len(preprocessor.feature_names_in_)}):")
        print(preprocessor.feature_names_in_)

if __name__ == "__main__":
    inspect_preprocessor()

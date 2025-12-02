import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def repro():
    path = Path('models/production/preprocessor.joblib')
    if not path.exists():
        print(f"File not found: {path}")
        return

    print(f"Loading {path}...")
    data = joblib.load(path)
    preprocessor = data.get('preprocessor')
    
    valid_numeric = data.get('valid_numeric', [])
    valid_categorical = data.get('valid_categorical', [])
    valid_binary = data.get('valid_binary', [])
    input_features = valid_numeric + valid_categorical + valid_binary
    
    print(f"Expected features ({len(input_features)}): {input_features}")
    
    # Create dummy dataframe with expected features
    df = pd.DataFrame(columns=input_features)
    # Add one row of dummy data
    for col in valid_numeric:
        df.loc[0, col] = 0
    for col in valid_categorical:
        df.loc[0, col] = 'dummy'
    for col in valid_binary:
        df.loc[0, col] = 0
        
    # Add extra columns to simulate 44 features
    for i in range(14):
        df[f'extra_{i}'] = 0
        
    print(f"Created DataFrame with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    print(f"N Features In: {getattr(preprocessor, 'n_features_in_', 'Not found')}")
    import sklearn
    print(f"Sklearn version: {sklearn.__version__}")

    try:
        print("\n--- Test 1: DataFrame with 44 columns ---")
        X = preprocessor.transform(df)
        print("SUCCESS")
    except Exception as e:
        print(f"FAILED: {e}")

    try:
        print("\n--- Test 2: Numpy Array with 44 columns ---")
        arr_44 = np.random.rand(1, 44)
        X = preprocessor.transform(arr_44)
        print("SUCCESS")
    except Exception as e:
        print(f"FAILED: {e}")

    try:
        print("\n--- Test 3: Numpy Array with 30 columns ---")
        arr_30 = np.random.rand(1, 30)
        X = preprocessor.transform(arr_30)
        print("SUCCESS")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    repro()

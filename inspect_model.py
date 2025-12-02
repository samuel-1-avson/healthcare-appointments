import joblib
import sys
import os
import pandas as pd
import numpy as np

def inspect():
    print("Inspecting artifacts...")
    try:
        model = joblib.load('models/production/model.joblib')
        print(f"Model type: {type(model)}")
        if hasattr(model, 'steps'):
            print("Model is a Pipeline. Steps:")
            for name, step in model.steps:
                print(f"  - {name}: {type(step)}")
        else:
            print("Model is NOT a Pipeline.")
            
    except Exception as e:
        print(f"Failed to load model: {e}")

    try:
        preprocessor = joblib.load('models/production/preprocessor.joblib')
        print(f"Preprocessor type: {type(preprocessor)}")
        if isinstance(preprocessor, dict):
            print("Preprocessor is a dict.")
        else:
            if hasattr(preprocessor, 'transformers_'):
                print("Preprocessor has transformers.")
            if hasattr(preprocessor, 'n_features_in_'):
                print(f"n_features_in_: {preprocessor.n_features_in_}")
    except Exception as e:
        print(f"Failed to load preprocessor: {e}")

if __name__ == "__main__":
    inspect()

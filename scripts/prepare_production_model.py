#!/usr/bin/env python
"""
Prepare Production Model
========================
Copy and configure the best model for production deployment.

Usage:
    python scripts/prepare_production_model.py
    python scripts/prepare_production_model.py --model models/tuned/xgboost_tuned.joblib
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def prepare_production_model(
    model_path: str,
    preprocessor_path: str,
    output_dir: str = "models/production"
):
    """
    Prepare model for production deployment.
    
    Parameters
    ----------
    model_path : str
        Path to trained model
    preprocessor_path : str
        Path to preprocessor
    output_dir : str
        Output directory for production model
    """
    import joblib
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("Preparing Production Model")
    print("=" * 50)
    
    # Copy model
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    dest_model = output_dir / "model.joblib"
    shutil.copy(model_path, dest_model)
    print(f"✅ Model copied to {dest_model}")
    
    # Copy preprocessor
    preprocessor_path = Path(preprocessor_path)
    if preprocessor_path.exists():
        dest_preprocessor = output_dir / "preprocessor.joblib"
        shutil.copy(preprocessor_path, dest_preprocessor)
        print(f"✅ Preprocessor copied to {dest_preprocessor}")
    else:
        print(f"⚠️ Preprocessor not found: {preprocessor_path}")
    
    # Load model to get info
    model = joblib.load(model_path)
    model_type = type(model).__name__
    
    # Create metadata
    metadata = {
        "model_name": model_path.stem.replace("_tuned", ""),
        "model_version": "1.0.0",
        "model_type": model_type,
        "source_path": str(model_path),
        "created_at": datetime.now().isoformat(),
        "description": "Production model for no-show prediction"
    }
    
    # Try to load training metrics if available
    metrics_path = model_path.parent.parent / "evaluation" / "evaluation_results.csv"
    if metrics_path.exists():
        import pandas as pd
        metrics_df = pd.read_csv(metrics_path)
        model_name = metadata["model_name"]
        model_metrics = metrics_df[metrics_df['Model'].str.contains(model_name, case=False, na=False)]
        if not model_metrics.empty:
            metadata["metrics"] = model_metrics.iloc[0].to_dict()
    
    # Save metadata
    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved to {metadata_path}")
    
    print("\n" + "=" * 50)
    print("Production Model Ready!")
    print("=" * 50)
    print(f"Location: {output_dir}")
    print(f"Model: {metadata['model_name']} ({metadata['model_type']})")
    print("\nFiles:")
    for f in output_dir.iterdir():
        print(f"  - {f.name}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Prepare model for production")
    
    parser.add_argument(
        "--model",
        type=str,
        default="models/tuned/random_forest_tuned.joblib",
        help="Path to model file"
    )
    parser.add_argument(
        "--preprocessor",
        type=str,
        default="models/tuned/preprocessor.joblib",
        help="Path to preprocessor"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/production",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    prepare_production_model(
        args.model,
        args.preprocessor,
        args.output
    )


if __name__ == "__main__":
    main()
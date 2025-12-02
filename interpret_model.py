#!/usr/bin/env python
"""
Model Interpretation CLI
========================
Generate interpretability analysis for a trained model.

Usage:
    python interpret_model.py --model models/tuned/random_forest.joblib
    python interpret_model.py --model-dir models/tuned/
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import yaml
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent))

from src.ml.preprocessing import NoShowPreprocessor
from src.ml.interpret import ModelInterpreter, create_interpretation_report


def main():
    parser = argparse.ArgumentParser(description="Generate model interpretability analysis")
    
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model (.joblib)")
    parser.add_argument("--preprocessor", type=str, default=None,
                       help="Path to preprocessor (.joblib)")
    parser.add_argument("--data", type=str,
                       default="data/processed/appointments_features.csv")
    parser.add_argument("--config", type=str,
                       default="config/ml_config.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("interpret_model")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model = joblib.load(args.model)
    model_name = Path(args.model).stem
    
    # Load preprocessor
    if args.preprocessor:
        preprocessor = NoShowPreprocessor.load(args.preprocessor)
    else:
        # Try to find preprocessor in same directory
        model_dir = Path(args.model).parent
        preprocessor_path = model_dir / "preprocessor.joblib"
        if preprocessor_path.exists():
            preprocessor = NoShowPreprocessor.load(preprocessor_path)
        else:
            logger.error("Preprocessor not found. Please specify with --preprocessor")
            sys.exit(1)
    
    # Load and prepare data
    logger.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    
    X, y = preprocessor.select_features(df)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Limit samples
    max_samples = min(args.max_samples, len(X_test_transformed))
    X_sample = X_test_transformed[:max_samples]
    y_sample = y_test[:max_samples]
    
    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else \
                 Path("outputs/interpretability") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize interpreter
    interpreter = ModelInterpreter(config)
    interpreter.output_dir = output_dir / "figures"
    interpreter.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fit interpreter
    logger.info("Fitting interpreter...")
    background_size = min(500, len(X_test_transformed))
    interpreter.fit(
        model,
        X_test_transformed[:background_size],
        feature_names=preprocessor.feature_names_,
        model_name=model_name
    )
    
    # Generate all plots
    logger.info("Generating interpretability plots...")
    interpreter.generate_all_plots(X_sample, y_sample, X_sample[:1])
    
    # Generate report
    report_path = output_dir / "interpretation_report.txt"
    report = create_interpretation_report(
        interpreter, X_sample, y_sample, report_path
    )
    
    # Save importance data
    interpreter.save_results(output_dir / "data")
    
    print(f"\n✅ Interpretation complete!")
    print(f"   Figures: {interpreter.output_dir}")
    print(f"   Report: {report_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TOP 10 FEATURES (by SHAP importance)")
    print("=" * 60)
    shap_imp = interpreter.get_shap_importance()
    if shap_imp:
        print(shap_imp.to_dataframe().head(10).to_string(index=False))


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Hyperparameter Tuning CLI
=========================
Dedicated script for hyperparameter tuning.

Usage:
    python tune_model.py                              # Tune all models
    python tune_model.py --model random_forest        # Tune specific model
    python tune_model.py --n-iter 100 --cv 10         # Custom settings

Examples:
    # Quick tuning with fewer iterations
    python tune_model.py --n-iter 20
    
    # Thorough tuning for production
    python tune_model.py --n-iter 100 --cv 10 --model xgboost
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
from src.ml.tuning import HyperparameterTuner
from src.ml.evaluate import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for no-show models")
    
    parser.add_argument("--data", type=str, 
                       default="data/processed/appointments_features.csv")
    parser.add_argument("--config", type=str, 
                       default="config/ml_config.yaml")
    parser.add_argument("--model", type=str, nargs='+', default=None,
                       help="Model(s) to tune")
    parser.add_argument("--n-iter", type=int, default=50,
                       help="Number of random search iterations")
    parser.add_argument("--cv", type=int, default=5,
                       help="Number of CV folds")
    parser.add_argument("--scoring", type=str, default="roc_auc",
                       help="Scoring metric for optimization")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("tune_model")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with CLI args
    config['tuning']['n_iter'] = args.n_iter
    config['tuning']['cv_folds'] = args.cv
    config['tuning']['scoring'] = args.scoring
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    logger.info(f"Loaded {len(df):,} samples")
    
    # Prepare data
    logger.info("Preparing data...")
    preprocessor = NoShowPreprocessor(config)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    X_train_transformed = preprocessor.fit_transform(X_train, y_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Initialize tuner
    tuner = HyperparameterTuner(config)
    
    # Run tuning
    logger.info("Starting hyperparameter tuning...")
    results = tuner.tune_all_models(
        X_train_transformed,
        y_train,
        model_names=args.model
    )
    
    # Display results
    comparison = tuner.get_comparison_table()
    print("\n" + "=" * 60)
    print("TUNING RESULTS")
    print("=" * 60)
    print(comparison.to_string(index=False))
    
    # Best model
    best_name, best_model = tuner.get_best_model()
    print(f"\n🏆 Best Model: {best_name}")
    print(f"   Best {args.scoring}: {tuner.results[best_name].best_score:.4f}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    evaluator = ModelEvaluator(config)
    test_results = evaluator.evaluate_all(
        tuner.best_models,
        X_test_transformed,
        y_test
    )
    
    test_comparison = evaluator.get_comparison_table()
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(test_comparison.to_string(index=False))
    
    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else \
                 Path("outputs/tuning") / datetime.now().strftime("%Y%m%d_%H%M%S")
    
    tuner.save_results(output_dir)
    
    # Save preprocessor
    preprocessor.save(output_dir / "preprocessor.joblib")
    
    print(f"\n✅ Results saved to: {output_dir}")
    
    return tuner, test_results


if __name__ == "__main__":
    main()
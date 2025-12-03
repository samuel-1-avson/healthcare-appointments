#!/usr/bin/env python
"""
Model Training CLI
==================
Command-line interface for training no-show prediction models.

This script supports:
- Baseline model training
- Hyperparameter tuning
- Model evaluation
- Interpretability analysis

Usage:
    python train_model.py                              # Basic training
    python train_model.py --tune                       # With hyperparameter tuning
    python train_model.py --tune --interpret           # Full pipeline
    python train_model.py --model xgboost --evaluate   # Specific model

Examples:
    # Quick baseline training
    python train_model.py --evaluate
    
    # Full pipeline with tuning and interpretation
    python train_model.py --tune --interpret --evaluate --verbose
    
    # Train and tune specific model
    python train_model.py --model random_forest --tune --evaluate
"""

import argparse
import logging
import sys
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import yaml
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.preprocessing import NoShowPreprocessor
from src.ml.train import ModelTrainer
from src.ml.evaluate import ModelEvaluator
from src.ml.tuning import HyperparameterTuner
from src.ml.interpret import ModelInterpreter, create_interpretation_report
from src.utils import setup_logging, timer


class TrainingPipeline:
    """
    Complete training pipeline orchestrator.
    
    This class manages the full ML training workflow including:
    - Data loading and preprocessing
    - Model training (baseline or tuned)
    - Evaluation
    - Interpretability analysis
    - Results persistence
    """
    
    def __init__(
        self,
        config_path: str = "config/ml_config.yaml",
        data_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the training pipeline.
        
        Parameters
        ----------
        config_path : str
            Path to ML configuration file
        data_path : str, optional
            Path to processed data file
        output_dir : str, optional
            Output directory for results
        verbose : bool
            Enable verbose logging
        """
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger("TrainingPipeline")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set paths
        self.data_path = data_path or self.config.get('paths', {}).get(
            'features_data', 'data/processed/appointments_features.csv'
        )
        
        # Create experiment directory
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) if output_dir else Path(
            self.config.get('output', {}).get('experiments_dir', 'outputs/experiments')
        ) / self.experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.df = None
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_transformed = None
        self.X_test_transformed = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        
        self.logger.info(f"Training Pipeline initialized")
        self.logger.info(f"  Experiment ID: {self.experiment_id}")
        self.logger.info(f"  Output directory: {self.output_dir}")

        # Setup MLflow
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment("noshow_prediction")
            self.logger.info(f"  MLflow tracking URI: {self.mlflow_tracking_uri}")
        except Exception as e:
            self.logger.warning(f"Failed to setup MLflow: {e}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def load_data(self) -> pd.DataFrame:
        """Load and validate data."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: Loading Data")
        self.logger.info("=" * 60)
        
        data_path = Path(self.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        self.df = pd.read_csv(data_path)
        self.logger.info(f"Loaded {len(self.df):,} rows, {len(self.df.columns)} columns")
        self.logger.info(f"Target distribution: {self.df['no_show'].value_counts().to_dict()}")
        
        return self.df
    
    def prepare_data(self) -> None:
        """Prepare data for training."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 2: Preparing Data")
        self.logger.info("=" * 60)
        
        if self.df is None:
            self.load_data()
        
        # Initialize preprocessor
        self.preprocessor = NoShowPreprocessor(self.config)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.preprocessor.prepare_data(self.df)
        
        # Transform data
        self.X_train_transformed = self.preprocessor.fit_transform(
            self.X_train, self.y_train
        )
        self.X_test_transformed = self.preprocessor.transform(self.X_test)
        
        self.logger.info(f"Training samples: {len(self.X_train):,}")
        self.logger.info(f"Test samples: {len(self.X_test):,}")
        self.logger.info(f"Features after transformation: {self.X_train_transformed.shape[1]}")
        
        # Save preprocessor
        preprocessor_path = self.output_dir / "models" / "preprocessor.joblib"
        preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        self.preprocessor.save(preprocessor_path)
    
    def train_baseline(
        self,
        model_names: Optional[list] = None
    ) -> Dict[str, Any]:
        """Train baseline models without tuning."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 3: Training Baseline Models")
        self.logger.info("=" * 60)
        
        if self.X_train_transformed is None:
            self.prepare_data()
        
        trainer = ModelTrainer(self.config)
        self.models = trainer.train_baseline_models(
            self.X_train_transformed,
            self.y_train
        )
        
        # Get comparison
        comparison = trainer.compare_models()
        self.logger.info("\nModel Comparison (CV):")
        self.logger.info(comparison.to_string(index=False))
        
        # Get best model
        self.best_model_name, self.best_model = trainer.get_best_model()
        
        # Save models
        models_dir = self.output_dir / "models" / "baseline"
        trainer.save_models(models_dir)
        
        return self.models
    
    def train_with_tuning(
        self,
        model_names: Optional[list] = None,
        n_iter: int = 50
    ) -> Dict[str, Any]:
        """Train models with hyperparameter tuning."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 3: Hyperparameter Tuning")
        self.logger.info("=" * 60)
        
        if self.X_train_transformed is None:
            self.prepare_data()
        
        # Initialize tuner
        tuner = HyperparameterTuner(self.config)
        
        # Run tuning
        tuning_results = tuner.tune_all_models(
            self.X_train_transformed,
            self.y_train,
            model_names=model_names
        )
        
        # Get comparison
        comparison = tuner.get_comparison_table()
        self.logger.info("\nTuning Results:")
        self.logger.info(comparison.to_string(index=False))
        
        # Get best model
        self.best_model_name, self.best_model = tuner.get_best_model()
        self.models = tuner.best_models
        
        # Save tuning results
        tuning_dir = self.output_dir / "tuning"
        tuner.save_results(tuning_dir)
        
        # Generate tuning plots
        figures_dir = self.output_dir / "figures" / "tuning"
        tuner.plot_comparison(save=True, output_dir=figures_dir)
        tuner.plot_cv_scores_distribution(save=True, output_dir=figures_dir)
        
        # Learning curve for best model
        tuner.plot_learning_curve(
            self.best_model_name,
            self.X_train_transformed,
            self.y_train,
            save=True,
            output_dir=figures_dir
        )
        
        return self.models
    
    def evaluate(self) -> Dict:
        """Evaluate models on test set."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 4: Evaluation")
        self.logger.info("=" * 60)
        
        if not self.models:
            raise RuntimeError("No models trained. Run train_baseline or train_with_tuning first.")
        
        evaluator = ModelEvaluator(self.config)
        
        # Set output directory for figures
        evaluator.figures_dir = self.output_dir / "figures" / "evaluation"
        evaluator.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluate all models
        self.results = evaluator.evaluate_all(
            self.models,
            self.X_test_transformed,
            self.y_test
        )
        
        # Get comparison table
        comparison = evaluator.get_comparison_table()
        self.logger.info("\nTest Set Results:")
        self.logger.info(comparison.to_string(index=False))
        
        # Generate plots
        evaluator.generate_all_plots()
        
        # Calculate business impact
        y_pred = self.best_model.predict(self.X_test_transformed)
        impact = evaluator.calculate_business_impact(
            self.y_test.values, y_pred, self.best_model_name
        )
        
        # Save results
        results_dir = self.output_dir / "evaluation"
        evaluator.save_results(results_dir)
        
        return self.results
    
    def interpret(self, max_samples: int = 1000) -> None:
        """Generate model interpretability analysis."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 5: Model Interpretability")
        self.logger.info("=" * 60)
        
        if self.best_model is None:
            raise RuntimeError("No model available. Train models first.")
        
        try:
            import shap
        except ImportError:
            self.logger.warning("SHAP not installed. Skipping interpretability analysis.")
            self.logger.warning("Install with: pip install shap")
            return
        
        # Initialize interpreter
        interpreter = ModelInterpreter(self.config)
        interpreter.output_dir = self.output_dir / "figures" / "interpretability"
        interpreter.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use sample of training data as background
        background_size = min(500, len(self.X_train_transformed))
        X_background = self.X_train_transformed[:background_size]
        
        # Fit interpreter
        interpreter.fit(
            self.best_model,
            X_background,
            feature_names=self.preprocessor.feature_names_,
            model_name=self.best_model_name
        )
        
        # Generate all plots
        sample_size = min(max_samples, len(self.X_test_transformed))
        X_sample = self.X_test_transformed[:sample_size]
        
        interpreter.generate_all_plots(
            X_sample,
            self.y_test[:sample_size],
            X_sample[:1]  # Sample for individual explanation
        )
        
        # Generate report
        report_path = self.output_dir / "reports" / "interpretation_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        create_interpretation_report(
            interpreter,
            X_sample,
            self.y_test[:sample_size],
            report_path
        )
        
        # Save importance results
        interpreter.save_results(self.output_dir / "interpretability")
        
        self.logger.info("Interpretability analysis complete!")
    
    def save_summary(self) -> None:
        """Save experiment summary."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 6: Saving Summary")
        self.logger.info("=" * 60)
        
        summary = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'config_used': {
                'target': self.config['ml_project']['target_column'],
                'primary_metric': self.config['evaluation']['primary_metric'],
                'test_size': self.config['splitting']['test_size']
            },
            'data': {
                'total_samples': len(self.df) if self.df is not None else 0,
                'training_samples': len(self.X_train) if self.X_train is not None else 0,
                'test_samples': len(self.X_test) if self.X_test is not None else 0,
                'features': len(self.preprocessor.feature_names_) if self.preprocessor else 0
            },
            'models_trained': list(self.models.keys()),
            'best_model': {
                'name': self.best_model_name,
                'test_results': self.results[self.best_model_name].to_dict() if self.best_model_name in self.results else {}
            },
            'output_directory': str(self.output_dir)
        }
        
        # Save summary
        summary_path = self.output_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Summary saved to {summary_path}")
        
        # Print summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPERIMENT SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Experiment ID: {self.experiment_id}")
        self.logger.info(f"Best Model: {self.best_model_name}")
        if self.best_model_name in self.results:
            result = self.results[self.best_model_name]
            self.logger.info(f"  ROC-AUC: {result.roc_auc:.4f}")
            self.logger.info(f"  F1 Score: {result.f1:.4f}")
            self.logger.info(f"  Precision: {result.precision:.4f}")
            self.logger.info(f"  Recall: {result.recall:.4f}")
        self.logger.info(f"Output: {self.output_dir}")
    
    def run(
        self,
        tune: bool = False,
        evaluate: bool = True,
        interpret: bool = False,
        model_names: Optional[list] = None
    ) -> Dict:
        """
        Run the complete training pipeline.
        
        Parameters
        ----------
        tune : bool
            Whether to perform hyperparameter tuning
        evaluate : bool
            Whether to evaluate on test set
        interpret : bool
            Whether to generate interpretability analysis
        model_names : list, optional
            Specific models to train
        
        Returns
        -------
        dict
            Pipeline results
        """
        start_time = datetime.now()
        
        self.logger.info("=" * 60)
        self.logger.info("STARTING TRAINING PIPELINE")
        self.logger.info("=" * 60)
        self.logger.info(f"Tune: {tune}")
        self.logger.info(f"Evaluate: {evaluate}")
        self.logger.info(f"Interpret: {interpret}")
        
        try:
            # Start MLflow run
            with mlflow.start_run(run_name=self.experiment_id) as run:
                self.logger.info(f"Started MLflow run: {run.info.run_id}")
                
                # Log configuration
                mlflow.log_params(self.config.get('ml_project', {}))
                mlflow.log_param("tune", tune)
                
                # Load and prepare data
                self.load_data()
                self.prepare_data()
                
                # Train models
                if tune:
                    self.train_with_tuning(model_names)
                else:
                    self.train_baseline(model_names)
                
                # Evaluate
                if evaluate:
                    self.evaluate()
                    
                    # Log metrics for best model
                    if self.best_model_name and self.best_model_name in self.results:
                        result = self.results[self.best_model_name]
                        mlflow.log_metrics({
                            "roc_auc": result.roc_auc,
                            "f1": result.f1,
                            "precision": result.precision,
                            "recall": result.recall,
                            "accuracy": result.accuracy
                        })
                        mlflow.log_param("best_model", self.best_model_name)
                
                # Log model artifact
                if self.best_model:
                    mlflow.sklearn.log_model(self.best_model, "model")
                    self.logger.info("Logged best model to MLflow")

                # Interpret
                if interpret:
                    self.interpret()
                
                # Save summary
                self.save_summary()
                
                elapsed = (datetime.now() - start_time).total_seconds()
                self.logger.info(f"\n✅ Pipeline completed in {elapsed:.1f} seconds")
                
                return {
                    'experiment_id': self.experiment_id,
                    'best_model': self.best_model_name,
                    'results': self.results,
                    'output_dir': str(self.output_dir),
                    'mlflow_run_id': run.info.run_id
                }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Train no-show prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_model.py                           # Basic training
  python train_model.py --tune                    # With hyperparameter tuning
  python train_model.py --tune --interpret        # Full pipeline
  python train_model.py --model random_forest     # Specific model
        """
    )
    
    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/appointments_features.csv",
        help="Path to processed data file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/ml_config.yaml",
        help="Path to ML configuration file"
    )
    
    # Training arguments
    parser.add_argument(
        "--model",
        type=str,
        nargs='+',
        default=None,
        help="Specific model(s) to train (e.g., random_forest xgboost)"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform hyperparameter tuning"
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=50,
        help="Number of iterations for random search tuning"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate models on test set"
    )
    parser.add_argument(
        "--interpret",
        action="store_true",
        help="Generate model interpretability analysis"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = TrainingPipeline(
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    results = pipeline.run(
        tune=args.tune,
        evaluate=args.evaluate,
        interpret=args.interpret,
        model_names=args.model
    )
    
    print(f"\n🎉 Training complete! Results saved to: {results['output_dir']}")
    
    return results


if __name__ == "__main__":
    main()
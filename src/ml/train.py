"""
Model Training Module
=====================
Handles training of classification models for no-show prediction.

This module provides:
- Baseline model training (Logistic Regression, Random Forest, etc.)
- Cross-validation with multiple metrics
- Experiment logging and tracking
- Model persistence
"""

import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import warnings

# sklearn models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# sklearn utilities
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.base import clone

import joblib

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not installed. Install with: pip install xgboost")


@dataclass
class ExperimentLog:
    """Data class for experiment logging."""
    experiment_id: str
    timestamp: str
    model_name: str
    model_params: Dict
    cv_scores: Dict
    test_scores: Optional[Dict] = None
    training_time: float = 0.0
    notes: str = ""


class ModelTrainer:
    """
    Train and compare classification models for no-show prediction.
    
    This class handles:
    - Training multiple baseline models
    - Cross-validation with configurable metrics
    - Experiment logging and tracking
    - Model persistence
    
    Attributes
    ----------
    config : dict
        ML configuration dictionary
    models : dict
        Dictionary of trained model objects
    cv_results : dict
        Cross-validation results for each model
    experiment_logs : list
        List of ExperimentLog objects
    
    Example
    -------
    >>> trainer = ModelTrainer(ml_config)
    >>> trainer.train_baseline_models(X_train, y_train)
    >>> comparison = trainer.compare_models()
    >>> trainer.save_best_model('models/baseline/')
    """
    
    def __init__(self, config: dict, preprocessor=None):
        """
        Initialize the model trainer.
        
        Parameters
        ----------
        config : dict
            ML configuration dictionary
        preprocessor : NoShowPreprocessor, optional
            Fitted preprocessor to include in pipeline
        """
        self.config = config
        self.preprocessor = preprocessor
        self.logger = logging.getLogger("healthcare_pipeline.ml.ModelTrainer")
        
        # State
        self.models: Dict[str, Any] = {}
        self.cv_results: Dict[str, Dict] = {}
        self.experiment_logs: List[ExperimentLog] = []
        self.best_model_name: Optional[str] = None
        
        # Configuration
        self.random_state = config['splitting']['random_state']
        self.cv_config = config['cross_validation']
        self.primary_metric = config['evaluation']['primary_metric']
        
        # Generate experiment ID
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger.info(f"ModelTrainer initialized (experiment_id: {self.experiment_id})")
    
    def _get_baseline_models(self) -> Dict[str, Any]:
        """
        Get dictionary of baseline model instances.
        
        Returns configured models based on ml_config.yaml
        
        Returns
        -------
        dict
            Dictionary of model name -> model instance
        """
        models_config = self.config['baseline_models']
        models = {}
        
        # Logistic Regression
        if models_config.get('logistic_regression', {}).get('enabled', True):
            params = models_config['logistic_regression'].get('params', {})
            models['logistic_regression'] = LogisticRegression(**params)
            self.logger.info("Added Logistic Regression to baseline models")
        
        # Random Forest
        if models_config.get('random_forest', {}).get('enabled', True):
            params = models_config['random_forest'].get('params', {})
            models['random_forest'] = RandomForestClassifier(**params)
            self.logger.info("Added Random Forest to baseline models")
        
        # Gradient Boosting
        if models_config.get('gradient_boosting', {}).get('enabled', True):
            params = models_config['gradient_boosting'].get('params', {})
            models['gradient_boosting'] = GradientBoostingClassifier(**params)
            self.logger.info("Added Gradient Boosting to baseline models")
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE and models_config.get('xgboost', {}).get('enabled', True):
            params = models_config['xgboost'].get('params', {})
            models['xgboost'] = XGBClassifier(**params)
            self.logger.info("Added XGBoost to baseline models")
        
        # Decision Tree (simple baseline)
        models['decision_tree'] = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            random_state=self.random_state,
            class_weight='balanced'
        )
        self.logger.info("Added Decision Tree to baseline models")
        
        return models
    
    def _run_cross_validation(
        self, 
        model, 
        X: np.ndarray, 
        y: np.ndarray,
        cv: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Run cross-validation for a single model.
        
        Parameters
        ----------
        model : estimator
            sklearn-compatible model
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        cv : int
            Number of CV folds
        
        Returns
        -------
        dict
            Dictionary of metric_name -> array of scores
        """
        scoring = self.cv_config['scoring']
        
        # Map scoring names to sklearn scoring strings
        scoring_map = {
            'roc_auc': 'roc_auc',
            'f1': 'f1',
            'precision': 'precision',
            'recall': 'recall',
            'accuracy': 'accuracy'
        }
        
        sklearn_scoring = {k: scoring_map.get(k, k) for k in scoring}
        
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=sklearn_scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        return cv_results
    
    def train_baseline_models(
        self, 
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        cv_folds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train all baseline models with cross-validation.
        
        Parameters
        ----------
        X_train : array-like
            Training feature matrix
        y_train : array-like
            Training target vector
        cv_folds : int, optional
            Number of CV folds (uses config if not provided)
        
        Returns
        -------
        dict
            Dictionary of model_name -> trained model
        """
        self.logger.info("="*60)
        self.logger.info("Training baseline models...")
        
        if cv_folds is None:
            cv_folds = self.cv_config['n_folds']
        
        # Convert to numpy if needed
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        
        # Get baseline models
        baseline_models = self._get_baseline_models()
        
        self.logger.info(f"Training {len(baseline_models)} models with {cv_folds}-fold CV")
        self.logger.info(f"Training data shape: {X_train.shape}")
        self.logger.info(f"Metrics: {self.cv_config['scoring']}")
        
        # Train each model
        for name, model in baseline_models.items():
            self.logger.info(f"\n{'='*40}")
            self.logger.info(f"Training: {name}")
            self.logger.info(f"{'='*40}")
            
            start_time = datetime.now()
            
            try:
                # Run cross-validation
                cv_results = self._run_cross_validation(model, X_train, y_train, cv_folds)
                
                # Fit on full training data
                model.fit(X_train, y_train)
                
                # Calculate training time
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Store results
                self.models[name] = model
                self.cv_results[name] = cv_results
                
                # Log results
                self._log_cv_results(name, cv_results, training_time)
                
                # Create experiment log
                experiment_log = ExperimentLog(
                    experiment_id=self.experiment_id,
                    timestamp=datetime.now().isoformat(),
                    model_name=name,
                    model_params=model.get_params(),
                    cv_scores=self._summarize_cv_results(cv_results),
                    training_time=training_time
                )
                self.experiment_logs.append(experiment_log)
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {str(e)}")
                continue
        
        self.logger.info("\n" + "="*60)
        self.logger.info(f"Training complete! Trained {len(self.models)} models")
        self.logger.info("="*60)
        
        return self.models
    
    def _log_cv_results(
        self, 
        model_name: str, 
        cv_results: Dict, 
        training_time: float
    ) -> None:
        """Log cross-validation results for a model."""
        
        self.logger.info(f"Results for {model_name}:")
        self.logger.info(f"  Training time: {training_time:.2f}s")
        
        for metric in self.cv_config['scoring']:
            test_key = f'test_{metric}'
            train_key = f'train_{metric}'
            
            if test_key in cv_results:
                test_scores = cv_results[test_key]
                self.logger.info(
                    f"  {metric}: {test_scores.mean():.4f} (+/- {test_scores.std()*2:.4f})"
                )
            
            if train_key in cv_results:
                train_scores = cv_results[train_key]
                # Check for overfitting
                if test_key in cv_results:
                    gap = train_scores.mean() - test_scores.mean()
                    if gap > 0.05:
                        self.logger.warning(f"    ⚠️ Potential overfitting (gap: {gap:.4f})")
    
    def _summarize_cv_results(self, cv_results: Dict) -> Dict:
        """Convert CV results to summary statistics."""
        summary = {}
        
        for key, values in cv_results.items():
            if isinstance(values, np.ndarray) and key.startswith(('test_', 'train_')):
                summary[key] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max())
                }
        
        return summary
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all trained models.
        
        Returns
        -------
        pd.DataFrame
            Comparison table with metrics for each model
        """
        if not self.cv_results:
            raise ValueError("No models trained yet. Call train_baseline_models first.")
        
        self.logger.info("Comparing models...")
        
        rows = []
        
        for model_name, cv_results in self.cv_results.items():
            row = {'model': model_name}
            
            for metric in self.cv_config['scoring']:
                test_key = f'test_{metric}'
                if test_key in cv_results:
                    scores = cv_results[test_key]
                    row[f'{metric}_mean'] = scores.mean()
                    row[f'{metric}_std'] = scores.std()
            
            rows.append(row)
        
        comparison_df = pd.DataFrame(rows)
        
        # Sort by primary metric
        primary_col = f'{self.primary_metric}_mean'
        if primary_col in comparison_df.columns:
            comparison_df = comparison_df.sort_values(primary_col, ascending=False)
            self.best_model_name = comparison_df.iloc[0]['model']
            self.logger.info(f"Best model: {self.best_model_name} ({primary_col}: {comparison_df.iloc[0][primary_col]:.4f})")
        
        return comparison_df
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best performing model.
        
        Returns
        -------
        tuple
            (model_name, model_object)
        """
        if self.best_model_name is None:
            self.compare_models()
        
        return self.best_model_name, self.models[self.best_model_name]
    
    def get_feature_importance(
        self, 
        model_name: Optional[str] = None,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get feature importance for a model.
        
        Parameters
        ----------
        model_name : str, optional
            Model name (uses best model if not provided)
        feature_names : list, optional
            List of feature names
        
        Returns
        -------
        pd.DataFrame
            Feature importance DataFrame
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        # Get importance based on model type
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])  # For logistic regression
        else:
            self.logger.warning(f"Model {model_name} doesn't have feature importance")
            return pd.DataFrame()
        
        # Create DataFrame
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_models(
        self, 
        output_dir: Union[str, Path],
        save_all: bool = True
    ) -> Dict[str, Path]:
        """
        Save trained models to disk.
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory
        save_all : bool
            If True, save all models. If False, save only best model.
        
        Returns
        -------
        dict
            Dictionary of model_name -> saved file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        models_to_save = self.models if save_all else {self.best_model_name: self.models[self.best_model_name]}
        
        for name, model in models_to_save.items():
            filepath = output_dir / f"{name}.joblib"
            joblib.dump(model, filepath)
            saved_paths[name] = filepath
            self.logger.info(f"Saved {name} to {filepath}")
        
        # Save experiment logs
        logs_path = output_dir / "experiment_logs.json"
        logs_data = [asdict(log) for log in self.experiment_logs]
        with open(logs_path, 'w') as f:
            json.dump(logs_data, f, indent=2, default=str)
        self.logger.info(f"Saved experiment logs to {logs_path}")
        
        # Save comparison results
        comparison_df = self.compare_models()
        comparison_path = output_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        self.logger.info(f"Saved model comparison to {comparison_path}")
        
        return saved_paths
    
    def load_model(
        self, 
        filepath: Union[str, Path]
    ) -> Any:
        """
        Load a model from disk.
        
        Parameters
        ----------
        filepath : str or Path
            Path to saved model
        
        Returns
        -------
        model
            Loaded model object
        """
        model = joblib.load(filepath)
        self.logger.info(f"Loaded model from {filepath}")
        return model
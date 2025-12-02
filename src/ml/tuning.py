"""
Hyperparameter Tuning Module
============================
Advanced hyperparameter optimization for no-show prediction models.

This module provides:
- Grid Search and Random Search implementations
- Cross-validation with multiple metrics
- Early stopping for iterative models
- Tuning results tracking and visualization
- Best model selection and persistence
"""

import pandas as pd
import numpy as np
import logging
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict, field
from copy import deepcopy

# sklearn imports
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    cross_validate
)
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, make_scorer
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

import joblib

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class TuningResult:
    """Container for hyperparameter tuning results."""
    model_name: str
    best_params: Dict
    best_score: float
    cv_results: Dict
    training_time: float
    n_iterations: int
    scoring_metric: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert numpy types to Python types for JSON serialization
        result['cv_results'] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in result['cv_results'].items()
        }
        return result


@dataclass
class TuningExperiment:
    """Container for a complete tuning experiment."""
    experiment_id: str
    started_at: str
    completed_at: Optional[str] = None
    config: Dict = field(default_factory=dict)
    results: Dict[str, TuningResult] = field(default_factory=dict)
    best_model_name: Optional[str] = None
    best_overall_score: Optional[float] = None
    notes: str = ""


class HyperparameterTuner:
    """
    Hyperparameter tuning for classification models.
    
    This class provides comprehensive hyperparameter optimization including:
    - Grid Search and Randomized Search
    - Multi-metric evaluation
    - Early stopping support
    - Results tracking and visualization
    - Best model selection
    
    Attributes
    ----------
    config : dict
        ML configuration dictionary
    results : dict
        Dictionary of model_name -> TuningResult
    best_models : dict
        Dictionary of model_name -> best estimator
    
    Example
    -------
    >>> tuner = HyperparameterTuner(ml_config)
    >>> tuner.tune_model('random_forest', X_train, y_train)
    >>> tuner.tune_all_models(X_train, y_train)
    >>> best_name, best_model = tuner.get_best_model()
    >>> tuner.save_results('outputs/tuning/')
    """
    
    def __init__(self, config: dict, preprocessor=None):
        """
        Initialize the hyperparameter tuner.
        
        Parameters
        ----------
        config : dict
            ML configuration dictionary
        preprocessor : object, optional
            Fitted preprocessor for creating pipelines
        """
        self.config = config
        self.preprocessor = preprocessor
        self.logger = logging.getLogger("healthcare_pipeline.ml.HyperparameterTuner")
        
        # Tuning configuration
        self.tuning_config = config.get('tuning', {})
        self.strategy = self.tuning_config.get('strategy', 'random')
        self.n_iter = self.tuning_config.get('n_iter', 50)
        self.cv_folds = self.tuning_config.get('cv_folds', 5)
        self.scoring = self.tuning_config.get('scoring', 'roc_auc')
        self.n_jobs = self.tuning_config.get('n_jobs', -1)
        self.verbose = self.tuning_config.get('verbose', 1)
        self.random_state = self.tuning_config.get('random_state', 42)
        
        # Parameter grids
        self.param_grids = self.tuning_config.get('param_grids', {})
        
        # Results storage
        self.results: Dict[str, TuningResult] = {}
        self.best_models: Dict[str, BaseEstimator] = {}
        self.search_objects: Dict[str, Union[GridSearchCV, RandomizedSearchCV]] = {}
        
        # Experiment tracking
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment = TuningExperiment(
            experiment_id=self.experiment_id,
            started_at=datetime.now().isoformat(),
            config=self.tuning_config
        )
        
        self.logger.info(f"HyperparameterTuner initialized (experiment_id: {self.experiment_id})")
        self.logger.info(f"  Strategy: {self.strategy}")
        self.logger.info(f"  CV folds: {self.cv_folds}")
        self.logger.info(f"  Scoring: {self.scoring}")
        self.logger.info(f"  Max iterations: {self.n_iter}")
    
    def _get_base_model(self, model_name: str) -> BaseEstimator:
        """
        Get a base model instance.
        
        Parameters
        ----------
        model_name : str
            Name of the model
        
        Returns
        -------
        BaseEstimator
            Model instance with default parameters
        """
        models = {
            'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state),
            'decision_tree': DecisionTreeClassifier(random_state=self.random_state),
        }
        
        if XGBOOST_AVAILABLE:
            models['xgboost'] = XGBClassifier(
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
        
        return models[model_name]
    
    def _get_param_grid(self, model_name: str) -> Dict:
        """
        Get parameter grid for a model.
        
        Parameters
        ----------
        model_name : str
            Name of the model
        
        Returns
        -------
        dict
            Parameter grid
        """
        if model_name in self.param_grids:
            return self.param_grids[model_name]
        
        # Default parameter grids
        default_grids = {
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs'],
                'class_weight': ['balanced', None]
            },
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'class_weight': ['balanced']
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5]
            },
            'decision_tree': {
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced']
            },
            'xgboost': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3],
                'scale_pos_weight': [1, 3]
            }
        }
        
        return default_grids.get(model_name, {})
    
    def _calculate_n_combinations(self, param_grid: Dict) -> int:
        """Calculate total number of parameter combinations."""
        n_combinations = 1
        for values in param_grid.values():
            if isinstance(values, list):
                n_combinations *= len(values)
        return n_combinations
    
    def _get_cv_splitter(self) -> StratifiedKFold:
        """Get cross-validation splitter."""
        return StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
    
    def tune_model(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Optional[Dict] = None,
        strategy: Optional[str] = None,
        n_iter: Optional[int] = None
    ) -> TuningResult:
        """
        Tune hyperparameters for a single model.
        
        Parameters
        ----------
        model_name : str
            Name of the model to tune
        X : np.ndarray
            Training feature matrix
        y : np.ndarray
            Training target vector
        param_grid : dict, optional
            Custom parameter grid (uses config if not provided)
        strategy : str, optional
            Search strategy ('grid' or 'random')
        n_iter : int, optional
            Number of iterations for random search
        
        Returns
        -------
        TuningResult
            Tuning results container
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Tuning: {model_name}")
        self.logger.info("=" * 60)
        
        # Get parameters
        if param_grid is None:
            param_grid = self._get_param_grid(model_name)
        
        if strategy is None:
            strategy = self.strategy
        
        if n_iter is None:
            n_iter = self.n_iter
        
        # Get base model
        base_model = self._get_base_model(model_name)
        
        # Calculate search space size
        n_combinations = self._calculate_n_combinations(param_grid)
        self.logger.info(f"  Parameter grid has {n_combinations} combinations")
        self.logger.info(f"  Strategy: {strategy}")
        
        # Get CV splitter
        cv = self._get_cv_splitter()
        
        # Create search object
        start_time = datetime.now()
        
        if strategy == 'grid' or n_combinations <= n_iter:
            self.logger.info(f"  Using GridSearchCV (testing all {n_combinations} combinations)")
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring=self.scoring,
                cv=cv,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                return_train_score=True,
                refit=True
            )
            actual_n_iter = n_combinations
        else:
            actual_n_iter = min(n_iter, n_combinations)
            self.logger.info(f"  Using RandomizedSearchCV ({actual_n_iter} iterations)")
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=actual_n_iter,
                scoring=self.scoring,
                cv=cv,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=self.random_state,
                return_train_score=True,
                refit=True
            )
        
        # Perform search
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X, y)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Store results
        result = TuningResult(
            model_name=model_name,
            best_params=search.best_params_,
            best_score=search.best_score_,
            cv_results=dict(search.cv_results_),
            training_time=training_time,
            n_iterations=actual_n_iter,
            scoring_metric=self.scoring
        )
        
        self.results[model_name] = result
        self.best_models[model_name] = search.best_estimator_
        self.search_objects[model_name] = search
        
        # Log results
        self._log_tuning_results(result)
        
        return result
    
    def _log_tuning_results(self, result: TuningResult) -> None:
        """Log tuning results."""
        self.logger.info(f"\n  Results for {result.model_name}:")
        self.logger.info(f"    Best {result.scoring_metric}: {result.best_score:.4f}")
        self.logger.info(f"    Training time: {result.training_time:.2f}s")
        self.logger.info(f"    Iterations: {result.n_iterations}")
        self.logger.info(f"    Best parameters:")
        for param, value in result.best_params.items():
            self.logger.info(f"      {param}: {value}")
    
    def tune_all_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_names: Optional[List[str]] = None
    ) -> Dict[str, TuningResult]:
        """
        Tune hyperparameters for all specified models.
        
        Parameters
        ----------
        X : np.ndarray
            Training feature matrix
        y : np.ndarray
            Training target vector
        model_names : list, optional
            List of model names to tune (default: all available)
        
        Returns
        -------
        dict
            Dictionary of model_name -> TuningResult
        """
        if model_names is None:
            model_names = list(self.param_grids.keys())
            if not model_names:
                model_names = ['logistic_regression', 'random_forest', 'gradient_boosting']
                if XGBOOST_AVAILABLE:
                    model_names.append('xgboost')
        
        self.logger.info("=" * 60)
        self.logger.info(f"HYPERPARAMETER TUNING - {len(model_names)} models")
        self.logger.info("=" * 60)
        self.logger.info(f"Models to tune: {model_names}")
        self.logger.info(f"Training data shape: {X.shape}")
        
        for model_name in model_names:
            try:
                self.tune_model(model_name, X, y)
            except Exception as e:
                self.logger.error(f"Error tuning {model_name}: {str(e)}")
                continue
        
        # Update experiment
        self.experiment.completed_at = datetime.now().isoformat()
        self.experiment.results = {name: r.to_dict() for name, r in self.results.items()}
        
        # Find best overall model
        if self.results:
            best_name = max(self.results.keys(), key=lambda k: self.results[k].best_score)
            self.experiment.best_model_name = best_name
            self.experiment.best_overall_score = self.results[best_name].best_score
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("TUNING COMPLETE!")
        self.logger.info("=" * 60)
        
        return self.results
    
    def get_best_model(self) -> Tuple[str, BaseEstimator]:
        """
        Get the best model based on tuning results.
        
        Returns
        -------
        tuple
            (model_name, best_estimator)
        """
        if not self.results:
            raise ValueError("No models have been tuned yet.")
        
        best_name = max(self.results.keys(), key=lambda k: self.results[k].best_score)
        best_model = self.best_models[best_name]
        
        self.logger.info(f"Best model: {best_name} ({self.scoring}: {self.results[best_name].best_score:.4f})")
        
        return best_name, best_model
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Get a comparison table of all tuning results.
        
        Returns
        -------
        pd.DataFrame
            Comparison table
        """
        if not self.results:
            return pd.DataFrame()
        
        rows = []
        for name, result in self.results.items():
            cv_results = result.cv_results
            
            # Get mean and std of test scores
            mean_test_score = cv_results.get('mean_test_score', [result.best_score])[0]
            std_test_score = cv_results.get('std_test_score', [0])[0]
            
            rows.append({
                'Model': name,
                f'Best {self.scoring}': result.best_score,
                'Mean CV Score': mean_test_score,
                'Std CV Score': std_test_score,
                'Training Time (s)': result.training_time,
                'Iterations': result.n_iterations
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values(f'Best {self.scoring}', ascending=False).reset_index(drop=True)
        
        return df
    
    def get_best_params_table(self) -> pd.DataFrame:
        """
        Get a table of best parameters for all models.
        
        Returns
        -------
        pd.DataFrame
            Best parameters table
        """
        if not self.results:
            return pd.DataFrame()
        
        rows = []
        for name, result in self.results.items():
            row = {'Model': name}
            row.update(result.best_params)
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def plot_tuning_results(
        self,
        model_name: str,
        param_name: str,
        save: bool = True,
        output_dir: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot tuning results for a specific parameter.
        
        Parameters
        ----------
        model_name : str
            Name of the model
        param_name : str
            Name of the parameter to plot
        save : bool
            Whether to save the figure
        output_dir : Path, optional
            Output directory
        
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not found in results")
        
        result = self.results[model_name]
        cv_results = result.cv_results
        
        # Get parameter values and scores
        param_key = f'param_{param_name}'
        if param_key not in cv_results:
            self.logger.warning(f"Parameter '{param_name}' not found in results")
            return None
        
        param_values = cv_results[param_key]
        mean_scores = cv_results['mean_test_score']
        std_scores = cv_results['std_test_score']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert param values to strings for categorical plotting
        param_str = [str(v) for v in param_values]
        unique_params = sorted(set(param_str), key=lambda x: param_str.index(x))
        
        # Group by parameter value
        grouped_data = {}
        for p, score, std in zip(param_str, mean_scores, std_scores):
            if p not in grouped_data:
                grouped_data[p] = {'scores': [], 'stds': []}
            grouped_data[p]['scores'].append(score)
            grouped_data[p]['stds'].append(std)
        
        # Calculate mean for each parameter value
        x_positions = range(len(unique_params))
        means = [np.mean(grouped_data[p]['scores']) for p in unique_params]
        stds = [np.mean(grouped_data[p]['stds']) for p in unique_params]
        
        ax.bar(x_positions, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(unique_params, rotation=45, ha='right')
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel(f'Mean {self.scoring}', fontsize=12)
        ax.set_title(f'{model_name}: {param_name} vs {self.scoring}', fontsize=14)
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / f"tuning_{model_name}_{param_name}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved tuning plot to {filepath}")
        
        return fig
    
    def plot_learning_curve(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        train_sizes: np.ndarray = None,
        save: bool = True,
        output_dir: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot learning curve for a tuned model.
        
        Parameters
        ----------
        model_name : str
            Name of the model
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        train_sizes : np.ndarray, optional
            Training set sizes to evaluate
        save : bool
            Whether to save the figure
        output_dir : Path, optional
            Output directory
        
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        from sklearn.model_selection import learning_curve
        
        if model_name not in self.best_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.best_models[model_name]
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        self.logger.info(f"Computing learning curve for {model_name}...")
        
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=self._get_cv_splitter(),
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        
        # Calculate means and stds
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        test_mean = test_scores.mean(axis=1)
        test_std = test_scores.std(axis=1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                       alpha=0.1, color='blue')
        ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std,
                       alpha=0.1, color='orange')
        ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training score')
        ax.plot(train_sizes_abs, test_mean, 'o-', color='orange', label='Cross-validation score')
        
        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel(f'{self.scoring}', fontsize=12)
        ax.set_title(f'Learning Curve: {model_name}', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Check for overfitting/underfitting
        final_gap = train_mean[-1] - test_mean[-1]
        if final_gap > 0.1:
            ax.annotate(
                f'⚠️ Potential overfitting\n(gap: {final_gap:.3f})',
                xy=(train_sizes_abs[-1], test_mean[-1]),
                xytext=(train_sizes_abs[-1] * 0.7, test_mean[-1] * 0.9),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red'),
                color='red'
            )
        
        plt.tight_layout()
        
        if save and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / f"learning_curve_{model_name}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved learning curve to {filepath}")
        
        return fig
    
    def plot_comparison(
        self,
        save: bool = True,
        output_dir: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot comparison of all tuned models.
        
        Parameters
        ----------
        save : bool
            Whether to save the figure
        output_dir : Path, optional
            Output directory
        
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        if not self.results:
            raise ValueError("No tuning results available")
        
        comparison = self.get_comparison_table()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Best scores comparison
        ax1 = axes[0]
        colors = plt.cm.Set2(np.linspace(0, 1, len(comparison)))
        bars = ax1.barh(comparison['Model'], comparison[f'Best {self.scoring}'], color=colors)
        ax1.set_xlabel(f'{self.scoring}', fontsize=12)
        ax1.set_title(f'Best {self.scoring} by Model', fontsize=14)
        ax1.grid(True, axis='x', alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, comparison[f'Best {self.scoring}']):
            ax1.text(score + 0.002, bar.get_y() + bar.get_height()/2, 
                    f'{score:.4f}', va='center', fontsize=10)
        
        # Plot 2: Training time comparison
        ax2 = axes[1]
        bars = ax2.barh(comparison['Model'], comparison['Training Time (s)'], color=colors)
        ax2.set_xlabel('Training Time (seconds)', fontsize=12)
        ax2.set_title('Training Time by Model', fontsize=14)
        ax2.grid(True, axis='x', alpha=0.3)
        
        # Add value labels
        for bar, time in zip(bars, comparison['Training Time (s)']):
            ax2.text(time + 1, bar.get_y() + bar.get_height()/2,
                    f'{time:.1f}s', va='center', fontsize=10)
        
        plt.suptitle('Hyperparameter Tuning Results Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / "tuning_comparison.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved comparison plot to {filepath}")
        
        return fig
    
    def plot_cv_scores_distribution(
        self,
        save: bool = True,
        output_dir: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot distribution of CV scores for each model.
        
        Parameters
        ----------
        save : bool
            Whether to save the figure
        output_dir : Path, optional
            Output directory
        
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        if not self.search_objects:
            raise ValueError("No tuning results available")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        data = []
        labels = []
        
        for name, search in self.search_objects.items():
            cv_results = search.cv_results_
            # Get all fold scores for the best model
            best_idx = search.best_index_
            
            # Collect scores from all folds
            fold_scores = []
            for i in range(self.cv_folds):
                fold_key = f'split{i}_test_score'
                if fold_key in cv_results:
                    fold_scores.append(cv_results[fold_key][best_idx])
            
            if fold_scores:
                data.append(fold_scores)
                labels.append(name)
        
        # Create box plot
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.Set2(np.linspace(0, 1, len(data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(f'{self.scoring}', fontsize=12)
        ax.set_title('CV Score Distribution (Best Parameters)', fontsize=14)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Rotate x labels if needed
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / "cv_scores_distribution.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved CV scores distribution to {filepath}")
        
        return fig
    
    def save_results(
        self,
        output_dir: Union[str, Path],
        save_models: bool = True
    ) -> Dict[str, Path]:
        """
        Save all tuning results and models.
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory
        save_models : bool
            Whether to save the best models
        
        Returns
        -------
        dict
            Dictionary of artifact_name -> file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save comparison table
        comparison = self.get_comparison_table()
        comparison_path = output_dir / "tuning_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        saved_files['comparison'] = comparison_path
        
        # Save best parameters
        params_table = self.get_best_params_table()
        params_path = output_dir / "best_parameters.csv"
        params_table.to_csv(params_path, index=False)
        saved_files['parameters'] = params_path
        
        # Save detailed results as JSON
        results_dict = {name: result.to_dict() for name, result in self.results.items()}
        results_path = output_dir / "tuning_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        saved_files['results'] = results_path
        
        # Save experiment summary
        experiment_dict = {
            'experiment_id': self.experiment.experiment_id,
            'started_at': self.experiment.started_at,
            'completed_at': self.experiment.completed_at,
            'config': self.experiment.config,
            'best_model_name': self.experiment.best_model_name,
            'best_overall_score': self.experiment.best_overall_score
        }
        experiment_path = output_dir / "experiment_summary.json"
        with open(experiment_path, 'w') as f:
            json.dump(experiment_dict, f, indent=2, default=str)
        saved_files['experiment'] = experiment_path
        
        # Save best models
        if save_models:
            models_dir = output_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            for name, model in self.best_models.items():
                model_path = models_dir / f"{name}_tuned.joblib"
                joblib.dump(model, model_path)
                saved_files[f'model_{name}'] = model_path
        
        # Generate and save plots
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        try:
            self.plot_comparison(save=True, output_dir=figures_dir)
            self.plot_cv_scores_distribution(save=True, output_dir=figures_dir)
        except Exception as e:
            self.logger.warning(f"Failed to generate some plots: {e}")
        
        self.logger.info(f"Saved tuning results to {output_dir}")
        self.logger.info(f"  Files saved: {len(saved_files)}")
        
        return saved_files
    
    def load_results(
        self,
        input_dir: Union[str, Path]
    ) -> None:
        """
        Load tuning results from disk.
        
        Parameters
        ----------
        input_dir : str or Path
            Input directory containing saved results
        """
        input_dir = Path(input_dir)
        
        # Load results JSON
        results_path = input_dir / "tuning_results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                results_dict = json.load(f)
            
            for name, result_data in results_dict.items():
                self.results[name] = TuningResult(**result_data)
        
        # Load models
        models_dir = input_dir / "models"
        if models_dir.exists():
            for model_file in models_dir.glob("*_tuned.joblib"):
                name = model_file.stem.replace('_tuned', '')
                self.best_models[name] = joblib.load(model_file)
        
        self.logger.info(f"Loaded tuning results from {input_dir}")
        self.logger.info(f"  Models loaded: {list(self.best_models.keys())}")
"""
Model Interpretability Module
=============================
Comprehensive model interpretability for no-show prediction.

This module provides:
- SHAP (SHapley Additive exPlanations) analysis
- Permutation importance
- Partial Dependence Plots (PDP)
- Individual prediction explanations
- Feature importance comparisons

Key Concepts:
-------------
- SHAP values explain how each feature contributes to a prediction
- Permutation importance measures feature importance by shuffling
- PDPs show the marginal effect of features on predictions
"""

import pandas as pd
import numpy as np
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import json

# sklearn imports
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.inspection import partial_dependence

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Try importing SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. Install with: pip install shap")

import joblib


@dataclass
class FeatureImportance:
    """Container for feature importance results."""
    feature_names: List[str]
    importance_values: np.ndarray
    importance_type: str  # 'shap', 'permutation', 'model'
    std_values: Optional[np.ndarray] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.importance_values
        })
        if self.std_values is not None:
            df['std'] = self.std_values
        return df.sort_values('importance', ascending=False).reset_index(drop=True)


@dataclass
class PredictionExplanation:
    """Container for individual prediction explanation."""
    prediction: float
    probability: float
    base_value: float
    feature_names: List[str]
    feature_values: np.ndarray
    shap_values: np.ndarray
    top_positive_features: List[Tuple[str, float, float]]  # (name, value, shap)
    top_negative_features: List[Tuple[str, float, float]]  # (name, value, shap)
    risk_tier: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'prediction': int(self.prediction),
            'probability': float(self.probability),
            'base_value': float(self.base_value),
            'risk_tier': self.risk_tier,
            'top_positive_features': [
                {'feature': f, 'value': float(v), 'impact': float(s)}
                for f, v, s in self.top_positive_features
            ],
            'top_negative_features': [
                {'feature': f, 'value': float(v), 'impact': float(s)}
                for f, v, s in self.top_negative_features
            ]
        }


class ModelInterpreter:
    """
    Model interpretability analysis for classification models.
    
    This class provides comprehensive model interpretation including:
    - SHAP value analysis
    - Permutation importance
    - Partial dependence plots
    - Individual prediction explanations
    
    Attributes
    ----------
    config : dict
        ML configuration dictionary
    model : BaseEstimator
        Fitted sklearn-compatible model
    feature_names : list
        List of feature names
    shap_values : np.ndarray
        Computed SHAP values
    
    Example
    -------
    >>> interpreter = ModelInterpreter(config)
    >>> interpreter.fit(model, X_train, feature_names)
    >>> interpreter.plot_shap_summary(X_test)
    >>> explanation = interpreter.explain_prediction(X_test[0])
    """
    
    def __init__(self, config: dict):
        """
        Initialize the model interpreter.
        
        Parameters
        ----------
        config : dict
            ML configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("healthcare_pipeline.ml.ModelInterpreter")
        
        # Interpretability configuration
        self.interp_config = config.get('interpretability', {})
        self.shap_config = self.interp_config.get('shap', {})
        self.perm_config = self.interp_config.get('permutation', {})
        self.pdp_config = self.interp_config.get('pdp', {})
        
        # State
        self.model = None
        self.model_name = None
        self.feature_names = None
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        self.is_fitted = False
        
        # Results storage
        self.importance_results: Dict[str, FeatureImportance] = {}
        
        # Output directory
        self.output_dir = Path(config.get('output', {}).get('figures_dir', 'outputs/figures/ml'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("ModelInterpreter initialized")
        if not SHAP_AVAILABLE:
            self.logger.warning("SHAP not available. Some features will be disabled.")
    
    def fit(
        self,
        model: BaseEstimator,
        X_background: np.ndarray,
        feature_names: Optional[List[str]] = None,
        model_name: str = "model"
    ) -> 'ModelInterpreter':
        """
        Fit the interpreter with a model and background data.
        
        Parameters
        ----------
        model : BaseEstimator
            Fitted sklearn-compatible model
        X_background : np.ndarray
            Background data for SHAP (training data sample)
        feature_names : list, optional
            List of feature names
        model_name : str
            Name of the model for labeling
        
        Returns
        -------
        self
            For method chaining
        """
        self.model = model
        self.model_name = model_name
        
        # Set feature names
        if feature_names is not None:
            self.feature_names = list(feature_names)
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_background.shape[1])]
        
        self.logger.info(f"Fitting interpreter for {model_name}")
        self.logger.info(f"  Features: {len(self.feature_names)}")
        self.logger.info(f"  Background samples: {len(X_background)}")
        
        # Create SHAP explainer
        if SHAP_AVAILABLE and self.shap_config.get('enabled', True):
            self._create_shap_explainer(X_background)
        
        self.is_fitted = True
        return self
    
    def _create_shap_explainer(self, X_background: np.ndarray) -> None:
        """
        Create appropriate SHAP explainer based on model type.
        
        Parameters
        ----------
        X_background : np.ndarray
            Background data for SHAP
        """
        model_type = type(self.model).__name__
        self.logger.info(f"Creating SHAP explainer for {model_type}")
        
        # Subsample background data if needed
        max_samples = self.shap_config.get('background_samples', 100)
        if len(X_background) > max_samples:
            indices = np.random.choice(len(X_background), max_samples, replace=False)
            X_background = X_background[indices]
            self.logger.info(f"  Subsampled to {max_samples} background samples")
        
        try:
            # Try TreeExplainer first (fast for tree-based models)
            if model_type in ['RandomForestClassifier', 'GradientBoostingClassifier', 
                             'XGBClassifier', 'LGBMClassifier', 'DecisionTreeClassifier']:
                self.explainer = shap.TreeExplainer(self.model)
                self.logger.info("  Using TreeExplainer")
            
            # LinearExplainer for linear models
            elif model_type in ['LogisticRegression', 'LinearSVC', 'SGDClassifier']:
                self.explainer = shap.LinearExplainer(self.model, X_background)
                self.logger.info("  Using LinearExplainer")
            
            # Fall back to KernelExplainer (model-agnostic but slow)
            else:
                self.logger.info("  Using KernelExplainer (may be slow)")
                
                # Create prediction function
                if hasattr(self.model, 'predict_proba'):
                    predict_fn = lambda x: self.model.predict_proba(x)[:, 1]
                else:
                    predict_fn = self.model.predict
                
                self.explainer = shap.KernelExplainer(predict_fn, X_background)
            
            # Get expected value
            if hasattr(self.explainer, 'expected_value'):
                ev = self.explainer.expected_value
                if isinstance(ev, np.ndarray):
                    self.expected_value = ev[1] if len(ev) > 1 else ev[0]
                else:
                    self.expected_value = ev
                    
        except Exception as e:
            self.logger.error(f"Failed to create SHAP explainer: {e}")
            self.explainer = None
    
    def compute_shap_values(
        self,
        X: np.ndarray,
        max_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute SHAP values for given data.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix to explain
        max_samples : int, optional
            Maximum samples to compute (for speed)
        
        Returns
        -------
        np.ndarray
            SHAP values matrix
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            raise RuntimeError("SHAP is not available or explainer not fitted")
        
        # Subsample if needed
        if max_samples is None:
            max_samples = self.shap_config.get('max_samples', 1000)
        
        if len(X) > max_samples:
            self.logger.info(f"Subsampling {len(X)} to {max_samples} samples for SHAP")
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]
        
        self.logger.info(f"Computing SHAP values for {len(X)} samples...")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = self.explainer.shap_values(X)
        
        # Handle multi-class output (we want class 1 = no-show)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Class 1 (no-show)
        
        self.shap_values = shap_values
        self.logger.info(f"SHAP values shape: {shap_values.shape}")
        
        return shap_values
    
    def compute_permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10,
        scoring: str = 'roc_auc'
    ) -> FeatureImportance:
        """
        Compute permutation importance.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        n_repeats : int
            Number of permutation repeats
        scoring : str
            Scoring metric
        
        Returns
        -------
        FeatureImportance
            Permutation importance results
        """
        self.logger.info(f"Computing permutation importance ({n_repeats} repeats)...")
        
        if n_repeats is None:
            n_repeats = self.perm_config.get('n_repeats', 10)
        if scoring is None:
            scoring = self.perm_config.get('scoring', 'roc_auc')
        
        result = permutation_importance(
            self.model, X, y,
            n_repeats=n_repeats,
            random_state=self.config.get('splitting', {}).get('random_state', 42),
            scoring=scoring,
            n_jobs=-1
        )
        
        importance = FeatureImportance(
            feature_names=self.feature_names,
            importance_values=result.importances_mean,
            importance_type='permutation',
            std_values=result.importances_std
        )
        
        self.importance_results['permutation'] = importance
        self.logger.info("Permutation importance computed")
        
        return importance
    
    def get_model_feature_importance(self) -> Optional[FeatureImportance]:
        """
        Get built-in feature importance from the model.
        
        Returns
        -------
        FeatureImportance or None
            Model-based feature importance
        """
        importance_values = None
        
        # Tree-based models have feature_importances_
        if hasattr(self.model, 'feature_importances_'):
            importance_values = self.model.feature_importances_
            self.logger.info("Using model's feature_importances_")
        
        # Linear models have coef_
        elif hasattr(self.model, 'coef_'):
            importance_values = np.abs(self.model.coef_[0])
            self.logger.info("Using absolute coefficient values")
        
        if importance_values is not None:
            importance = FeatureImportance(
                feature_names=self.feature_names,
                importance_values=importance_values,
                importance_type='model'
            )
            self.importance_results['model'] = importance
            return importance
        
        return None
    
    def get_shap_importance(self) -> Optional[FeatureImportance]:
        """
        Get feature importance from SHAP values.
        
        Returns
        -------
        FeatureImportance or None
            SHAP-based feature importance
        """
        if self.shap_values is None:
            self.logger.warning("SHAP values not computed. Call compute_shap_values first.")
            return None
        
        # Mean absolute SHAP value
        importance_values = np.abs(self.shap_values).mean(axis=0)
        
        importance = FeatureImportance(
            feature_names=self.feature_names,
            importance_values=importance_values,
            importance_type='shap'
        )
        
        self.importance_results['shap'] = importance
        return importance
    
    def explain_prediction(
        self,
        x: np.ndarray,
        top_n: int = 5
    ) -> PredictionExplanation:
        """
        Explain a single prediction.
        
        Parameters
        ----------
        x : np.ndarray
            Single sample to explain (1D array)
        top_n : int
            Number of top features to include
        
        Returns
        -------
        PredictionExplanation
            Explanation for the prediction
        """
        if not self.is_fitted:
            raise RuntimeError("Interpreter not fitted. Call fit() first.")
        
        # Ensure 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Get prediction
        prediction = self.model.predict(x)[0]
        probability = self.model.predict_proba(x)[0, 1]
        
        # Get SHAP values
        if SHAP_AVAILABLE and self.explainer is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shap_vals = self.explainer.shap_values(x)
            
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # Class 1
            
            shap_vals = shap_vals.flatten()
            
            # Get base value
            base_value = self.expected_value if self.expected_value is not None else 0.5
            
            # Sort features by SHAP value
            feature_shap = list(zip(self.feature_names, x.flatten(), shap_vals))
            
            # Top positive (increase no-show probability)
            top_positive = sorted(feature_shap, key=lambda x: x[2], reverse=True)[:top_n]
            top_positive = [(f, float(v), float(s)) for f, v, s in top_positive if s > 0]
            
            # Top negative (decrease no-show probability)
            top_negative = sorted(feature_shap, key=lambda x: x[2])[:top_n]
            top_negative = [(f, float(v), float(s)) for f, v, s in top_negative if s < 0]
            
        else:
            shap_vals = np.zeros(len(self.feature_names))
            base_value = 0.5
            top_positive = []
            top_negative = []
        
        # Determine risk tier
        if probability >= 0.7:
            risk_tier = "CRITICAL"
        elif probability >= 0.5:
            risk_tier = "HIGH"
        elif probability >= 0.3:
            risk_tier = "MEDIUM"
        elif probability >= 0.15:
            risk_tier = "LOW"
        else:
            risk_tier = "MINIMAL"
        
        return PredictionExplanation(
            prediction=prediction,
            probability=probability,
            base_value=base_value,
            feature_names=self.feature_names,
            feature_values=x.flatten(),
            shap_values=shap_vals,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            risk_tier=risk_tier
        )
    
    # ==================== VISUALIZATION METHODS ====================
    
    def plot_shap_summary(
        self,
        X: np.ndarray,
        plot_type: str = 'dot',
        max_display: int = 20,
        save: bool = True,
        filename: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Create SHAP summary plot.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        plot_type : str
            'dot' for beeswarm, 'bar' for bar chart
        max_display : int
            Maximum features to display
        save : bool
            Whether to save the plot
        filename : str, optional
            Output filename
        
        Returns
        -------
        plt.Figure or None
            Matplotlib figure
        """
        if not SHAP_AVAILABLE:
            self.logger.warning("SHAP not available")
            return None
        
        # Compute SHAP values if not already done
        if self.shap_values is None or len(self.shap_values) != len(X):
            self.compute_shap_values(X)
        
        # Use the subset that matches shap_values
        X_plot = X[:len(self.shap_values)]
        
        self.logger.info(f"Creating SHAP summary plot ({plot_type})...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if plot_type == 'bar':
            shap.summary_plot(
                self.shap_values,
                X_plot,
                feature_names=self.feature_names,
                plot_type='bar',
                max_display=max_display,
                show=False
            )
        else:
            shap.summary_plot(
                self.shap_values,
                X_plot,
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
        
        plt.title(f'SHAP Summary Plot - {self.model_name}', fontsize=14, pad=20)
        plt.tight_layout()
        
        if save:
            if filename is None:
                filename = f"shap_summary_{plot_type}_{self.model_name}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved SHAP summary plot to {filepath}")
        
        return plt.gcf()
    
    def plot_shap_bar(
        self,
        X: np.ndarray,
        max_display: int = 15,
        save: bool = True
    ) -> Optional[plt.Figure]:
        """
        Create SHAP bar plot (mean absolute SHAP values).
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        max_display : int
            Maximum features to display
        save : bool
            Whether to save the plot
        
        Returns
        -------
        plt.Figure or None
            Matplotlib figure
        """
        return self.plot_shap_summary(X, plot_type='bar', max_display=max_display, save=save)
    
    def plot_shap_waterfall(
        self,
        x: np.ndarray,
        sample_idx: int = 0,
        max_display: int = 15,
        save: bool = True
    ) -> Optional[plt.Figure]:
        """
        Create SHAP waterfall plot for a single prediction.
        
        Parameters
        ----------
        x : np.ndarray
            Single sample or sample index
        sample_idx : int
            Index of sample to explain (if x is 2D)
        max_display : int
            Maximum features to display
        save : bool
            Whether to save the plot
        
        Returns
        -------
        plt.Figure or None
            Matplotlib figure
        """
        if not SHAP_AVAILABLE:
            self.logger.warning("SHAP not available")
            return None
        
        # Handle input
        if x.ndim == 2:
            x = x[sample_idx]
        
        x = x.reshape(1, -1)
        
        # Compute SHAP values for this sample
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_vals = self.explainer.shap_values(x)
        
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        
        # Get expected value
        if hasattr(self.explainer, 'expected_value'):
            ev = self.explainer.expected_value
            if isinstance(ev, np.ndarray):
                base_value = ev[1] if len(ev) > 1 else ev[0]
            else:
                base_value = ev
        else:
            base_value = 0.5
        
        self.logger.info("Creating SHAP waterfall plot...")
        
        # Create Explanation object for newer SHAP versions
        explanation = shap.Explanation(
            values=shap_vals[0],
            base_values=base_value,
            data=x[0],
            feature_names=self.feature_names
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.waterfall_plot(explanation, max_display=max_display, show=False)
        
        # Get prediction probability
        prob = self.model.predict_proba(x)[0, 1]
        plt.title(f'SHAP Waterfall - Prediction: {prob:.1%} no-show risk', fontsize=14, pad=20)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"shap_waterfall_{self.model_name}.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved SHAP waterfall plot to {filepath}")
        
        return plt.gcf()
    
    def plot_shap_dependence(
        self,
        X: np.ndarray,
        feature: Union[str, int],
        interaction_feature: Optional[Union[str, int]] = 'auto',
        save: bool = True
    ) -> Optional[plt.Figure]:
        """
        Create SHAP dependence plot for a feature.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        feature : str or int
            Feature name or index
        interaction_feature : str, int, or 'auto'
            Feature for color coding
        save : bool
            Whether to save the plot
        
        Returns
        -------
        plt.Figure or None
            Matplotlib figure
        """
        if not SHAP_AVAILABLE:
            self.logger.warning("SHAP not available")
            return None
        
        # Compute SHAP values if needed
        if self.shap_values is None or len(self.shap_values) != len(X):
            self.compute_shap_values(X)
        
        X_plot = X[:len(self.shap_values)]
        
        # Get feature index
        if isinstance(feature, str):
            if feature in self.feature_names:
                feature_idx = self.feature_names.index(feature)
            else:
                self.logger.warning(f"Feature '{feature}' not found")
                return None
            feature_name = feature
        else:
            feature_idx = feature
            feature_name = self.feature_names[feature_idx]
        
        self.logger.info(f"Creating SHAP dependence plot for '{feature_name}'...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        shap.dependence_plot(
            feature_idx,
            self.shap_values,
            X_plot,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            ax=ax,
            show=False
        )
        
        plt.title(f'SHAP Dependence: {feature_name}', fontsize=14)
        plt.tight_layout()
        
        if save:
            safe_name = feature_name.replace(' ', '_').replace('/', '_')
            filepath = self.output_dir / f"shap_dependence_{safe_name}_{self.model_name}.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved SHAP dependence plot to {filepath}")
        
        return plt.gcf()
    
    def plot_top_dependence_plots(
        self,
        X: np.ndarray,
        top_n: int = 6,
        save: bool = True
    ) -> List[plt.Figure]:
        """
        Create dependence plots for top N most important features.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        top_n : int
            Number of top features
        save : bool
            Whether to save plots
        
        Returns
        -------
        list
            List of matplotlib figures
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Get top features by mean |SHAP|
        importance = np.abs(self.shap_values).mean(axis=0)
        top_indices = np.argsort(importance)[::-1][:top_n]
        
        figures = []
        for idx in top_indices:
            feature_name = self.feature_names[idx]
            fig = self.plot_shap_dependence(X, idx, save=save)
            if fig:
                figures.append(fig)
                plt.close(fig)
        
        return figures
    
    def plot_feature_importance_comparison(
        self,
        X: np.ndarray,
        y: np.ndarray,
        top_n: int = 15,
        save: bool = True
    ) -> plt.Figure:
        """
        Compare feature importance from different methods.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        top_n : int
            Number of top features to show
        save : bool
            Whether to save the plot
        
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        self.logger.info("Computing feature importance comparison...")
        
        # Get model-based importance
        model_imp = self.get_model_feature_importance()
        
        # Get permutation importance
        perm_imp = self.compute_permutation_importance(X, y)
        
        # Get SHAP importance if available
        if SHAP_AVAILABLE and self.explainer is not None:
            self.compute_shap_values(X)
            shap_imp = self.get_shap_importance()
        else:
            shap_imp = None
        
        # Create comparison dataframe
        comparison_data = pd.DataFrame({'feature': self.feature_names})
        
        if model_imp:
            model_df = model_imp.to_dataframe()
            model_df = model_df.rename(columns={'importance': 'model'})
            comparison_data = comparison_data.merge(model_df[['feature', 'model']], on='feature', how='left')
        
        if perm_imp:
            perm_df = perm_imp.to_dataframe()
            perm_df = perm_df.rename(columns={'importance': 'permutation'})
            comparison_data = comparison_data.merge(perm_df[['feature', 'permutation']], on='feature', how='left')
        
        if shap_imp:
            shap_df = shap_imp.to_dataframe()
            shap_df = shap_df.rename(columns={'importance': 'shap'})
            comparison_data = comparison_data.merge(shap_df[['feature', 'shap']], on='feature', how='left')
        
        # Normalize each method to 0-1 scale
        for col in ['model', 'permutation', 'shap']:
            if col in comparison_data.columns:
                max_val = comparison_data[col].max()
                if max_val > 0:
                    comparison_data[col] = comparison_data[col] / max_val
        
        # Calculate average importance for sorting
        imp_cols = [c for c in ['model', 'permutation', 'shap'] if c in comparison_data.columns]
        comparison_data['avg_importance'] = comparison_data[imp_cols].mean(axis=1)
        comparison_data = comparison_data.sort_values('avg_importance', ascending=False).head(top_n)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(comparison_data))
        width = 0.25
        
        colors = {'model': '#2ecc71', 'permutation': '#3498db', 'shap': '#e74c3c'}
        
        for i, col in enumerate(imp_cols):
            offset = width * (i - len(imp_cols)/2 + 0.5)
            ax.barh(x + offset, comparison_data[col], width, 
                   label=col.capitalize(), color=colors.get(col, 'gray'), alpha=0.8)
        
        ax.set_yticks(x)
        ax.set_yticklabels(comparison_data['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Normalized Importance', fontsize=12)
        ax.set_title(f'Feature Importance Comparison - {self.model_name}', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"importance_comparison_{self.model_name}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved importance comparison to {filepath}")
        
        return fig
    
    def plot_partial_dependence(
        self,
        X: np.ndarray,
        features: Optional[List[Union[str, int]]] = None,
        n_cols: int = 3,
        save: bool = True
    ) -> plt.Figure:
        """
        Create Partial Dependence Plots.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        features : list, optional
            Features to plot (auto-selects top features if None)
        n_cols : int
            Number of columns in subplot grid
        save : bool
            Whether to save the plot
        
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        self.logger.info("Creating Partial Dependence Plots...")
        
        # Auto-select features if not provided
        if features is None:
            # Get top features from model importance
            model_imp = self.get_model_feature_importance()
            if model_imp:
                imp_df = model_imp.to_dataframe()
                n_features = self.pdp_config.get('n_features', 6)
                top_features = imp_df.head(n_features)['feature'].tolist()
                features = [self.feature_names.index(f) for f in top_features if f in self.feature_names]
            else:
                features = list(range(min(6, len(self.feature_names))))
        
        # Convert feature names to indices
        feature_indices = []
        for f in features:
            if isinstance(f, str):
                if f in self.feature_names:
                    feature_indices.append(self.feature_names.index(f))
            else:
                feature_indices.append(f)
        
        n_features = len(feature_indices)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (feature_idx, ax) in enumerate(zip(feature_indices, axes)):
            try:
                PartialDependenceDisplay.from_estimator(
                    self.model, X, [feature_idx],
                    feature_names=self.feature_names,
                    ax=ax,
                    kind='average'
                )
                ax.set_title(self.feature_names[feature_idx], fontsize=11)
            except Exception as e:
                self.logger.warning(f"Failed to plot PDP for feature {feature_idx}: {e}")
                ax.set_visible(False)
        
        # Hide unused axes
        for idx in range(len(feature_indices), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Partial Dependence Plots - {self.model_name}', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"pdp_{self.model_name}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved PDP to {filepath}")
        
        return fig
    
    def plot_prediction_explanation(
        self,
        explanation: PredictionExplanation,
        save: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize a prediction explanation.
        
        Parameters
        ----------
        explanation : PredictionExplanation
            Explanation object from explain_prediction
        save : bool
            Whether to save the plot
        filename : str, optional
            Output filename
        
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Probability gauge
        ax1 = axes[0]
        prob = explanation.probability
        
        # Create a simple bar for probability
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
        thresholds = [0.15, 0.3, 0.5, 1.0]
        
        for i, (thresh, color) in enumerate(zip(thresholds, colors)):
            prev = thresholds[i-1] if i > 0 else 0
            ax1.barh(0, thresh - prev, left=prev, height=0.5, color=color, alpha=0.3)
        
        ax1.axvline(prob, color='black', linewidth=3, label=f'Prediction: {prob:.1%}')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_xlabel('No-Show Probability', fontsize=12)
        ax1.set_title(f'Risk Assessment: {explanation.risk_tier}', fontsize=14)
        ax1.set_yticks([])
        ax1.legend(loc='upper right')
        
        # Add risk labels
        risk_labels = ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH/CRITICAL']
        positions = [0.075, 0.225, 0.4, 0.75]
        for label, pos in zip(risk_labels, positions):
            ax1.text(pos, -0.3, label, ha='center', fontsize=9)
        
        # Right: Top contributing features
        ax2 = axes[1]
        
        # Combine positive and negative features
        all_features = explanation.top_positive_features + explanation.top_negative_features
        all_features = sorted(all_features, key=lambda x: abs(x[2]), reverse=True)[:10]
        
        if all_features:
            names = [f[0] for f in all_features]
            impacts = [f[2] for f in all_features]
            colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in impacts]
            
            y_pos = np.arange(len(names))
            ax2.barh(y_pos, impacts, color=colors, alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(names)
            ax2.invert_yaxis()
            ax2.axvline(0, color='black', linewidth=0.5)
            ax2.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
            ax2.set_title('Top Contributing Features', fontsize=14)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#e74c3c', alpha=0.7, label='Increases no-show risk'),
                Patch(facecolor='#2ecc71', alpha=0.7, label='Decreases no-show risk')
            ]
            ax2.legend(handles=legend_elements, loc='lower right', fontsize=9)
        
        plt.suptitle(f'Prediction Explanation (Probability: {prob:.1%})', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save:
            if filename is None:
                filename = f"prediction_explanation_{self.model_name}.png"
            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved prediction explanation to {filepath}")
        
        return fig
    
    def generate_all_plots(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_sample: Optional[np.ndarray] = None
    ) -> None:
        """
        Generate all interpretability plots.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        X_sample : np.ndarray, optional
            Sample for individual explanation
        """
        self.logger.info("Generating all interpretability plots...")
        
        # SHAP plots
        if SHAP_AVAILABLE and self.explainer is not None:
            try:
                self.plot_shap_summary(X, plot_type='dot')
                plt.close()
                
                self.plot_shap_bar(X)
                plt.close()
                
                self.plot_top_dependence_plots(X, top_n=4)
                
                if X_sample is not None:
                    self.plot_shap_waterfall(X_sample)
                    plt.close()
                    
            except Exception as e:
                self.logger.warning(f"Failed to generate some SHAP plots: {e}")
        
        # Feature importance comparison
        try:
            self.plot_feature_importance_comparison(X, y)
            plt.close()
        except Exception as e:
            self.logger.warning(f"Failed to generate importance comparison: {e}")
        
        # Partial dependence plots
        try:
            self.plot_partial_dependence(X)
            plt.close()
        except Exception as e:
            self.logger.warning(f"Failed to generate PDP: {e}")
        
        # Individual explanation
        if X_sample is not None:
            try:
                explanation = self.explain_prediction(X_sample[0] if X_sample.ndim == 2 else X_sample)
                self.plot_prediction_explanation(explanation)
                plt.close()
            except Exception as e:
                self.logger.warning(f"Failed to generate prediction explanation: {e}")
        
        self.logger.info("All plots generated!")
    
    def save_results(
        self,
        output_dir: Union[str, Path]
    ) -> Dict[str, Path]:
        """
        Save all interpretability results.
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory
        
        Returns
        -------
        dict
            Dictionary of artifact_name -> file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save importance results
        for name, importance in self.importance_results.items():
            df = importance.to_dataframe()
            filepath = output_dir / f"importance_{name}_{self.model_name}.csv"
            df.to_csv(filepath, index=False)
            saved_files[f'importance_{name}'] = filepath
        
        # Save SHAP values if available
        if self.shap_values is not None:
            shap_path = output_dir / f"shap_values_{self.model_name}.npy"
            np.save(shap_path, self.shap_values)
            saved_files['shap_values'] = shap_path
        
        self.logger.info(f"Saved interpretability results to {output_dir}")
        
        return saved_files


def create_interpretation_report(
    interpreter: ModelInterpreter,
    X: np.ndarray,
    y: np.ndarray,
    output_path: Union[str, Path]
) -> str:
    """
    Create a text report summarizing model interpretability.
    
    Parameters
    ----------
    interpreter : ModelInterpreter
        Fitted interpreter
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    output_path : str or Path
        Output file path
    
    Returns
    -------
    str
        Report content
    """
    lines = [
        "=" * 60,
        "MODEL INTERPRETABILITY REPORT",
        "=" * 60,
        f"Model: {interpreter.model_name}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Number of features: {len(interpreter.feature_names)}",
        f"Number of samples analyzed: {len(X)}",
        "",
        "-" * 60,
        "TOP 10 MOST IMPORTANT FEATURES",
        "-" * 60,
    ]
    
    # Get importance
    model_imp = interpreter.get_model_feature_importance()
    if model_imp:
        imp_df = model_imp.to_dataframe()
        for i, row in imp_df.head(10).iterrows():
            lines.append(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
    
    # SHAP analysis
    if interpreter.shap_values is not None:
        lines.extend([
            "",
            "-" * 60,
            "SHAP ANALYSIS SUMMARY",
            "-" * 60,
        ])
        
        shap_imp = interpreter.get_shap_importance()
        if shap_imp:
            lines.append("\nTop features by mean |SHAP|:")
            shap_df = shap_imp.to_dataframe()
            for i, row in shap_df.head(10).iterrows():
                lines.append(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
    
    # Sample prediction explanation
    lines.extend([
        "",
        "-" * 60,
        "SAMPLE PREDICTION EXPLANATION",
        "-" * 60,
    ])
    
    # Explain a high-risk prediction
    probs = interpreter.model.predict_proba(X)[:, 1]
    high_risk_idx = np.argmax(probs)
    explanation = interpreter.explain_prediction(X[high_risk_idx])
    
    lines.extend([
        f"Sample with highest risk (probability: {explanation.probability:.1%})",
        f"Risk tier: {explanation.risk_tier}",
        "",
        "Factors INCREASING no-show risk:",
    ])
    
    for feat, val, impact in explanation.top_positive_features[:5]:
        lines.append(f"  • {feat}: {impact:+.4f}")
    
    lines.append("\nFactors DECREASING no-show risk:")
    for feat, val, impact in explanation.top_negative_features[:5]:
        lines.append(f"  • {feat}: {impact:+.4f}")
    
    lines.extend([
        "",
        "=" * 60,
        "END OF REPORT",
        "=" * 60,
    ])
    
    report = "\n".join(lines)
    
    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    return report
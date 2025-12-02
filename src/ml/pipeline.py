"""
ML Pipeline Module
==================
Complete sklearn pipelines for no-show prediction.

This module provides:
- End-to-end ML pipelines (preprocessing + model)
- Pipeline persistence and loading
- Pipeline inspection utilities
- Production-ready pipeline builder
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import json

# sklearn imports
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA

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


class FeatureNameTracker(BaseEstimator, TransformerMixin):
    """
    Transformer that tracks feature names through the pipeline.
    
    This is useful for maintaining feature names after transformations,
    especially for interpretability purposes.
    """
    
    def __init__(self):
        self.feature_names_in_ = None
        self.feature_names_out_ = None
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        elif hasattr(X, 'shape'):
            self.feature_names_in_ = [f'feature_{i}' for i in range(X.shape[1])]
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            self.feature_names_out_ = X.columns.tolist()
            return X.values
        return X
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_ or self.feature_names_in_ or [])


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clip outliers using IQR method.
    
    Parameters
    ----------
    factor : float
        IQR multiplier for defining outliers (default: 1.5)
    """
    
    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_bounds_ = None
        self.upper_bounds_ = None
    
    def fit(self, X, y=None):
        X = np.array(X)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        
        self.lower_bounds_ = q1 - self.factor * iqr
        self.upper_bounds_ = q3 + self.factor * iqr
        
        return self
    
    def transform(self, X):
        X = np.array(X).copy()
        X = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        return X


class NoShowPipelineBuilder:
    """
    Builder class for creating complete ML pipelines.
    
    This class provides a fluent interface for building sklearn pipelines
    with preprocessing and model components.
    
    Attributes
    ----------
    config : dict
        ML configuration dictionary
    pipeline : Pipeline
        Built sklearn pipeline
    
    Example
    -------
    >>> builder = NoShowPipelineBuilder(config)
    >>> pipeline = (builder
    ...     .with_preprocessing()
    ...     .with_feature_selection(k=20)
    ...     .with_model('random_forest')
    ...     .build())
    >>> pipeline.fit(X_train, y_train)
    """
    
    def __init__(self, config: dict):
        """
        Initialize the pipeline builder.
        
        Parameters
        ----------
        config : dict
            ML configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("healthcare_pipeline.ml.PipelineBuilder")
        
        # Pipeline components
        self._preprocessor = None
        self._feature_selector = None
        self._model = None
        self._steps = []
        
        # Feature configuration
        self.numeric_features = config['features'].get('numeric', [])
        self.categorical_features = config['features'].get('categorical', [])
        self.binary_features = config['features'].get('binary', [])
        
        # Validated features (set during build)
        self._valid_numeric = []
        self._valid_categorical = []
        self._valid_binary = []
        
        self.logger.info("NoShowPipelineBuilder initialized")
    
    def validate_features(self, df: pd.DataFrame) -> 'NoShowPipelineBuilder':
        """
        Validate that configured features exist in the dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        
        Returns
        -------
        self
            For method chaining
        """
        available_cols = set(df.columns.str.lower())
        df_cols_map = {col.lower(): col for col in df.columns}
        
        # Validate each feature type
        self._valid_numeric = [
            df_cols_map[f.lower()] for f in self.numeric_features 
            if f.lower() in available_cols
        ]
        self._valid_categorical = [
            df_cols_map[f.lower()] for f in self.categorical_features 
            if f.lower() in available_cols
        ]
        self._valid_binary = [
            df_cols_map[f.lower()] for f in self.binary_features 
            if f.lower() in available_cols
        ]
        
        self.logger.info(f"Validated features: {len(self._valid_numeric)} numeric, "
                        f"{len(self._valid_categorical)} categorical, "
                        f"{len(self._valid_binary)} binary")
        
        return self
    
    def with_preprocessing(
        self,
        numeric_scaler: str = 'standard',
        handle_outliers: bool = False
    ) -> 'NoShowPipelineBuilder':
        """
        Add preprocessing step to the pipeline.
        
        Parameters
        ----------
        numeric_scaler : str
            Type of scaler: 'standard', 'minmax', 'robust'
        handle_outliers : bool
            Whether to clip outliers
        
        Returns
        -------
        self
            For method chaining
        """
        preprocessing_config = self.config['preprocessing']
        
        # --- Numeric Transformer ---
        numeric_steps = [
            ('imputer', SimpleImputer(
                strategy=preprocessing_config['numeric_strategy'].get('imputer', 'median')
            ))
        ]
        
        if handle_outliers:
            numeric_steps.append(('outlier_clipper', OutlierClipper()))
        
        # Select scaler
        if numeric_scaler == 'standard':
            numeric_steps.append(('scaler', StandardScaler()))
        elif numeric_scaler == 'minmax':
            numeric_steps.append(('scaler', MinMaxScaler()))
        elif numeric_scaler == 'robust':
            numeric_steps.append(('scaler', RobustScaler()))
        
        numeric_transformer = Pipeline(steps=numeric_steps)
        
        # --- Categorical Transformer ---
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False,
                drop='first'
            ))
        ])
        
        # --- Binary Transformer ---
        binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])
        
        # --- Column Transformer ---
        transformers = []
        
        if self._valid_numeric:
            transformers.append(('numeric', numeric_transformer, self._valid_numeric))
        if self._valid_categorical:
            transformers.append(('categorical', categorical_transformer, self._valid_categorical))
        if self._valid_binary:
            transformers.append(('binary', binary_transformer, self._valid_binary))
        
        self._preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',
            verbose_feature_names_out=True
        )
        
        self._steps.append(('preprocessor', self._preprocessor))
        self.logger.info("Added preprocessing step")
        
        return self
    
    def with_feature_selection(
        self,
        method: str = 'kbest',
        k: int = 20,
        score_func: str = 'f_classif'
    ) -> 'NoShowPipelineBuilder':
        """
        Add feature selection step.
        
        Parameters
        ----------
        method : str
            Selection method: 'kbest', 'pca'
        k : int
            Number of features to select
        score_func : str
            Scoring function for SelectKBest: 'f_classif', 'mutual_info'
        
        Returns
        -------
        self
            For method chaining
        """
        if method == 'kbest':
            if score_func == 'f_classif':
                func = f_classif
            elif score_func == 'mutual_info':
                func = mutual_info_classif
            else:
                func = f_classif
            
            self._feature_selector = SelectKBest(score_func=func, k=k)
            
        elif method == 'pca':
            self._feature_selector = PCA(n_components=k)
        
        self._steps.append(('feature_selection', self._feature_selector))
        self.logger.info(f"Added feature selection: {method} (k={k})")
        
        return self
    
    def with_model(
        self,
        model_name: str = 'random_forest',
        custom_params: Optional[Dict] = None
    ) -> 'NoShowPipelineBuilder':
        """
        Add model to the pipeline.
        
        Parameters
        ----------
        model_name : str
            Name of the model
        custom_params : dict, optional
            Custom parameters to override defaults
        
        Returns
        -------
        self
            For method chaining
        """
        models_config = self.config.get('baseline_models', {})
        
        # Get default params from config
        model_config = models_config.get(model_name, {})
        params = model_config.get('params', {})
        
        # Override with custom params
        if custom_params:
            params.update(custom_params)
        
        # Create model instance
        if model_name == 'logistic_regression':
            self._model = LogisticRegression(**params)
        elif model_name == 'random_forest':
            self._model = RandomForestClassifier(**params)
        elif model_name == 'gradient_boosting':
            self._model = GradientBoostingClassifier(**params)
        elif model_name == 'decision_tree':
            self._model = DecisionTreeClassifier(**params)
        elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
            self._model = XGBClassifier(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self._steps.append(('classifier', self._model))
        self.logger.info(f"Added model: {model_name}")
        
        return self
    
    def with_custom_step(
        self,
        name: str,
        transformer: BaseEstimator
    ) -> 'NoShowPipelineBuilder':
        """
        Add a custom transformer step.
        
        Parameters
        ----------
        name : str
            Step name
        transformer : BaseEstimator
            sklearn-compatible transformer
        
        Returns
        -------
        self
            For method chaining
        """
        self._steps.append((name, transformer))
        self.logger.info(f"Added custom step: {name}")
        return self
    
    def build(self) -> Pipeline:
        """
        Build and return the final pipeline.
        
        Returns
        -------
        Pipeline
            Complete sklearn pipeline
        """
        if not self._steps:
            raise ValueError("No steps added to pipeline. Add at least a model.")
        
        memory = self.config.get('pipeline', {}).get('memory', None)
        verbose = self.config.get('pipeline', {}).get('verbose', False)
        
        pipeline = Pipeline(
            steps=self._steps,
            memory=memory,
            verbose=verbose
        )
        
        self.logger.info(f"Built pipeline with {len(self._steps)} steps: "
                        f"{[step[0] for step in self._steps]}")
        
        return pipeline
    
    def reset(self) -> 'NoShowPipelineBuilder':
        """
        Reset the builder state.
        
        Returns
        -------
        self
            For method chaining
        """
        self._preprocessor = None
        self._feature_selector = None
        self._model = None
        self._steps = []
        return self


class PipelineManager:
    """
    Manager for training, evaluating, and persisting pipelines.
    
    This class handles:
    - Pipeline training with cross-validation
    - Pipeline persistence (save/load)
    - Pipeline inspection and visualization
    - Production pipeline packaging
    """
    
    def __init__(self, config: dict):
        """
        Initialize the pipeline manager.
        
        Parameters
        ----------
        config : dict
            ML configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("healthcare_pipeline.ml.PipelineManager")
        
        self.pipelines: Dict[str, Pipeline] = {}
        self.pipeline_metadata: Dict[str, Dict] = {}
        
        self.logger.info("PipelineManager initialized")
    
    def create_pipeline(
        self,
        df: pd.DataFrame,
        model_name: str = 'random_forest',
        include_feature_selection: bool = False,
        feature_selection_k: int = 20
    ) -> Pipeline:
        """
        Create a complete pipeline for a given model.
        
        Parameters
        ----------
        df : pd.DataFrame
            Sample dataframe for feature validation
        model_name : str
            Name of the model to use
        include_feature_selection : bool
            Whether to include feature selection
        feature_selection_k : int
            Number of features to select
        
        Returns
        -------
        Pipeline
            Configured sklearn pipeline
        """
        builder = NoShowPipelineBuilder(self.config)
        builder.validate_features(df)
        builder.with_preprocessing()
        
        if include_feature_selection:
            builder.with_feature_selection(k=feature_selection_k)
        
        builder.with_model(model_name)
        
        pipeline = builder.build()
        self.pipelines[model_name] = pipeline
        
        # Store metadata
        self.pipeline_metadata[model_name] = {
            'created_at': datetime.now().isoformat(),
            'steps': [step[0] for step in pipeline.steps],
            'include_feature_selection': include_feature_selection
        }
        
        return pipeline
    
    def create_all_pipelines(
        self,
        df: pd.DataFrame,
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Pipeline]:
        """
        Create pipelines for all specified models.
        
        Parameters
        ----------
        df : pd.DataFrame
            Sample dataframe for feature validation
        model_names : list, optional
            List of model names (default: all enabled in config)
        
        Returns
        -------
        dict
            Dictionary of model_name -> Pipeline
        """
        if model_names is None:
            models_config = self.config.get('baseline_models', {})
            model_names = [
                name for name, cfg in models_config.items()
                if cfg.get('enabled', True)
            ]
            
            # Add xgboost if available
            if XGBOOST_AVAILABLE and 'xgboost' not in model_names:
                if self.config.get('baseline_models', {}).get('xgboost', {}).get('enabled', True):
                    model_names.append('xgboost')
        
        for model_name in model_names:
            try:
                self.create_pipeline(df, model_name)
                self.logger.info(f"Created pipeline for {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to create pipeline for {model_name}: {e}")
        
        return self.pipelines
    
    def fit_pipeline(
        self,
        pipeline_name: str,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Pipeline:
        """
        Fit a pipeline on training data.
        
        Parameters
        ----------
        pipeline_name : str
            Name of the pipeline
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector
        
        Returns
        -------
        Pipeline
            Fitted pipeline
        """
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline '{pipeline_name}' not found")
        
        pipeline = self.pipelines[pipeline_name]
        
        self.logger.info(f"Fitting pipeline: {pipeline_name}")
        start_time = datetime.now()
        
        pipeline.fit(X, y)
        
        training_time = (datetime.now() - start_time).total_seconds()
        self.pipeline_metadata[pipeline_name]['training_time'] = training_time
        self.pipeline_metadata[pipeline_name]['is_fitted'] = True
        
        self.logger.info(f"Pipeline {pipeline_name} fitted in {training_time:.2f}s")
        
        return pipeline
    
    def get_feature_names(self, pipeline_name: str) -> List[str]:
        """
        Get feature names from a fitted pipeline.
        
        Parameters
        ----------
        pipeline_name : str
            Name of the pipeline
        
        Returns
        -------
        list
            List of feature names
        """
        pipeline = self.pipelines.get(pipeline_name)
        if pipeline is None:
            raise ValueError(f"Pipeline '{pipeline_name}' not found")
        
        # Try to get feature names from preprocessor
        if 'preprocessor' in pipeline.named_steps:
            preprocessor = pipeline.named_steps['preprocessor']
            if hasattr(preprocessor, 'get_feature_names_out'):
                return preprocessor.get_feature_names_out().tolist()
        
        return []
    
    def save_pipeline(
        self,
        pipeline_name: str,
        filepath: Union[str, Path]
    ) -> Path:
        """
        Save a pipeline to disk.
        
        Parameters
        ----------
        pipeline_name : str
            Name of the pipeline
        filepath : str or Path
            Output file path
        
        Returns
        -------
        Path
            Path to saved file
        """
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline '{pipeline_name}' not found")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline and metadata together
        save_dict = {
            'pipeline': self.pipelines[pipeline_name],
            'metadata': self.pipeline_metadata.get(pipeline_name, {}),
            'config': self.config
        }
        
        joblib.dump(save_dict, filepath)
        self.logger.info(f"Saved pipeline '{pipeline_name}' to {filepath}")
        
        return filepath
    
    def load_pipeline(
        self,
        filepath: Union[str, Path],
        pipeline_name: Optional[str] = None
    ) -> Pipeline:
        """
        Load a pipeline from disk.
        
        Parameters
        ----------
        filepath : str or Path
            Path to saved pipeline
        pipeline_name : str, optional
            Name to assign to the loaded pipeline
        
        Returns
        -------
        Pipeline
            Loaded pipeline
        """
        filepath = Path(filepath)
        
        save_dict = joblib.load(filepath)
        
        pipeline = save_dict['pipeline']
        metadata = save_dict.get('metadata', {})
        
        if pipeline_name is None:
            pipeline_name = filepath.stem
        
        self.pipelines[pipeline_name] = pipeline
        self.pipeline_metadata[pipeline_name] = metadata
        
        self.logger.info(f"Loaded pipeline '{pipeline_name}' from {filepath}")
        
        return pipeline
    
    def save_all_pipelines(
        self,
        output_dir: Union[str, Path]
    ) -> Dict[str, Path]:
        """
        Save all pipelines to a directory.
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory
        
        Returns
        -------
        dict
            Dictionary of pipeline_name -> saved file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        for name in self.pipelines:
            filepath = output_dir / f"{name}_pipeline.joblib"
            self.save_pipeline(name, filepath)
            saved_paths[name] = filepath
        
        # Save metadata summary
        metadata_path = output_dir / "pipelines_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.pipeline_metadata, f, indent=2, default=str)
        
        self.logger.info(f"Saved {len(saved_paths)} pipelines to {output_dir}")
        
        return saved_paths
    
    def get_pipeline_summary(self, pipeline_name: str) -> str:
        """
        Get a human-readable summary of a pipeline.
        
        Parameters
        ----------
        pipeline_name : str
            Name of the pipeline
        
        Returns
        -------
        str
            Summary string
        """
        if pipeline_name not in self.pipelines:
            return f"Pipeline '{pipeline_name}' not found"
        
        pipeline = self.pipelines[pipeline_name]
        metadata = self.pipeline_metadata.get(pipeline_name, {})
        
        lines = [
            f"Pipeline: {pipeline_name}",
            "=" * 40,
            f"Created: {metadata.get('created_at', 'Unknown')}",
            f"Fitted: {metadata.get('is_fitted', False)}",
            f"Training time: {metadata.get('training_time', 'N/A')}s",
            "",
            "Steps:",
        ]
        
        for i, (name, step) in enumerate(pipeline.steps):
            step_type = type(step).__name__
            lines.append(f"  {i+1}. {name}: {step_type}")
        
        return "\n".join(lines)


def create_production_pipeline(
    config: dict,
    df: pd.DataFrame,
    model_name: str = 'random_forest',
    model_params: Optional[Dict] = None
) -> Pipeline:
    """
    Create a production-ready pipeline.
    
    This is a convenience function for creating a complete pipeline
    suitable for production deployment.
    
    Parameters
    ----------
    config : dict
        ML configuration
    df : pd.DataFrame
        Sample dataframe for feature validation
    model_name : str
        Name of the model
    model_params : dict, optional
        Custom model parameters
    
    Returns
    -------
    Pipeline
        Production-ready pipeline
    """
    builder = NoShowPipelineBuilder(config)
    builder.validate_features(df)
    builder.with_preprocessing(handle_outliers=True)
    builder.with_model(model_name, model_params)
    
    return builder.build()
"""
ML Preprocessing Module
=======================
Handles data preparation for machine learning models.

This module provides:
- Feature selection based on configuration
- Train/test splitting with stratification
- sklearn ColumnTransformer for mixed data types
- Custom transformers for healthcare-specific preprocessing
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import joblib


class NoShowPreprocessor:
    """
    Preprocessor for Healthcare No-Show Prediction.
    
    This class handles all data preprocessing steps required
    before training ML models, including:
    - Feature selection and validation
    - Train/test splitting with stratification
    - Building sklearn preprocessing pipelines
    - Feature type-specific transformations
    
    Attributes
    ----------
    config : dict
        ML configuration dictionary
    feature_names_ : list
        List of feature names after preprocessing
    preprocessor_ : ColumnTransformer
        Fitted sklearn ColumnTransformer
    
    Example
    -------
    >>> preprocessor = NoShowPreprocessor(ml_config)
    >>> X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    >>> X_train_transformed = preprocessor.fit_transform(X_train, y_train)
    >>> X_test_transformed = preprocessor.transform(X_test)
    """
    
    def __init__(self, config: dict):
        """
        Initialize the preprocessor with configuration.
        
        Parameters
        ----------
        config : dict
            ML configuration dictionary containing feature lists,
            preprocessing strategies, and splitting parameters
        """
        self.config = config
        self.logger = logging.getLogger("healthcare_pipeline.ml.Preprocessor")
        
        # Feature configuration
        self.target_col = config['ml_project']['target_column']
        self.numeric_features = config['features'].get('numeric', [])
        self.categorical_features = config['features'].get('categorical', [])
        self.binary_features = config['features'].get('binary', [])
        self.exclude_features = config['features'].get('exclude', [])
        
        # Splitting configuration
        self.test_size = config['splitting']['test_size']
        self.random_state = config['splitting']['random_state']
        self.stratify = config['splitting']['stratify']
        
        # State
        self.preprocessor_ = None
        self.feature_names_ = None
        self.is_fitted_ = False
        
        self.logger.info("NoShowPreprocessor initialized")
        self.logger.info(f"  Target: {self.target_col}")
        self.logger.info(f"  Numeric features: {len(self.numeric_features)}")
        self.logger.info(f"  Categorical features: {len(self.categorical_features)}")
        self.logger.info(f"  Binary features: {len(self.binary_features)}")
    
    def validate_features(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """
        Validate that configured features exist in the dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        
        Returns
        -------
        tuple
            (valid_numeric, valid_categorical, valid_binary) feature lists
        
        Raises
        ------
        ValueError
            If target column is not found
        """
        self.logger.info("Validating features...")
        
        available_cols = set(df.columns.str.lower())
        df_cols_map = {col.lower(): col for col in df.columns}
        
        # Check target column
        target_lower = self.target_col.lower()
        if target_lower not in available_cols:
            raise ValueError(f"Target column '{self.target_col}' not found in dataframe")
        
        # Validate numeric features
        valid_numeric = []
        for feat in self.numeric_features:
            if feat.lower() in available_cols:
                valid_numeric.append(df_cols_map[feat.lower()])
            else:
                self.logger.warning(f"Numeric feature '{feat}' not found, skipping")
        
        # Validate categorical features
        valid_categorical = []
        for feat in self.categorical_features:
            if feat.lower() in available_cols:
                valid_categorical.append(df_cols_map[feat.lower()])
            else:
                self.logger.warning(f"Categorical feature '{feat}' not found, skipping")
        
        # Validate binary features
        valid_binary = []
        for feat in self.binary_features:
            if feat.lower() in available_cols:
                valid_binary.append(df_cols_map[feat.lower()])
            else:
                self.logger.warning(f"Binary feature '{feat}' not found, skipping")
        
        self.logger.info(f"Validated features:")
        self.logger.info(f"  Numeric: {len(valid_numeric)}/{len(self.numeric_features)}")
        self.logger.info(f"  Categorical: {len(valid_categorical)}/{len(self.categorical_features)}")
        self.logger.info(f"  Binary: {len(valid_binary)}/{len(self.binary_features)}")
        
        return valid_numeric, valid_categorical, valid_binary
    
    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Select features and target from dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with all columns
        
        Returns
        -------
        tuple
            (X, y) where X is features DataFrame and y is target Series
        """
        self.logger.info("Selecting features...")
        
        # Validate and get available features
        valid_numeric, valid_categorical, valid_binary = self.validate_features(df)
        
        # Combine all features
        all_features = valid_numeric + valid_categorical + valid_binary
        
        if len(all_features) == 0:
            raise ValueError("No valid features found!")
        
        # Store validated feature lists
        self._valid_numeric = valid_numeric
        self._valid_categorical = valid_categorical
        self._valid_binary = valid_binary
        
        # Find actual target column name (case-insensitive)
        target_col_actual = None
        for col in df.columns:
            if col.lower() == self.target_col.lower():
                target_col_actual = col
                break
        
        # Select X and y
        X = df[all_features].copy()
        y = df[target_col_actual].copy()
        
        self.logger.info(f"Selected {len(all_features)} features")
        self.logger.info(f"Target distribution: {y.value_counts(normalize=True).to_dict()}")
        
        return X, y
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        test_size: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and test sets.
        
        Uses stratified sampling to maintain class distribution.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector
        test_size : float, optional
            Test set proportion (uses config if not provided)
        
        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        if test_size is None:
            test_size = self.test_size
        
        self.logger.info(f"Splitting data (test_size={test_size}, stratify={self.stratify})...")
        
        stratify_param = y if self.stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        self.logger.info(f"Training set: {len(X_train):,} samples")
        self.logger.info(f"Test set: {len(X_test):,} samples")
        self.logger.info(f"Train target distribution: {y_train.value_counts(normalize=True).round(3).to_dict()}")
        self.logger.info(f"Test target distribution: {y_test.value_counts(normalize=True).round(3).to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def build_preprocessor(self) -> ColumnTransformer:
        """
        Build sklearn ColumnTransformer for preprocessing.
        
        Creates separate pipelines for:
        - Numeric features: Imputation + Scaling
        - Categorical features: Imputation + OneHot Encoding
        - Binary features: Imputation only
        
        Returns
        -------
        ColumnTransformer
            Configured preprocessing pipeline
        """
        self.logger.info("Building preprocessing pipeline...")
        
        preprocessing_config = self.config['preprocessing']
        
        # --- Numeric Transformer ---
        numeric_strategy = preprocessing_config['numeric_strategy']
        
        # Select scaler
        scaler_type = numeric_strategy.get('scaler', 'standard')
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=numeric_strategy.get('imputer', 'median'))),
            ('scaler', scaler)
        ])
        
        # --- Categorical Transformer ---
        categorical_strategy = preprocessing_config['categorical_strategy']
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=categorical_strategy.get('imputer', 'most_frequent'))),
            ('encoder', OneHotEncoder(
                handle_unknown=categorical_strategy.get('handle_unknown', 'ignore'),
                sparse_output=False,
                drop='first'  # Avoid multicollinearity
            ))
        ])
        
        # --- Binary Transformer ---
        binary_strategy = preprocessing_config['binary_strategy']
        
        binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=binary_strategy.get('imputer', 'most_frequent')))
        ])
        
        # --- Combine All Transformers ---
        transformers = []
        
        if self._valid_numeric:
            transformers.append(('numeric', numeric_transformer, self._valid_numeric))
            self.logger.info(f"  Added numeric transformer for {len(self._valid_numeric)} features")
        
        if self._valid_categorical:
            transformers.append(('categorical', categorical_transformer, self._valid_categorical))
            self.logger.info(f"  Added categorical transformer for {len(self._valid_categorical)} features")
        
        if self._valid_binary:
            transformers.append(('binary', binary_transformer, self._valid_binary))
            self.logger.info(f"  Added binary transformer for {len(self._valid_binary)} features")
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',  # Drop columns not specified
            verbose_feature_names_out=True
        )
        
        return preprocessor
    
    def fit_transform(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Fit preprocessor and transform data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series, optional
            Target vector (not used but accepted for sklearn compatibility)
        
        Returns
        -------
        np.ndarray
            Transformed feature matrix
        """
        self.logger.info("Fitting and transforming data...")
        
        # Build preprocessor if not already done
        if self.preprocessor_ is None:
            self.preprocessor_ = self.build_preprocessor()
        
        # Fit and transform
        X_transformed = self.preprocessor_.fit_transform(X)
        
        # Get feature names
        self.feature_names_ = self.preprocessor_.get_feature_names_out().tolist()
        self.is_fitted_ = True
        
        self.logger.info(f"Transformed shape: {X_transformed.shape}")
        self.logger.info(f"Number of features after transformation: {len(self.feature_names_)}")
        
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        
        Returns
        -------
        np.ndarray
            Transformed feature matrix
        
        Raises
        ------
        ValueError
            If preprocessor has not been fitted
        """
        if not self.is_fitted_:
            raise ValueError("Preprocessor has not been fitted. Call fit_transform first.")
        
        return self.preprocessor_.transform(X)
    
    def prepare_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Complete data preparation pipeline.
        
        Performs all preprocessing steps:
        1. Select features
        2. Split data
        3. Return ready-for-training data
        
        Note: Does NOT fit the preprocessor - that should be done
        only on training data to prevent data leakage.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw/processed dataframe
        
        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test) as DataFrames/Series
        """
        self.logger.info("="*60)
        self.logger.info("Preparing data for ML training...")
        
        # Step 1: Select features
        X, y = self.select_features(df)
        
        # Step 2: Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        self.logger.info("="*60)
        self.logger.info("Data preparation complete!")
        self.logger.info(f"  Training samples: {len(X_train):,}")
        self.logger.info(f"  Test samples: {len(X_test):,}")
        self.logger.info(f"  Features: {X_train.shape[1]}")
        self.logger.info("="*60)
        
        return X_train, X_test, y_train, y_test
    
    def get_cv_splitter(self) -> StratifiedKFold:
        """
        Get cross-validation splitter.
        
        Returns
        -------
        StratifiedKFold
            Configured CV splitter
        """
        cv_config = self.config['cross_validation']
        
        return StratifiedKFold(
            n_splits=cv_config['n_folds'],
            shuffle=cv_config['shuffle'],
            random_state=self.random_state
        )
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save fitted preprocessor to disk.
        
        Parameters
        ----------
        filepath : str or Path
            Output file path (.joblib)
        """
        if not self.is_fitted_:
            raise ValueError("Preprocessor has not been fitted.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'preprocessor': self.preprocessor_,
            'feature_names': self.feature_names_,
            'config': self.config,
            'valid_numeric': self._valid_numeric,
            'valid_categorical': self._valid_categorical,
            'valid_binary': self._valid_binary
        }
        
        joblib.dump(save_dict, filepath)
        self.logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'NoShowPreprocessor':
        """
        Load fitted preprocessor from disk.
        
        Parameters
        ----------
        filepath : str or Path
            Input file path (.joblib)
        
        Returns
        -------
        NoShowPreprocessor
            Loaded preprocessor
        """
        save_dict = joblib.load(filepath)
        
        instance = cls(save_dict['config'])
        instance.preprocessor_ = save_dict['preprocessor']
        instance.feature_names_ = save_dict['feature_names']
        instance._valid_numeric = save_dict['valid_numeric']
        instance._valid_categorical = save_dict['valid_categorical']
        instance._valid_binary = save_dict['valid_binary']
        instance.is_fitted_ = True
        
        return instance


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer for feature selection.
    
    This transformer selects a subset of features from a DataFrame
    and can be used in sklearn pipelines.
    """
    
    def __init__(self, feature_names: List[str]):
        """
        Initialize with list of feature names to select.
        
        Parameters
        ----------
        feature_names : list
            List of column names to select
        """
        self.feature_names = feature_names
    
    def fit(self, X, y=None):
        """Fit - nothing to do."""
        return self
    
    def transform(self, X):
        """Select specified features."""
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names]
        else:
            raise ValueError("FeatureSelector requires DataFrame input")
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names."""
        return np.array(self.feature_names)
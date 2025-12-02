"""
Machine Learning Module
=======================
This module contains all ML-related functionality for the
Healthcare No-Show Prediction System.

Modules:
--------
- preprocessing: sklearn transformers and data preparation
- train: Model training logic and experiment management
- evaluate: Model evaluation metrics and comparisons
- pipeline: Complete sklearn pipelines
- tuning: Hyperparameter optimization
- interpret: Model interpretability (SHAP, feature importance)

Usage:
------
    from src.ml import (
        NoShowPreprocessor, 
        ModelTrainer, 
        ModelEvaluator,
        NoShowPipelineBuilder,
        PipelineManager,
        HyperparameterTuner,
        ModelInterpreter
    )
"""

from .preprocessing import NoShowPreprocessor
from .train import ModelTrainer
from .evaluate import ModelEvaluator
from .pipeline import NoShowPipelineBuilder, PipelineManager, create_production_pipeline
from .tuning import HyperparameterTuner, TuningResult
from .interpret import ModelInterpreter, create_interpretation_report

__all__ = [
    # Preprocessing
    'NoShowPreprocessor',
    
    # Training
    'ModelTrainer',
    
    # Evaluation  
    'ModelEvaluator',
    
    # Pipeline
    'NoShowPipelineBuilder',
    'PipelineManager',
    'create_production_pipeline',
    
    # Tuning
    'HyperparameterTuner',
    'TuningResult',
    
    # Interpretability
    'ModelInterpreter',
    'create_interpretation_report',
]

__version__ = '1.0.0'
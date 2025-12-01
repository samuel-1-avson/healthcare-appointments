"""
Healthcare Appointments No-Show Analysis Pipeline
=================================================

A comprehensive data pipeline for analyzing medical appointment no-shows
and generating intervention recommendations.
"""

__version__ = "1.0.0"
__author__ = "Data Analytics Team"

# Import main modules for easy access
from .data_loader import DataLoader
from .data_cleaner import DataCleaner
from .feature_engineer import FeatureEngineer
from .risk_scorer import RiskScorer
from .visualizations import Visualizer
from .utils import setup_logging, load_config, timer

__all__ = [
    "DataLoader",
    "DataCleaner", 
    "FeatureEngineer",
    "RiskScorer",
    "Visualizer",
    "setup_logging",
    "load_config",
    "timer"
]
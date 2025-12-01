"""
Utility Functions Module
========================
Helper functions used across the pipeline.
"""

import os
import time
import logging
import yaml
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import Dict, Any, Optional
import pandas as pd
import colorlog


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the pipeline with colored output.
    
    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : str, optional
        Path to log file (if None, only console output)
    
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("healthcare_pipeline")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler with color
    console_handler = colorlog.StreamHandler()
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
    
    Returns
    -------
    dict
        Configuration dictionary
    
    Raises
    ------
    FileNotFoundError
        If config file doesn't exist
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def timer(func):
    """
    Decorator to time function execution.
    
    Parameters
    ----------
    func : callable
        Function to time
    
    Returns
    -------
    callable
        Wrapped function with timing
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        
        logger = logging.getLogger("healthcare_pipeline")
        logger.info(f"{func.__name__} took {end - start:.2f} seconds")
        
        return result
    return wrapper


def create_directories(config: Dict[str, Any]) -> None:
    """
    Create all necessary directories from config.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    """
    directories = [
        config['paths']['data_dir'],
        os.path.join(config['paths']['data_dir'], 'raw'),
        os.path.join(config['paths']['data_dir'], 'processed'),
        config['paths']['outputs_dir'],
        config['paths']['figures_dir'],
        config['paths']['reports_dir'],
        config['paths']['sql_dir']
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("healthcare_pipeline")
    logger.info(f"Created {len(directories)} directories")


def save_dataframe(
    df: pd.DataFrame,
    filepath: str,
    index: bool = False
) -> None:
    """
    Save DataFrame with logging and error handling.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    filepath : str
        Path to save file
    index : bool
        Whether to include index
    
    Raises
    ------
    IOError
        If save fails
    """
    logger = logging.getLogger("healthcare_pipeline")
    
    try:
        # Create directory if doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on extension
        if filepath.endswith('.csv'):
            df.to_csv(filepath, index=index)
        elif filepath.endswith('.parquet'):
            df.to_parquet(filepath, index=index)
        elif filepath.endswith('.xlsx'):
            df.to_excel(filepath, index=index)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        logger.info(f"Saved {len(df):,} rows to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save {filepath}: {str(e)}")
        raise


def generate_report_metadata() -> Dict[str, Any]:
    """
    Generate metadata for reports.
    
    Returns
    -------
    dict
        Metadata dictionary
    """
    return {
        'generated_at': datetime.now().isoformat(),
        'version': '1.0.0',
        'author': 'Healthcare Analytics Pipeline',
        'python_version': '3.9+',
        'pipeline_stage': 'complete'
    }


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list,
    min_rows: int = 1
) -> bool:
    """
    Validate DataFrame has required structure.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : list
        Required column names
    min_rows : int
        Minimum number of rows
    
    Returns
    -------
    bool
        True if valid, raises exception otherwise
    
    Raises
    ------
    ValueError
        If validation fails
    """
    # Check if DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input is not a pandas DataFrame")
    
    # Check minimum rows
    if len(df) < min_rows:
        raise ValueError(f"DataFrame has {len(df)} rows, minimum is {min_rows}")
    
    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True


def calculate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate summary statistics for report.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns
    -------
    dict
        Statistics dictionary
    """
    stats = {
        'total_appointments': len(df),
        'unique_patients': df['patientid'].nunique() if 'patientid' in df.columns else None,
        'date_range': {
            'start': df['appointmentday'].min() if 'appointmentday' in df.columns else None,
            'end': df['appointmentday'].max() if 'appointmentday' in df.columns else None
        },
        'no_show_rate': df['no_show'].mean() if 'no_show' in df.columns else None,
        'total_no_shows': df['no_show'].sum() if 'no_show' in df.columns else None
    }
    
    return stats


def print_pipeline_banner() -> None:
    """Print a nice banner for pipeline start."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     HEALTHCARE APPOINTMENTS NO-SHOW ANALYSIS PIPELINE       ║
    ║                         Version 1.0.0                        ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_pipeline_summary(stats: Dict[str, Any]) -> None:
    """
    Print summary statistics.
    
    Parameters
    ----------
    stats : dict
        Statistics dictionary from calculate_statistics
    """
    summary = f"""
    ┌────────────────────────────────────────────────────────────┐
    │                    PIPELINE SUMMARY                        │
    ├────────────────────────────────────────────────────────────┤
    │  Total Appointments: {stats['total_appointments']:,}
    │  Unique Patients: {stats['unique_patients']:,}
    │  No-Show Rate: {stats['no_show_rate']*100:.1f}%
    │  Total No-Shows: {stats['total_no_shows']:,}
    └────────────────────────────────────────────────────────────┘
    """
    print(summary)
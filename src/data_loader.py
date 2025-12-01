"""
Data Loading Module
===================
Functions to load appointment data from various sources.
"""

import pandas as pd
import sqlite3
import logging
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

from .utils import timer, validate_dataframe


class DataLoader:
    """Handle loading data from multiple sources."""
    
    def __init__(self, config: dict):
        """
        Initialize DataLoader with configuration.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("healthcare_pipeline.DataLoader")
    
    @timer
    def load_csv(
        self, 
        filepath: Optional[str] = None,
        encoding: str = 'utf-8'
    ) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Parameters
        ----------
        filepath : str, optional
            Path to CSV file (uses config if not provided)
        encoding : str
            File encoding
        
        Returns
        -------
        pd.DataFrame
            Loaded data
        
        Raises
        ------
        FileNotFoundError
            If file doesn't exist
        """
        if filepath is None:
            filepath = self.config['paths']['raw_data']
        
        file_path = Path(filepath)
        
        if not file_path.exists():
            self.logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            self.logger.info(f"Loading CSV from {filepath}")
            df = pd.read_csv(filepath, encoding=encoding)
            self.logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading CSV: {str(e)}")
            raise
    
    @timer
    def load_from_url(
        self,
        url: Optional[str] = None,
        encoding: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load data from URL.
        
        Parameters
        ----------
        url : str, optional
            URL to load from (uses config if not provided)
        encoding : str, optional
            File encoding (uses config if not provided)
        
        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        if url is None:
            url = self.config['source']['url']
        if encoding is None:
            encoding = self.config['source'].get('encoding', 'utf-8')
        
        self.logger.info(f"Loading data from URL: {url}")
        
        try:
            df = pd.read_csv(url, encoding=encoding)
            self.logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
            
            # Save raw data locally
            if self.config['pipeline'].get('save_intermediate', True):
                raw_path = self.config['paths']['raw_data']
                Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(raw_path, index=False)
                self.logger.info(f"Saved raw data to {raw_path}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading from URL: {str(e)}")
            raise
    
    @timer
    def load_from_database(
        self,
        connection_string: Optional[str] = None,
        query: str = "SELECT * FROM appointments",
        table: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load data from SQL database.
        
        Parameters
        ----------
        connection_string : str, optional
            Database connection string (uses SQLite from config if not provided)
        query : str
            SQL query to execute
        table : str, optional
            Table name (alternative to query)
        
        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        if connection_string is None:
            connection_string = self.config['paths']['database']
        
        self.logger.info(f"Loading data from database: {connection_string}")
        
        try:
            # Handle SQLite connection
            if connection_string.endswith('.db'):
                conn = sqlite3.connect(connection_string)
            else:
                # For other databases, would need appropriate connection
                raise NotImplementedError("Only SQLite currently supported")
            
            # Load data
            if table:
                df = pd.read_sql_table(table, conn)
            else:
                df = pd.read_sql_query(query, conn)
            
            conn.close()
            
            self.logger.info(f"Loaded {len(df):,} rows from database")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading from database: {str(e)}")
            raise
    
    def load(self, source: str = "auto") -> pd.DataFrame:
        """
        Smart loader that detects source type.
        
        Parameters
        ----------
        source : str
            Source type ('csv', 'url', 'database', 'auto')
        
        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        self.logger.info(f"Loading data with source type: {source}")
        
        if source == "auto":
            # Try to detect based on configuration
            raw_path = Path(self.config['paths']['raw_data'])
            if raw_path.exists():
                self.logger.info("Found existing raw data file")
                return self.load_csv()
            else:
                self.logger.info("Raw data not found, loading from URL")
                return self.load_from_url()
        
        elif source == "csv":
            return self.load_csv()
        
        elif source == "url":
            return self.load_from_url()
        
        elif source == "database":
            return self.load_from_database()
        
        else:
            raise ValueError(f"Unknown source type: {source}")
    
    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validate loaded data has expected structure.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to validate
        
        Returns
        -------
        bool
            True if valid
        """
        required_columns = [
            'PatientId', 'AppointmentID', 'Gender',
            'ScheduledDay', 'AppointmentDay', 'Age',
            'Neighbourhood', 'No-show'
        ]
        
        try:
            validate_dataframe(df, required_columns, min_rows=100)
            self.logger.info("Data validation passed")
            return True
        except ValueError as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            raise
"""
Data Cleaning Module
====================
Functions to clean and standardize appointment data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List, Union
from datetime import datetime

from .utils import timer, validate_dataframe


class DataCleaner:
    """Handle all data cleaning operations."""
    
    def __init__(self, config: dict):
        """
        Initialize DataCleaner with configuration.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("healthcare_pipeline.DataCleaner")
        self.cleaning_report = {
            'rows_before': 0,
            'rows_after': 0,
            'columns_standardized': [],
            'dates_fixed': [],
            'outliers_fixed': 0,
            'duplicates_removed': 0,
            'missing_handled': {}
        }
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        
        Returns
        -------
        pd.DataFrame
            DataFrame with standardized column names
        """
        self.logger.info("Standardizing column names...")
        
        # Apply column mapping from config
        column_mapping = self.config['cleaning'].get('column_mapping', {})
        
        # First apply direct mappings
        df = df.rename(columns=column_mapping)
        
        # Then standardize all names
        df.columns = df.columns.str.replace('-', '_')
        df.columns = df.columns.str.replace(' ', '_')
        df.columns = df.columns.str.lower()
        
        # Track changes
        self.cleaning_report['columns_standardized'] = df.columns.tolist()
        
        self.logger.info(f"Standardized {len(df.columns)} column names")
        return df
    
    @timer
    def clean_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert date columns to datetime and fix issues.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        
        Returns
        -------
        pd.DataFrame
            DataFrame with cleaned date columns
        """
        self.logger.info("Cleaning date columns...")
        
        date_columns = self.config['cleaning'].get('date_columns', [])
        # Handle case variations
        date_columns_lower = [col.lower() for col in date_columns]
        
        dates_fixed = []
        
        for col in df.columns:
            if col.lower() in date_columns_lower or col in date_columns:
                if col in df.columns:
                    try:
                        # Convert to datetime
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        
                        # Count and handle NaT values
                        nat_count = df[col].isna().sum()
                        if nat_count > 0:
                            self.logger.warning(f"Found {nat_count} invalid dates in {col}")
                            # Forward fill or backward fill for invalid dates
                            df[col] = df[col].fillna(method='ffill')
                        
                        dates_fixed.append(col)
                        self.logger.info(f"Fixed dates in column: {col}")
                        
                    except Exception as e:
                        self.logger.error(f"Error cleaning dates in {col}: {str(e)}")
        
        self.cleaning_report['dates_fixed'] = dates_fixed
        return df
    
    def fix_age_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix negative and extreme ages.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        
        Returns
        -------
        pd.DataFrame
            DataFrame with fixed ages
        """
        self.logger.info("Fixing age outliers...")
        
        if 'age' not in df.columns:
            self.logger.warning("Age column not found")
            return df
        
        max_age = self.config['cleaning']['max_age']
        min_age = self.config['cleaning']['min_age']
        
        # Count problematic ages
        negative_ages = (df['age'] < min_age).sum()
        extreme_ages = (df['age'] > max_age).sum()
        
        # Fix negative ages (convert to positive)
        df.loc[df['age'] < min_age, 'age'] = df.loc[df['age'] < min_age, 'age'].abs()
        
        # Fix extreme ages (cap at max_age)
        median_age = df['age'].median()
        df.loc[df['age'] > max_age, 'age'] = median_age
        
        outliers_fixed = negative_ages + extreme_ages
        self.cleaning_report['outliers_fixed'] = outliers_fixed
        
        if outliers_fixed > 0:
            self.logger.info(f"Fixed {outliers_fixed} age outliers ({negative_ages} negative, {extreme_ages} extreme)")
        
        return df
    
    def fix_noshow_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert No-show encoding to binary (0/1).
        Original encoding: 'No' = showed up, 'Yes' = no-show (confusing!)
        New encoding: 0 = showed up, 1 = no-show
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        
        Returns
        -------
        pd.DataFrame
            DataFrame with fixed encoding
        """
        self.logger.info("Fixing no-show encoding...")
        
        # Find the no-show column (handle case variations)
        noshow_col = None
        for col in df.columns:
            if 'no_show' in col.lower() or 'no-show' in col.lower():
                noshow_col = col
                break
        
        if noshow_col is None:
            self.logger.warning("No-show column not found")
            return df
        
        # Create clear binary columns
        if df[noshow_col].dtype == 'object':
            # String encoding
            df['showed_up'] = df[noshow_col].map({'No': 1, 'Yes': 0})
            df['no_show'] = df[noshow_col].map({'No': 0, 'Yes': 1})
        else:
            # Already binary, just ensure correct column names
            df['no_show'] = df[noshow_col]
            df['showed_up'] = 1 - df[noshow_col]
        
        # Remove original confusing column if it's different
        if noshow_col not in ['no_show', 'showed_up']:
            df = df.drop(columns=[noshow_col])
        
        self.logger.info(f"Fixed no-show encoding: {df['no_show'].sum()} no-shows out of {len(df)} appointments")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        
        Returns
        -------
        pd.DataFrame
            DataFrame with handled missing values
        """
        self.logger.info("Handling missing values...")
        
        missing_before = df.isnull().sum()
        missing_cols = missing_before[missing_before > 0]
        
        if len(missing_cols) > 0:
            self.logger.info(f"Found missing values in {len(missing_cols)} columns")
            
            for col in missing_cols.index:
                missing_count = missing_cols[col]
                missing_pct = (missing_count / len(df)) * 100
                
                self.logger.info(f"  {col}: {missing_count} ({missing_pct:.1f}%)")
                
                # Fill missing values
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Fill numeric with median
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    self.logger.info(f"    Filled {col} with median: {median_val}")
                else:
                    # Fill categorical with mode
                    if not df[col].mode().empty:
                        mode_val = df[col].mode()[0]
                        df[col] = df[col].fillna(mode_val)
                        self.logger.info(f"    Filled {col} with mode: {mode_val}")
                    else:
                        # Fallback for empty columns or no mode
                        df[col] = df[col].fillna("Unknown")
                        self.logger.info(f"    Filled {col} with 'Unknown'")
            
            # Track missing handled
            self.cleaning_report['missing_handled'] = missing_cols.to_dict()
            
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows based on appointment ID.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        
        Returns
        -------
        pd.DataFrame
            DataFrame without duplicates
        """
        self.logger.info("Removing duplicates...")
        
        rows_before = len(df)
        
        # Find ID column
        id_col = None
        for col in df.columns:
            if 'appointmentid' in col.lower():
                id_col = col
                break
        
        if id_col:
            # Remove duplicates based on appointment ID
            df = df.drop_duplicates(subset=[id_col], keep='first')
        else:
            # Remove complete duplicates
            df = df.drop_duplicates(keep='first')
        
        rows_after = len(df)
        duplicates_removed = rows_before - rows_after
        
        self.cleaning_report['duplicates_removed'] = duplicates_removed
        
        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate rows")
        else:
            self.logger.info("No duplicates found")
        
        return df
    
    def standardize_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize text columns (trim whitespace, consistent case).
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        
        Returns
        -------
        pd.DataFrame
            DataFrame with standardized text
        """
        self.logger.info("Standardizing text columns...")
        
        text_columns = df.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            # Skip date columns that might be object type
            if 'date' not in col.lower() and 'day' not in col.lower():
                # Strip whitespace
                df[col] = df[col].astype(str).str.strip()
                
                # Standardize specific columns
                if 'gender' in col.lower():
                    df[col] = df[col].str.upper()
                elif 'neighbourhood' in col.lower():
                    df[col] = df[col].str.title()
        
        self.logger.info(f"Standardized {len(text_columns)} text columns")
        return df
    
    @timer
    def clean_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete cleaning pipeline.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw input DataFrame
        
        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame
        """
        self.logger.info("="*60)
        self.logger.info("Starting data cleaning pipeline...")
        
        # Track initial state
        self.cleaning_report['rows_before'] = len(df)
        self.cleaning_report['columns_before'] = len(df.columns)
        
        # Run cleaning steps
        df = self.clean_column_names(df)
        df = self.clean_dates(df)
        df = self.fix_age_outliers(df)
        df = self.fix_noshow_encoding(df)
        df = self.handle_missing_values(df)
        df = self.remove_duplicates(df)
        df = self.standardize_text_columns(df)
        
        # Track final state
        self.cleaning_report['rows_after'] = len(df)
        self.cleaning_report['columns_after'] = len(df.columns)
        
        # Log summary
        self.logger.info("="*60)
        self.logger.info("CLEANING SUMMARY:")
        self.logger.info(f"  Rows: {self.cleaning_report['rows_before']:,} → {self.cleaning_report['rows_after']:,}")
        self.logger.info(f"  Columns: {self.cleaning_report['columns_before']} → {self.cleaning_report['columns_after']}")
        self.logger.info(f"  Duplicates removed: {self.cleaning_report['duplicates_removed']}")
        self.logger.info(f"  Outliers fixed: {self.cleaning_report['outliers_fixed']}")
        self.logger.info("="*60)
        
        # Save cleaned data if configured
        if self.config['pipeline'].get('save_intermediate', True):
            output_path = self.config['paths']['processed_data']
            df.to_csv(output_path, index=False)
            self.logger.info(f"Saved cleaned data to {output_path}")
        
        return df
    
    def get_cleaning_report(self) -> Dict:
        """
        Get the cleaning report.
        
        Returns
        -------
        dict
            Cleaning report with statistics
        """
        return self.cleaning_report
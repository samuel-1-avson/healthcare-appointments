"""
Tests for Data Cleaning Module
==============================
Unit tests for the DataCleaner class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_cleaner import DataCleaner


# Fixtures
@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'cleaning': {
            'max_age': 100,
            'min_age': 0,
            'date_columns': ['ScheduledDay', 'AppointmentDay'],
            'column_mapping': {
                'No-show': 'No_show',
                'Hipertension': 'Hypertension'
            }
        },
        'pipeline': {
            'save_intermediate': False
        },
        'paths': {
            'processed_data': 'test_output.csv'
        }
    }


@pytest.fixture
def sample_data():
    """Sample dataframe for testing."""
    return pd.DataFrame({
        'PatientId': [1, 2, 3, 4, 5],
        'AppointmentID': ['A1', 'A2', 'A3', 'A4', 'A5'],
        'Gender': ['M', 'F', 'M', 'F', 'M'],
        'Age': [25, -30, 150, 45, 60],  # Include negative and extreme ages
        'No-show': ['No', 'Yes', 'No', 'Yes', 'No'],  # Original encoding
        'ScheduledDay': ['2016-04-29', '2016-04-29', '2016-04-30', '2016-05-01', '2016-05-01'],
        'AppointmentDay': ['2016-05-03', '2016-05-05', '2016-05-03', '2016-05-10', '2016-05-05'],
        'Hipertension': [0, 1, 0, 1, 0]
    })


class TestDataCleaner:
    """Test cases for DataCleaner class."""
    
    def test_initialization(self, sample_config):
        """Test DataCleaner initialization."""
        cleaner = DataCleaner(sample_config)
        assert cleaner.config == sample_config
        assert cleaner.cleaning_report is not None
    
    def test_clean_column_names(self, sample_config, sample_data):
        """Test column name standardization."""
        cleaner = DataCleaner(sample_config)
        df = cleaner.clean_column_names(sample_data)
        
        # Check column name transformations
        assert 'no_show' in df.columns  # No-show -> no_show
        assert 'hypertension' in df.columns  # Hipertension -> hypertension
        assert 'patientid' in df.columns  # PatientId -> patientid
        
        # Check no hyphens or spaces
        for col in df.columns:
            assert '-' not in col
            assert ' ' not in col
    
    def test_fix_age_outliers(self, sample_config, sample_data):
        """Test age outlier fixing."""
        cleaner = DataCleaner(sample_config)
        df = cleaner.clean_column_names(sample_data)  # Need lowercase columns
        df = cleaner.fix_age_outliers(df)
        
        # Check all ages are valid
        assert df['age'].min() >= 0
        assert df['age'].max() <= 100
        
        # Check negative age was fixed (made positive)
        assert 30 in df['age'].values  # -30 should become 30
        
        # Check extreme age was replaced with median
        assert 150 not in df['age'].values
    
    def test_fix_noshow_encoding(self, sample_config, sample_data):
        """Test no-show encoding fix."""
        cleaner = DataCleaner(sample_config)
        df = cleaner.clean_column_names(sample_data)
        df = cleaner.fix_noshow_encoding(df)
        
        # Check new columns exist
        assert 'no_show' in df.columns
        assert 'showed_up' in df.columns
        
        # Check encoding is correct
        # Original: 'No' = showed up, 'Yes' = no-show
        # New: no_show: 0 = showed up, 1 = no-show
        assert df['no_show'].sum() == 2  # 2 'Yes' in original data
        assert df['showed_up'].sum() == 3  # 3 'No' in original data
        
        # Check values are binary
        assert set(df['no_show'].unique()) <= {0, 1}
        assert set(df['showed_up'].unique()) <= {0, 1}
    
    def test_clean_dates(self, sample_config, sample_data):
        """Test date cleaning."""
        cleaner = DataCleaner(sample_config)
        df = cleaner.clean_column_names(sample_data)
        df = cleaner.clean_dates(df)
        
        # Check date columns are datetime
        assert pd.api.types.is_datetime64_any_dtype(df['scheduledday'])
        assert pd.api.types.is_datetime64_any_dtype(df['appointmentday'])
    
    def test_remove_duplicates(self, sample_config):
        """Test duplicate removal."""
        # Create data with duplicates
        data_with_dup = pd.DataFrame({
            'appointmentid': ['A1', 'A2', 'A2', 'A3'],  # A2 is duplicated
            'patientid': [1, 2, 2, 3],
            'age': [30, 40, 40, 50]
        })
        
        cleaner = DataCleaner(sample_config)
        df = cleaner.remove_duplicates(data_with_dup)
        
        # Check duplicate was removed
        assert len(df) == 3
        assert df['appointmentid'].nunique() == 3
    
    def test_handle_missing_values(self, sample_config):
        """Test missing value handling."""
        # Create data with missing values
        data_with_missing = pd.DataFrame({
            'age': [30, np.nan, 50, 40, np.nan],
            'gender': ['M', 'F', np.nan, 'M', 'F'],
            'score': [1, 2, 3, np.nan, 5]
        })
        
        cleaner = DataCleaner(sample_config)
        df = cleaner.handle_missing_values(data_with_missing)
        
        # Check no missing values remain
        assert df.isnull().sum().sum() == 0
        
        # Check numeric columns filled with median
        assert df.loc[1, 'age'] == data_with_missing['age'].median()
    
    def test_standardize_text_columns(self, sample_config):
        """Test text standardization."""
        # Create data with messy text
        messy_data = pd.DataFrame({
            'gender': [' M ', 'f', '  M', 'F  ', 'm'],
            'neighbourhood': ['  centro ', 'CENTRO', 'Jardim camburi', 'JARDIM CAMBURI', 'centro']
        })
        
        cleaner = DataCleaner(sample_config)
        df = cleaner.standardize_text_columns(messy_data)
        
        # Check gender standardization (should be uppercase)
        assert all(df['gender'].str.strip() == df['gender'])  # No extra whitespace
        assert all(df['gender'].isin(['M', 'F']))  # Standardized to uppercase
        
        # Check neighborhood standardization (should be title case)
        assert all(df['neighbourhood'].str.strip() == df['neighbourhood'])
    
    def test_clean_pipeline(self, sample_config, sample_data):
        """Test complete cleaning pipeline."""
        cleaner = DataCleaner(sample_config)
        df_clean = cleaner.clean_pipeline(sample_data)
        
        # Check all cleaning steps were applied
        assert 'no_show' in df_clean.columns  # Encoding fixed
        assert 'showed_up' in df_clean.columns
        assert df_clean['age'].min() >= 0  # Age outliers fixed
        assert df_clean['age'].max() <= 100
        assert pd.api.types.is_datetime64_any_dtype(df_clean['scheduledday'])  # Dates cleaned
        
        # Check report was generated
        report = cleaner.get_cleaning_report()
        assert report['rows_before'] == 5
        assert 'columns_standardized' in report
        assert 'outliers_fixed' in report


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self, sample_config):
        """Test handling of empty dataframe."""
        cleaner = DataCleaner(sample_config)
        empty_df = pd.DataFrame()
        
        # Should handle empty dataframe gracefully
        result = cleaner.clean_pipeline(empty_df)
        assert len(result) == 0
    
    def test_missing_columns(self, sample_config):
        """Test handling when expected columns are missing."""
        cleaner = DataCleaner(sample_config)
        minimal_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        # Should handle missing columns gracefully
        result = cleaner.clean_pipeline(minimal_df)
        assert len(result) == 3  # Should still process what it can
    
    def test_all_missing_values(self, sample_config):
        """Test handling column with all missing values."""
        cleaner = DataCleaner(sample_config)
        all_missing_df = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan],
            'col2': [1, 2, 3]
        })
        
        result = cleaner.handle_missing_values(all_missing_df)
        # Should handle all missing gracefully
        assert len(result) == 3


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Tests for Feature Engineering Module
====================================
Unit tests for the FeatureEngineer class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_engineer import FeatureEngineer


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'features': {
            'age_bins': [0, 12, 18, 35, 50, 65, 100],
            'age_labels': ['Child', 'Teen', 'Young Adult', 'Adult', 'Middle Age', 'Senior'],
            'lead_time_bins': [-1, 0, 7, 14, 30, 365],
            'lead_time_labels': ['Same Day', '1-7 days', '8-14 days', '15-30 days', '30+ days']
        },
        'pipeline': {
            'save_intermediate': False
        },
        'paths': {
            'features_data': 'test_features.csv'
        }
    }


@pytest.fixture
def sample_data_with_dates():
    """Sample dataframe with proper date columns."""
    base_date = datetime(2016, 5, 1)
    
    data = {
        'patientid': [1, 1, 2, 2, 3, 3, 4, 5],
        'appointmentid': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'],
        'scheduledday': [
            base_date - timedelta(days=10),
            base_date - timedelta(days=5),
            base_date - timedelta(days=15),
            base_date - timedelta(days=0),  # Same day
            base_date - timedelta(days=7),
            base_date - timedelta(days=30),
            base_date - timedelta(days=2),
            base_date - timedelta(days=20)
        ],
        'appointmentday': [base_date] * 8,
        'age': [5, 5, 16, 16, 25, 25, 45, 70],
        'no_show': [0, 1, 0, 0, 1, 1, 0, 0],
        'scholarship': [0, 0, 1, 1, 0, 0, 1, 0],
        'hypertension': [0, 0, 0, 0, 1, 1, 1, 1],
        'diabetes': [0, 0, 0, 0, 0, 0, 1, 1],
        'sms_received': [0, 1, 1, 0, 1, 1, 0, 1],
        'neighbourhood': ['Centro', 'Centro', 'Jardim', 'Jardim', 
                         'Centro', 'Centro', 'Praia', 'Praia']
    }
    
    return pd.DataFrame(data)


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""
    
    def test_initialization(self, sample_config):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer(sample_config)
        assert engineer.config == sample_config
        assert engineer.features_created == []
    
    def test_create_lead_time(self, sample_config, sample_data_with_dates):
        """Test lead time calculation."""
        engineer = FeatureEngineer(sample_config)
        df = engineer.create_lead_time(sample_data_with_dates)
        
        assert 'lead_days' in df.columns
        assert df['lead_days'].min() >= 0  # No negative lead days
        assert df.loc[3, 'lead_days'] == 0  # Same day appointment
        assert df.loc[0, 'lead_days'] == 10  # 10 days lead
    
    def test_create_age_groups(self, sample_config, sample_data_with_dates):
        """Test age group creation."""
        engineer = FeatureEngineer(sample_config)
        df = engineer.create_age_groups(sample_data_with_dates)
        
        assert 'age_group' in df.columns
        assert df.loc[0, 'age_group'] == 'Child'  # Age 5
        assert df.loc[2, 'age_group'] == 'Teen'  # Age 16
        assert df.loc[4, 'age_group'] == 'Young Adult'  # Age 25
        assert df.loc[6, 'age_group'] == 'Adult'  # Age 45
        assert df.loc[7, 'age_group'] == 'Senior'  # Age 70
    
    def test_create_lead_time_category(self, sample_config, sample_data_with_dates):
        """Test lead time categorization."""
        engineer = FeatureEngineer(sample_config)
        df = engineer.create_lead_time(sample_data_with_dates)
        df = engineer.create_lead_time_category(df)
        
        assert 'lead_time_category' in df.columns
        assert df.loc[3, 'lead_time_category'] == 'Same Day'
        assert df.loc[0, 'lead_time_category'] == '8-14 days'
    
    def test_create_time_features(self, sample_config, sample_data_with_dates):
        """Test time-based feature creation."""
        engineer = FeatureEngineer(sample_config)
        df = engineer.create_time_features(sample_data_with_dates)
        
        time_features = [
            'appointment_weekday', 'appointment_month', 'appointment_week',
            'is_monday', 'is_friday', 'is_weekend'
        ]
        
        for feature in time_features:
            assert feature in df.columns
        
        # All appointments are on same day (May 1, 2016 - Sunday)
        assert df['appointment_weekday'].iloc[0] == 'Sunday'
        assert df['is_weekend'].iloc[0] == 1
        assert df['is_monday'].iloc[0] == 0
    
    def test_create_patient_history(self, sample_config, sample_data_with_dates):
        """Test patient history feature creation."""
        engineer = FeatureEngineer(sample_config)
        df = engineer.create_patient_history(sample_data_with_dates)
        
        assert 'patient_total_appointments' in df.columns
        assert 'patient_previous_noshows' in df.columns
        assert 'patient_historical_noshow_rate' in df.columns
        assert 'is_first_appointment' in df.columns
        
        # Patient 1 has 2 appointments
        patient1_data = df[df['patientid'] == 1]
        assert patient1_data['patient_total_appointments'].max() == 2
        assert patient1_data['is_first_appointment'].iloc[0] == 1
        assert patient1_data['is_first_appointment'].iloc[1] == 0
    
    def test_create_health_features(self, sample_config, sample_data_with_dates):
        """Test health-related feature creation."""
        engineer = FeatureEngineer(sample_config)
        df = engineer.create_health_features(sample_data_with_dates)
        
        assert 'total_conditions' in df.columns
        assert 'has_chronic_condition' in df.columns
        
        # Check calculations
        assert df.loc[0, 'total_conditions'] == 0  # No conditions
        assert df.loc[4, 'total_conditions'] == 1  # Hypertension only
        assert df.loc[6, 'total_conditions'] == 2  # Hypertension + Diabetes
        
        assert df.loc[0, 'has_chronic_condition'] == 0
        assert df.loc[4, 'has_chronic_condition'] == 1
    
    def test_create_socioeconomic_features(self, sample_config, sample_data_with_dates):
        """Test socioeconomic feature creation."""
        engineer = FeatureEngineer(sample_config)
        df = engineer.create_socioeconomic_features(sample_data_with_dates)
        
        if 'scholarship' in df.columns:
            assert 'low_income_indicator' in df.columns
            assert df['low_income_indicator'].equals(df['scholarship'])
        
        # Should create neighborhood statistics
        assert 'neighborhood_noshow_rate' in df.columns
        assert 'neighborhood_risk' in df.columns
    
    def test_create_interaction_features(self, sample_config, sample_data_with_dates):
        """Test interaction feature creation."""
        engineer = FeatureEngineer(sample_config)
        
        # Need to create base features first
        df = engineer.create_lead_time(sample_data_with_dates)
        df = engineer.create_age_groups(df)
        df = engineer.create_lead_time_category(df)
        df = engineer.create_time_features(df)
        df = engineer.create_patient_history(df)
        df = engineer.create_interaction_features(df)
        
        # Check some interaction features exist
        interaction_cols = [col for col in df.columns if 
                          'young_long_lead' in col or 
                          'first_young' in col]
        assert len(interaction_cols) > 0
    
    def test_engineer_all_features(self, sample_config, sample_data_with_dates):
        """Test complete feature engineering pipeline."""
        engineer = FeatureEngineer(sample_config)
        df_features = engineer.engineer_all_features(sample_data_with_dates)
        
        # Check that new features were created
        original_cols = len(sample_data_with_dates.columns)
        new_cols = len(df_features.columns)
        assert new_cols > original_cols
        
        # Check feature list is populated
        assert len(engineer.get_features_created()) > 0
        
        # Check key features exist
        expected_features = [
            'lead_days', 'age_group', 'lead_time_category',
            'has_chronic_condition', 'patient_total_appointments'
        ]
        for feature in expected_features:
            assert feature in df_features.columns


class TestEdgeCasesFeatureEngineer:
    """Test edge cases for feature engineering."""
    
    def test_missing_columns(self, sample_config):
        """Test handling of missing expected columns."""
        engineer = FeatureEngineer(sample_config)
        
        # DataFrame with minimal columns
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        # Should handle gracefully
        df_result = engineer.engineer_all_features(df)
        assert len(df_result) == 3
    
    def test_single_row_dataframe(self, sample_config):
        """Test with single row DataFrame."""
        engineer = FeatureEngineer(sample_config)
        
        df = pd.DataFrame({
            'age': [30],
            'scheduledday': [datetime(2016, 4, 20)],
            'appointmentday': [datetime(2016, 5, 1)]
        })
        
        df_result = engineer.engineer_all_features(df)
        assert len(df_result) == 1
        assert 'lead_days' in df_result.columns
        assert df_result['lead_days'].iloc[0] == 11


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
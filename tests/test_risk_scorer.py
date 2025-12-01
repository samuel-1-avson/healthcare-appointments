"""
Tests for Risk Scoring Module
=============================
Unit tests for the RiskScorer class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.risk_scorer import RiskScorer, InterventionProtocol


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'features': {
            'risk_weights': {
                'patient_history': 3.0,
                'lead_time': 2.5,
                'day_of_week': 1.5,
                'age': 2.0,
                'sms': 1.0,
                'health': 1.5,
                'socioeconomic': 1.5,
                'neighborhood': 2.0
            },
            'risk_thresholds': {
                'critical': 3.5,
                'high': 3.0,
                'medium': 2.5,
                'low': 2.0
            }
        }
    }


@pytest.fixture
def sample_data_for_scoring():
    """Sample dataframe with features for risk scoring."""
    return pd.DataFrame({
        'patient_historical_noshow_rate': [0.0, 0.2, 0.5, 0.8, 0.1],
        'patient_previous_appointments': [5, 3, 2, 4, 10],
        'is_first_appointment': [0, 0, 0, 0, 1],
        'lead_days': [0, 7, 15, 30, 3],
        'appointment_weekday': ['Saturday', 'Monday', 'Tuesday', 'Friday', 'Wednesday'],
        'age_group': ['Senior', 'Young Adult', 'Young Adult', 'Teen', 'Adult'],
        'scholarship': [0, 1, 0, 1, 0],
        'has_chronic_condition': [1, 0, 0, 0, 1],
        'neighborhood_noshow_rate': [0.15, 0.25, 0.30, 0.20, 0.18],
        'sms_received': [0, 1, 1, 0, 1],
        'no_show': [0, 1, 1, 1, 0]
    })


class TestRiskScorer:
    """Test cases for RiskScorer class."""
    
    def test_initialization(self, sample_config):
        """Test RiskScorer initialization."""
        scorer = RiskScorer(sample_config)
        assert scorer.config == sample_config
        assert scorer.risk_weights == sample_config['features']['risk_weights']
        assert scorer.risk_thresholds == sample_config['features']['risk_thresholds']
        assert len(scorer.protocols) == 5  # 5 risk tiers
    
    def test_intervention_protocol_dataclass(self):
        """Test InterventionProtocol dataclass."""
        protocol = InterventionProtocol(
            tier='TEST',
            risk_level='Test Level',
            sms_reminders=2,
            phone_call=True,
            overbook_percentage=0.15,
            priority_scheduling=True,
            deposit_required=False,
            mobile_clinic=False,
            description='Test protocol'
        )
        
        assert protocol.tier == 'TEST'
        assert protocol.sms_reminders == 2
        assert protocol.phone_call is True
        assert protocol.overbook_percentage == 0.15
    
    def test_calculate_patient_risk(self, sample_config, sample_data_for_scoring):
        """Test patient risk calculation."""
        scorer = RiskScorer(sample_config)
        df = scorer.calculate_patient_risk(sample_data_for_scoring)
        
        assert 'patient_risk_score' in df.columns
        
        # Check scoring logic
        # First patient: 0% historical rate = minimal risk (1.0)
        assert df.loc[0, 'patient_risk_score'] == 1.0
        
        # Third patient: 50% historical rate = very high risk (5.0)
        assert df.loc[2, 'patient_risk_score'] == 5.0
        
        # Fifth patient: first appointment = unknown risk (2.5)
        assert df.loc[4, 'patient_risk_score'] == 2.5
    
    def test_calculate_time_risk(self, sample_config, sample_data_for_scoring):
        """Test time-based risk calculation."""
        scorer = RiskScorer(sample_config)
        df = scorer.calculate_time_risk(sample_data_for_scoring)
        
        assert 'time_risk_score' in df.columns
        assert 'lead_time_risk' in df.columns
        assert 'day_risk' in df.columns
        
        # Check lead time scoring
        # First patient: 0 lead days = excellent (1.0)
        assert df.loc[0, 'lead_time_risk'] == 1.0
        
        # Fourth patient: 30 lead days = very high (5.0)
        assert df.loc[3, 'lead_time_risk'] == 5.0
        
        # Check day scoring
        # First patient: Saturday = best (1.0)
        assert df.loc[0, 'day_risk'] == 1.0
        
        # Second patient: Monday = worst (4.0)
        assert df.loc[1, 'day_risk'] == 4.0
    
    def test_calculate_demographic_risk(self, sample_config, sample_data_for_scoring):
        """Test demographic risk calculation."""
        scorer = RiskScorer(sample_config)
        df = scorer.calculate_demographic_risk(sample_data_for_scoring)
        
        assert 'demographic_risk_score' in df.columns
        assert 'age_risk' in df.columns
        assert 'socio_risk' in df.columns
        
        # Check age risk
        # First patient: Senior = lowest risk (1.5)
        assert df.loc[0, 'age_risk'] == 1.5
        
        # Second patient: Young Adult = highest risk (4.0)
        assert df.loc[1, 'age_risk'] == 4.0
        
        # Check socioeconomic risk
        # Second patient: has scholarship = higher risk (3.5)
        assert df.loc[1, 'socio_risk'] == 3.5
    
    def test_calculate_health_risk(self, sample_config, sample_data_for_scoring):
        """Test health-based risk calculation."""
        scorer = RiskScorer(sample_config)
        df = scorer.calculate_health_risk(sample_data_for_scoring)
        
        assert 'health_risk_score' in df.columns
        
        # Check chronic condition effect
        # First patient: has chronic = lower risk (1.5)
        assert df.loc[0, 'health_risk_score'] == 1.5
        
        # Second patient: no chronic = higher risk (2.5)
        assert df.loc[1, 'health_risk_score'] == 2.5
    
    def test_calculate_neighborhood_risk(self, sample_config, sample_data_for_scoring):
        """Test neighborhood risk calculation."""
        scorer = RiskScorer(sample_config)
        df = scorer.calculate_neighborhood_risk(sample_data_for_scoring)
        
        assert 'neighborhood_risk_score' in df.columns
        
        # Check neighborhood scoring based on rates
        # First patient: 15% rate = low risk (1.0)
        assert df.loc[0, 'neighborhood_risk_score'] == 1.0
        
        # Third patient: 30% rate = very high risk (5.0)
        assert df.loc[2, 'neighborhood_risk_score'] == 5.0
    
    def test_calculate_composite_risk(self, sample_config, sample_data_for_scoring):
        """Test composite risk calculation."""
        scorer = RiskScorer(sample_config)
        df = scorer.calculate_composite_risk(sample_data_for_scoring)
        
        assert 'composite_risk_score' in df.columns
        assert 'risk_percentile' in df.columns
        
        # Check score is normalized (0-5 range)
        assert df['composite_risk_score'].min() >= 0
        assert df['composite_risk_score'].max() <= 5
        
        # Check percentile calculation
        assert df['risk_percentile'].min() >= 0
        assert df['risk_percentile'].max() <= 100
    
    def test_assign_risk_tier(self, sample_config, sample_data_for_scoring):
        """Test risk tier assignment."""
        scorer = RiskScorer(sample_config)
        df = scorer.calculate_composite_risk(sample_data_for_scoring)
        df = scorer.assign_risk_tier(df)
        
        assert 'risk_tier' in df.columns
        assert 'risk_tier_display' in df.columns
        
        # Check all tiers are valid
        valid_tiers = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']
        assert all(tier in valid_tiers for tier in df['risk_tier'].unique())
        
        # Check display format
        assert all('🔴' in display or '🟠' in display or '🟡' in display or 
                  '🟢' in display or '⭐' in display 
                  for display in df['risk_tier_display'].values)
    
    def test_get_intervention_protocol(self, sample_config):
        """Test getting intervention protocol."""
        scorer = RiskScorer(sample_config)
        
        # Test each tier
        critical_protocol = scorer.get_intervention_protocol('CRITICAL')
        assert critical_protocol.phone_call is True
        assert critical_protocol.deposit_required is True
        assert critical_protocol.overbook_percentage == 0.20
        
        minimal_protocol = scorer.get_intervention_protocol('MINIMAL')
        assert minimal_protocol.phone_call is False
        assert minimal_protocol.deposit_required is False
        assert minimal_protocol.priority_scheduling is True
    
    def test_add_intervention_recommendations(self, sample_config, sample_data_for_scoring):
        """Test adding intervention recommendations."""
        scorer = RiskScorer(sample_config)
        df = scorer.calculate_composite_risk(sample_data_for_scoring)
        df = scorer.assign_risk_tier(df)
        df = scorer.add_intervention_recommendations(df)
        
        intervention_columns = [
            'sms_reminders_needed',
            'phone_call_required',
            'overbook_percentage',
            'deposit_required',
            'intervention_description'
        ]
        
        for col in intervention_columns:
            assert col in df.columns
        
        # Check values are appropriate types
        assert df['sms_reminders_needed'].dtype in [int, float]
        assert df['phone_call_required'].dtype == bool
        assert df['overbook_percentage'].dtype == float
    
    def test_score_pipeline(self, sample_config, sample_data_for_scoring):
        """Test complete scoring pipeline."""
        scorer = RiskScorer(sample_config)
        df = scorer.score_pipeline(sample_data_for_scoring)
        
        # Check all expected columns are present
        expected_columns = [
            'composite_risk_score',
            'risk_tier',
            'risk_tier_display',
            'sms_reminders_needed',
            'phone_call_required',
            'intervention_description'
        ]
        
        for col in expected_columns:
            assert col in df.columns
        
        # Check risk summary
        summary = scorer.get_risk_summary(df)
        assert 'total_appointments' in summary
        assert 'risk_score_stats' in summary
        assert 'tier_distribution' in summary


class TestRiskScorerEdgeCases:
    """Test edge cases for risk scorer."""
    
    def test_empty_dataframe(self, sample_config):
        """Test with empty dataframe."""
        scorer = RiskScorer(sample_config)
        empty_df = pd.DataFrame()
        
        # Should handle empty dataframe
        result = scorer.score_pipeline(empty_df)
        assert len(result) == 0
    
    def test_missing_features(self, sample_config):
        """Test with missing expected features."""
        scorer = RiskScorer(sample_config)
        minimal_df = pd.DataFrame({
            'appointment_id': [1, 2, 3],
            'no_show': [0, 1, 0]
        })
        
        # Should still calculate what it can
        result = scorer.score_pipeline(minimal_df)
        assert 'composite_risk_score' in result.columns
        assert 'risk_tier' in result.columns
    
    def test_nan_values(self, sample_config):
        """Test handling of NaN values."""
        scorer = RiskScorer(sample_config)
        df_with_nan = pd.DataFrame({
            'lead_days': [5, np.nan, 10],
            'age_group': ['Adult', np.nan, 'Senior'],
            'no_show': [0, 1, 0]
        })
        
        # Should handle NaN values gracefully
        result = scorer.calculate_composite_risk(df_with_nan)
        assert not result['composite_risk_score'].isna().all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
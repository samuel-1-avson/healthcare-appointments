"""
Schema Validation Tests
=======================
Tests for Pydantic schema validation.
"""

import pytest
from pydantic import ValidationError
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.schemas import (
    AppointmentFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    RiskAssessment,
    InterventionRecommendation,
    create_risk_assessment,
    create_intervention
)


class TestAppointmentFeatures:
    """Tests for AppointmentFeatures schema."""
    
    def test_valid_appointment(self):
        """Test creating valid appointment."""
        appointment = AppointmentFeatures(
            age=35,
            gender="F",
            lead_days=7,
            sms_received=1
        )
        
        assert appointment.age == 35
        assert appointment.gender == "F"
        assert appointment.lead_days == 7
        assert appointment.sms_received == 1
    
    def test_default_values(self):
        """Test that default values are applied."""
        appointment = AppointmentFeatures(
            age=35,
            gender="M",
            lead_days=7
        )
        
        assert appointment.scholarship == 0
        assert appointment.hypertension == 0
        assert appointment.diabetes == 0
        assert appointment.alcoholism == 0
        assert appointment.handicap == 0
        assert appointment.sms_received == 0
    
    def test_age_validation(self):
        """Test age validation bounds."""
        # Valid ages
        AppointmentFeatures(age=0, gender="M", lead_days=1)
        AppointmentFeatures(age=120, gender="M", lead_days=1)
        
        # Invalid ages
        with pytest.raises(ValidationError):
            AppointmentFeatures(age=-1, gender="M", lead_days=1)
        
        with pytest.raises(ValidationError):
            AppointmentFeatures(age=121, gender="M", lead_days=1)
    
    def test_gender_validation(self):
        """Test gender validation."""
        # Valid genders
        AppointmentFeatures(age=35, gender="M", lead_days=1)
        AppointmentFeatures(age=35, gender="F", lead_days=1)
        
        # Case insensitive
        appt = AppointmentFeatures(age=35, gender="m", lead_days=1)
        assert appt.gender == "M"
    
    def test_binary_field_validation(self):
        """Test binary fields only accept 0 or 1."""
        # Valid
        AppointmentFeatures(age=35, gender="F", lead_days=1, sms_received=0)
        AppointmentFeatures(age=35, gender="F", lead_days=1, sms_received=1)
        
        # Invalid
        with pytest.raises(ValidationError):
            AppointmentFeatures(age=35, gender="F", lead_days=1, sms_received=2)
    
    def test_handicap_validation(self):
        """Test handicap field validation (0-4)."""
        # Valid values
        for h in range(5):
            AppointmentFeatures(age=35, gender="F", lead_days=1, handicap=h)
        
        # Invalid
        with pytest.raises(ValidationError):
            AppointmentFeatures(age=35, gender="F", lead_days=1, handicap=5)
    
    def test_optional_fields(self):
        """Test optional fields."""
        appointment = AppointmentFeatures(
            age=35,
            gender="F",
            lead_days=7,
            neighbourhood="JARDIM CAMBURI",
            appointment_weekday="Monday",
            patient_historical_noshow_rate=0.15
        )
        
        assert appointment.neighbourhood == "JARDIM CAMBURI"
        assert appointment.appointment_weekday == "Monday"
        assert appointment.patient_historical_noshow_rate == 0.15


class TestRiskAssessment:
    """Tests for risk assessment creation."""
    
    def test_create_minimal_risk(self):
        """Test creating minimal risk assessment."""
        risk = create_risk_assessment(0.1)
        
        assert risk.tier == "MINIMAL"
        assert risk.probability == 0.1
        assert risk.color is not None
        assert risk.emoji is not None
    
    def test_create_low_risk(self):
        """Test creating low risk assessment."""
        risk = create_risk_assessment(0.2)
        
        assert risk.tier == "LOW"
    
    def test_create_medium_risk(self):
        """Test creating medium risk assessment."""
        risk = create_risk_assessment(0.4)
        
        assert risk.tier == "MEDIUM"
    
    def test_create_high_risk(self):
        """Test creating high risk assessment."""
        risk = create_risk_assessment(0.6)
        
        assert risk.tier == "HIGH"
    
    def test_create_critical_risk(self):
        """Test creating critical risk assessment."""
        risk = create_risk_assessment(0.8)
        
        assert risk.tier == "CRITICAL"
    
    def test_risk_probability_bounds(self):
        """Test risk assessment at probability boundaries."""
        # Exact boundaries
        assert create_risk_assessment(0.0).tier == "MINIMAL"
        assert create_risk_assessment(0.15).tier == "LOW"
        assert create_risk_assessment(0.3).tier == "MEDIUM"
        assert create_risk_assessment(0.5).tier == "HIGH"
        assert create_risk_assessment(0.7).tier == "CRITICAL"
        assert create_risk_assessment(1.0).tier == "CRITICAL"


class TestInterventionRecommendation:
    """Tests for intervention recommendation creation."""
    
    def test_minimal_intervention(self):
        """Test minimal risk intervention."""
        intervention = create_intervention(0.1, "MINIMAL")
        
        assert intervention.sms_reminders == 1
        assert intervention.phone_call is False
    
    def test_critical_intervention(self):
        """Test critical risk intervention."""
        intervention = create_intervention(0.8, "CRITICAL")
        
        assert intervention.sms_reminders >= 2
        assert intervention.phone_call is True
    
    def test_intervention_has_action(self):
        """Test all interventions have action text."""
        for tier in ["MINIMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            intervention = create_intervention(0.5, tier)
            assert intervention.action is not None
            assert len(intervention.action) > 0


class TestBatchRequest:
    """Tests for batch request schema."""
    
    def test_valid_batch_request(self):
        """Test valid batch request."""
        request = BatchPredictionRequest(
            appointments=[
                AppointmentFeatures(age=35, gender="F", lead_days=7),
                AppointmentFeatures(age=50, gender="M", lead_days=14)
            ]
        )
        
        assert len(request.appointments) == 2
    
    def test_empty_batch_rejected(self):
        """Test empty appointments list is rejected."""
        with pytest.raises(ValidationError):
            BatchPredictionRequest(appointments=[])
    
    def test_batch_with_options(self):
        """Test batch request with options."""
        request = BatchPredictionRequest(
            appointments=[
                AppointmentFeatures(age=35, gender="F", lead_days=7)
            ],
            include_explanations=True,
            threshold=0.3
        )
        
        assert request.include_explanations is True
        assert request.threshold == 0.3
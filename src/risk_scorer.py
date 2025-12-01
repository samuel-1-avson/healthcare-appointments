"""
Risk Scoring Module
===================
Calculate appointment and patient risk scores for no-show prediction.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass

from .utils import timer


@dataclass
class InterventionProtocol:
    """Data class for intervention protocols."""
    tier: str
    risk_level: str
    sms_reminders: int
    phone_call: bool
    overbook_percentage: float
    priority_scheduling: bool
    deposit_required: bool
    mobile_clinic: bool
    description: str


class RiskScorer:
    """Calculate risk scores and recommend interventions."""
    
    def __init__(self, config: dict):
        """
        Initialize RiskScorer with configuration.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("healthcare_pipeline.RiskScorer")
        self.risk_weights = config['features']['risk_weights']
        self.risk_thresholds = config['features']['risk_thresholds']
        self.risk_factors_calculated = []
        
        # Define intervention protocols
        self._setup_intervention_protocols()
    
    def _setup_intervention_protocols(self):
        """Setup intervention protocols for each risk tier."""
        self.protocols = {
            'MINIMAL': InterventionProtocol(
                tier='⭐ MINIMAL',
                risk_level='Very Low',
                sms_reminders=1,
                phone_call=False,
                overbook_percentage=0.0,
                priority_scheduling=True,
                deposit_required=False,
                mobile_clinic=False,
                description='Standard process with priority booking privileges'
            ),
            'LOW': InterventionProtocol(
                tier='🟢 LOW',
                risk_level='Low',
                sms_reminders=1,
                phone_call=False,
                overbook_percentage=0.05,
                priority_scheduling=False,
                deposit_required=False,
                mobile_clinic=False,
                description='Standard process'
            ),
            'MEDIUM': InterventionProtocol(
                tier='🟡 MEDIUM',
                risk_level='Medium',
                sms_reminders=2,
                phone_call=False,
                overbook_percentage=0.10,
                priority_scheduling=False,
                deposit_required=False,
                mobile_clinic=False,
                description='Enhanced reminders and slight overbooking'
            ),
            'HIGH': InterventionProtocol(
                tier='🟠 HIGH',
                risk_level='High',
                sms_reminders=2,
                phone_call=True,
                overbook_percentage=0.15,
                priority_scheduling=True,
                deposit_required=False,
                mobile_clinic=True,
                description='Multiple reminders, phone confirmation, overbooking'
            ),
            'CRITICAL': InterventionProtocol(
                tier='🔴 CRITICAL',
                risk_level='Critical',
                sms_reminders=3,
                phone_call=True,
                overbook_percentage=0.20,
                priority_scheduling=True,
                deposit_required=True,
                mobile_clinic=True,
                description='Maximum intervention: deposit, calls, mobile clinic priority'
            )
        }
    
    def calculate_patient_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk score based on patient history.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with patient history features
        
        Returns
        -------
        pd.DataFrame
            DataFrame with patient risk score
        """
        self.logger.info("Calculating patient risk scores...")
        
        # Initialize risk score
        df['patient_risk_score'] = 0
        
        # Risk from patient history
        if 'patient_historical_noshow_rate' in df.columns:
            df['patient_risk_score'] = df.apply(
                lambda row: self._score_patient_history(row), axis=1
            )
            self.risk_factors_calculated.append('patient_history')
        
        # Risk from being a new patient
        if 'is_first_appointment' in df.columns:
            df.loc[df['is_first_appointment'] == 1, 'patient_risk_score'] = 2.5
            self.risk_factors_calculated.append('new_patient')
        
        self.logger.info(f"Patient risk scores: min={df['patient_risk_score'].min():.2f}, "
                        f"max={df['patient_risk_score'].max():.2f}, "
                        f"mean={df['patient_risk_score'].mean():.2f}")
        
        return df
    
    def _score_patient_history(self, row) -> float:
        """Score based on historical no-show rate."""
        if row['patient_previous_appointments'] == 0:
            return 2.5  # Unknown risk for first appointment
        
        rate = row['patient_historical_noshow_rate']
        if rate >= 0.5:
            return 5.0  # Very high risk
        elif rate >= 0.3:
            return 4.0  # High risk
        elif rate >= 0.15:
            return 3.0  # Medium risk
        elif rate >= 0.05:
            return 2.0  # Low risk
        else:
            return 1.0  # Minimal risk
    
    def calculate_time_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk score based on scheduling factors.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with time features
        
        Returns
        -------
        pd.DataFrame
            DataFrame with time risk score
        """
        self.logger.info("Calculating time-based risk scores...")
        
        # Lead time risk
        df['lead_time_risk'] = 1.0
        if 'lead_days' in df.columns:
            df['lead_time_risk'] = df['lead_days'].apply(self._score_lead_time)
            self.risk_factors_calculated.append('lead_time')
        
        # Day of week risk
        df['day_risk'] = 2.5
        if 'appointment_weekday' in df.columns:
            df['day_risk'] = df['appointment_weekday'].apply(self._score_day_of_week)
            self.risk_factors_calculated.append('day_of_week')
        
        # Combine time risks
        df['time_risk_score'] = (df['lead_time_risk'] + df['day_risk']) / 2
        
        self.logger.info(f"Time risk scores: min={df['time_risk_score'].min():.2f}, "
                        f"max={df['time_risk_score'].max():.2f}, "
                        f"mean={df['time_risk_score'].mean():.2f}")
        
        return df
    
    def _score_lead_time(self, lead_days: float) -> float:
        """Score based on lead time."""
        if lead_days <= 0:
            return 1.0  # Same day: excellent
        elif lead_days <= 3:
            return 2.0  # 1-3 days: good
        elif lead_days <= 7:
            return 3.0  # 4-7 days: medium
        elif lead_days <= 14:
            return 4.0  # 8-14 days: high
        else:
            return 5.0  # 15+ days: very high
    
    def _score_day_of_week(self, day: str) -> float:
        """Score based on day of week."""
        day_scores = {
            'Saturday': 1.0,    # Best
            'Wednesday': 2.0,   # Good
            'Thursday': 2.0,    # Good
            'Friday': 3.0,      # Medium
            'Tuesday': 3.5,     # Medium-High
            'Monday': 4.0,      # High
            'Sunday': 2.5       # (if any)
        }
        return day_scores.get(day, 2.5)
    
    def calculate_demographic_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk score based on demographics.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with demographic features
        
        Returns
        -------
        pd.DataFrame
            DataFrame with demographic risk score
        """
        self.logger.info("Calculating demographic risk scores...")
        
        # Age risk
        df['age_risk'] = 2.5
        if 'age_group' in df.columns:
            age_scores = {
                'Child': 2.0,
                'Teen': 3.0,
                'Young Adult': 4.0,  # Highest risk
                'Adult': 2.5,
                'Middle Age': 2.0,
                'Senior': 1.5  # Lowest risk
            }
            df['age_risk'] = df['age_group'].map(age_scores).fillna(2.5)
            self.risk_factors_calculated.append('age')
        
        # Gender risk (minimal difference in our data)
        df['gender_risk'] = 2.5
        if 'gender' in df.columns:
            df['gender_risk'] = df['gender'].apply(lambda x: 2.5)  # No significant difference
        
        # Socioeconomic risk
        df['socio_risk'] = 2.5
        if 'scholarship' in df.columns:
            df['socio_risk'] = df['scholarship'].apply(lambda x: 3.5 if x == 1 else 2.0)
            self.risk_factors_calculated.append('socioeconomic')
        
        # Combine demographic risks
        df['demographic_risk_score'] = (
            df['age_risk'] * 0.6 + 
            df['socio_risk'] * 0.4
        )
        
        self.logger.info(f"Demographic risk scores: min={df['demographic_risk_score'].min():.2f}, "
                        f"max={df['demographic_risk_score'].max():.2f}, "
                        f"mean={df['demographic_risk_score'].mean():.2f}")
        
        return df
    
    def calculate_health_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk score based on health conditions.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with health features
        
        Returns
        -------
        pd.DataFrame
            DataFrame with health risk score
        """
        self.logger.info("Calculating health-based risk scores...")
        
        df['health_risk_score'] = 2.5
        
        # Chronic conditions actually reduce no-show risk
        if 'has_chronic_condition' in df.columns:
            df['health_risk_score'] = df['has_chronic_condition'].apply(
                lambda x: 1.5 if x == 1 else 2.5
            )
            self.risk_factors_calculated.append('health_conditions')
        
        # Disability might affect attendance
        if 'has_disability' in df.columns:
            df.loc[df['has_disability'] == 1, 'health_risk_score'] += 0.5
        
        self.logger.info(f"Health risk scores: min={df['health_risk_score'].min():.2f}, "
                        f"max={df['health_risk_score'].max():.2f}, "
                        f"mean={df['health_risk_score'].mean():.2f}")
        
        return df
    
    def calculate_neighborhood_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk score based on neighborhood.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with neighborhood features
        
        Returns
        -------
        pd.DataFrame
            DataFrame with neighborhood risk score
        """
        self.logger.info("Calculating neighborhood-based risk scores...")
        
        df['neighborhood_risk_score'] = 2.5
        
        if 'neighborhood_noshow_rate' in df.columns:
            df['neighborhood_risk_score'] = df['neighborhood_noshow_rate'].apply(
                lambda x: 5.0 if x > 0.25 else
                         4.0 if x > 0.22 else
                         3.0 if x > 0.20 else
                         2.0 if x > 0.18 else 1.0
            )
            self.risk_factors_calculated.append('neighborhood')
        
        self.logger.info(f"Neighborhood risk scores: min={df['neighborhood_risk_score'].min():.2f}, "
                        f"max={df['neighborhood_risk_score'].max():.2f}, "
                        f"mean={df['neighborhood_risk_score'].mean():.2f}")
        
        return df
    
    def calculate_sms_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk adjustment based on SMS status.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with SMS features
        
        Returns
        -------
        pd.DataFrame
            DataFrame with SMS risk score
        """
        self.logger.info("Calculating SMS-based risk scores...")
        
        df['sms_risk_score'] = 2.5
        
        if 'sms_received' in df.columns:
            # SMS paradox: currently correlates with higher no-show
            # This is likely selection bias (SMS sent to high-risk patients)
            if 'lead_days' in df.columns:
                df['sms_risk_score'] = df.apply(
                    lambda row: 3.0 if row['sms_received'] == 1 and row['lead_days'] > 7
                    else 4.0 if row['sms_received'] == 0 and row['lead_days'] > 7
                    else 2.0,
                    axis=1
                )
            else:
                df['sms_risk_score'] = df['sms_received'].apply(
                    lambda x: 3.0 if x == 1 else 2.5
                )
            self.risk_factors_calculated.append('sms_status')
        
        return df
    
    @timer
    def calculate_composite_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate weighted composite risk score.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with all features
        
        Returns
        -------
        pd.DataFrame
            DataFrame with composite risk score
        """
        self.logger.info("="*60)
        self.logger.info("Calculating composite risk scores...")
        
        # Calculate all component risks
        df = self.calculate_patient_risk(df)
        df = self.calculate_time_risk(df)
        df = self.calculate_demographic_risk(df)
        df = self.calculate_health_risk(df)
        df = self.calculate_neighborhood_risk(df)
        df = self.calculate_sms_risk(df)
        
        # Get weights from config
        weights = self.risk_weights
        
        # Calculate weighted composite score
        df['composite_risk_score'] = (
            df.get('patient_risk_score', 2.5) * weights.get('patient_history', 1.0) +
            df.get('time_risk_score', 2.5) * weights.get('lead_time', 1.0) +
            df.get('day_risk', 2.5) * weights.get('day_of_week', 1.0) +
            df.get('demographic_risk_score', 2.5) * weights.get('age', 1.0) +
            df.get('sms_risk_score', 2.5) * weights.get('sms', 1.0) +
            df.get('health_risk_score', 2.5) * weights.get('health', 1.0) +
            df.get('neighborhood_risk_score', 2.5) * weights.get('neighborhood', 1.0)
        )
        
        # Normalize to 0-5 scale
        total_weight = sum(weights.values())
        df['composite_risk_score'] = df['composite_risk_score'] / total_weight
        
        # Add percentile rank
        df['risk_percentile'] = df['composite_risk_score'].rank(pct=True) * 100
        
        self.logger.info(f"Composite risk scores calculated:")
        self.logger.info(f"  Min: {df['composite_risk_score'].min():.2f}")
        self.logger.info(f"  Max: {df['composite_risk_score'].max():.2f}")
        self.logger.info(f"  Mean: {df['composite_risk_score'].mean():.2f}")
        self.logger.info(f"  Median: {df['composite_risk_score'].median():.2f}")
        self.logger.info(f"  Std: {df['composite_risk_score'].std():.2f}")
        
        return df
    
    def assign_risk_tier(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign risk tier based on composite score.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with composite risk score
        
        Returns
        -------
        pd.DataFrame
            DataFrame with risk tier assignment
        """
        self.logger.info("Assigning risk tiers...")
        
        if 'composite_risk_score' not in df.columns:
            self.logger.warning("Composite risk score not found, calculating...")
            df = self.calculate_composite_risk(df)
        
        thresholds = self.risk_thresholds
        
        def get_tier(score):
            if score >= thresholds['critical']:
                return 'CRITICAL'
            elif score >= thresholds['high']:
                return 'HIGH'
            elif score >= thresholds['medium']:
                return 'MEDIUM'
            elif score >= thresholds['low']:
                return 'LOW'
            else:
                return 'MINIMAL'
        
        df['risk_tier'] = df['composite_risk_score'].apply(get_tier)
        
        # Add tier emoji for display
        tier_display = {
            'CRITICAL': '🔴 CRITICAL',
            'HIGH': '🟠 HIGH',
            'MEDIUM': '🟡 MEDIUM',
            'LOW': '🟢 LOW',
            'MINIMAL': '⭐ MINIMAL'
        }
        df['risk_tier_display'] = df['risk_tier'].map(tier_display)
        
        # Calculate tier distribution
        tier_dist = df['risk_tier'].value_counts()
        self.logger.info("Risk tier distribution:")
        for tier in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']:
            if tier in tier_dist.index:
                count = tier_dist[tier]
                pct = count / len(df) * 100
                self.logger.info(f"  {tier_display[tier]}: {count:,} ({pct:.1f}%)")
        
        return df
    
    def get_intervention_protocol(self, risk_tier: str) -> InterventionProtocol:
        """
        Get recommended intervention protocol for a risk tier.
        
        Parameters
        ----------
        risk_tier : str
            Risk tier (CRITICAL, HIGH, MEDIUM, LOW, MINIMAL)
        
        Returns
        -------
        InterventionProtocol
            Recommended intervention protocol
        """
        return self.protocols.get(risk_tier, self.protocols['MEDIUM'])
    
    def add_intervention_recommendations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add intervention recommendations based on risk tier.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with risk tiers
        
        Returns
        -------
        pd.DataFrame
            DataFrame with intervention columns
        """
        self.logger.info("Adding intervention recommendations...")
        
        if 'risk_tier' not in df.columns:
            self.logger.warning("Risk tier not found, calculating...")
            df = self.assign_risk_tier(df)
        
        # Add intervention columns
        df['sms_reminders_needed'] = df['risk_tier'].map(
            {tier: protocol.sms_reminders for tier, protocol in self.protocols.items()}
        )
        df['phone_call_required'] = df['risk_tier'].map(
            {tier: protocol.phone_call for tier, protocol in self.protocols.items()}
        )
        df['overbook_percentage'] = df['risk_tier'].map(
            {tier: protocol.overbook_percentage for tier, protocol in self.protocols.items()}
        )
        df['deposit_required'] = df['risk_tier'].map(
            {tier: protocol.deposit_required for tier, protocol in self.protocols.items()}
        )
        df['intervention_description'] = df['risk_tier'].map(
            {tier: protocol.description for tier, protocol in self.protocols.items()}
        )
        
        self.logger.info(f"Added intervention recommendations for {len(df)} appointments")
        
        return df
    
    @timer
    def score_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run complete risk scoring pipeline.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with engineered features
        
        Returns
        -------
        pd.DataFrame
            DataFrame with all risk scores and recommendations
        """
        self.logger.info("="*60)
        self.logger.info("Starting risk scoring pipeline...")
        
        initial_columns = len(df.columns)
        
        # Calculate risk scores
        df = self.calculate_composite_risk(df)
        df = self.assign_risk_tier(df)
        df = self.add_intervention_recommendations(df)
        
        final_columns = len(df.columns)
        new_columns = final_columns - initial_columns
        
        # Log summary
        self.logger.info("="*60)
        self.logger.info("RISK SCORING SUMMARY:")
        self.logger.info(f"  Risk factors calculated: {self.risk_factors_calculated}")
        self.logger.info(f"  New columns added: {new_columns}")
        self.logger.info(f"  Average risk score: {df['composite_risk_score'].mean():.2f}")
        self.logger.info(f"  High/Critical risk: {(df['risk_tier'].isin(['HIGH', 'CRITICAL'])).sum():,} appointments")
        self.logger.info("="*60)
        
        return df
    
    def get_risk_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of risk scores.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with risk scores
        
        Returns
        -------
        dict
            Summary statistics
        """
        if 'composite_risk_score' not in df.columns:
            return {}
        
        summary = {
            'total_appointments': len(df),
            'risk_score_stats': {
                'min': df['composite_risk_score'].min(),
                'max': df['composite_risk_score'].max(),
                'mean': df['composite_risk_score'].mean(),
                'median': df['composite_risk_score'].median(),
                'std': df['composite_risk_score'].std()
            },
            'tier_distribution': df['risk_tier'].value_counts().to_dict() if 'risk_tier' in df.columns else {},
            'high_risk_count': (df['risk_tier'].isin(['HIGH', 'CRITICAL'])).sum() if 'risk_tier' in df.columns else 0,
            'interventions_needed': {
                'phone_calls': df['phone_call_required'].sum() if 'phone_call_required' in df.columns else 0,
                'deposits': df['deposit_required'].sum() if 'deposit_required' in df.columns else 0
            }
        }
        
        return summary
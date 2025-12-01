"""
Feature Engineering Module
==========================
Functions to create new features for analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta

from .utils import timer


class FeatureEngineer:
    """Create derived features for analysis and modeling."""
    
    def __init__(self, config: dict):
        """
        Initialize FeatureEngineer with configuration.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("healthcare_pipeline.FeatureEngineer")
        self.features_created = []
    
    def create_lead_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate days between scheduling and appointment.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        
        Returns
        -------
        pd.DataFrame
            DataFrame with lead_days column
        """
        self.logger.info("Creating lead time feature...")
        
        # Find date columns (handle case variations)
        scheduled_col = None
        appointment_col = None
        
        for col in df.columns:
            if 'scheduled' in col.lower():
                scheduled_col = col
            elif 'appointment' in col.lower() and 'day' in col.lower():
                appointment_col = col
        
        if scheduled_col and appointment_col:
            # Ensure datetime
            df[scheduled_col] = pd.to_datetime(df[scheduled_col])
            df[appointment_col] = pd.to_datetime(df[appointment_col])
            
            # Calculate lead days
            df['lead_days'] = (df[appointment_col] - df[scheduled_col]).dt.days
            
            # Fix any negative lead days (data entry errors)
            negative_count = (df['lead_days'] < 0).sum()
            if negative_count > 0:
                self.logger.warning(f"Found {negative_count} negative lead days, setting to 0")
                df.loc[df['lead_days'] < 0, 'lead_days'] = 0
            
            self.features_created.append('lead_days')
            self.logger.info(f"Created lead_days feature (range: {df['lead_days'].min()} to {df['lead_days'].max()} days)")
        else:
            self.logger.warning("Could not find required date columns for lead time")
        
        return df
    
    def create_age_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age group categories.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        
        Returns
        -------
        pd.DataFrame
            DataFrame with age_group column
        """
        self.logger.info("Creating age groups...")
        
        if 'age' not in df.columns:
            self.logger.warning("Age column not found")
            return df
        
        age_bins = self.config['features']['age_bins']
        age_labels = self.config['features']['age_labels']
        
        df['age_group'] = pd.cut(
            df['age'],
            bins=age_bins,
            labels=age_labels,
            include_lowest=True
        )
        
        self.features_created.append('age_group')
        self.logger.info(f"Created {len(age_labels)} age groups")
        
        return df
    
    def create_lead_time_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bucket lead time into categories.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        
        Returns
        -------
        pd.DataFrame
            DataFrame with lead_time_category column
        """
        self.logger.info("Creating lead time categories...")
        
        if 'lead_days' not in df.columns:
            self.logger.warning("Lead_days column not found, creating it first...")
            df = self.create_lead_time(df)
        
        lead_bins = self.config['features']['lead_time_bins']
        lead_labels = self.config['features']['lead_time_labels']
        
        df['lead_time_category'] = pd.cut(
            df['lead_days'],
            bins=lead_bins,
            labels=lead_labels,
            include_lowest=True
        )
        
        self.features_created.append('lead_time_category')
        self.logger.info(f"Created {len(lead_labels)} lead time categories")
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract day of week, month, hour, etc. from appointment date.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        
        Returns
        -------
        pd.DataFrame
            DataFrame with time-based features
        """
        self.logger.info("Creating time-based features...")
        
        # Find appointment date column
        appt_col = None
        for col in df.columns:
            if 'appointment' in col.lower() and 'day' in col.lower():
                appt_col = col
                break
        
        if appt_col:
            # Ensure datetime
            df[appt_col] = pd.to_datetime(df[appt_col])
            
            # Extract features
            df['appointment_weekday'] = df[appt_col].dt.day_name()
            df['appointment_month'] = df[appt_col].dt.month_name()
            df['appointment_week'] = df[appt_col].dt.isocalendar().week
            df['appointment_day'] = df[appt_col].dt.day
            df['appointment_quarter'] = df[appt_col].dt.quarter
            
            # Binary features
            df['is_monday'] = (df[appt_col].dt.dayofweek == 0).astype(int)
            df['is_friday'] = (df[appt_col].dt.dayofweek == 4).astype(int)
            df['is_weekend'] = (df[appt_col].dt.dayofweek >= 5).astype(int)
            df['is_month_start'] = (df[appt_col].dt.day <= 7).astype(int)
            df['is_month_end'] = (df[appt_col].dt.day >= 24).astype(int)
            
            time_features = [
                'appointment_weekday', 'appointment_month', 'appointment_week',
                'appointment_day', 'appointment_quarter', 'is_monday',
                'is_friday', 'is_weekend', 'is_month_start', 'is_month_end'
            ]
            self.features_created.extend(time_features)
            self.logger.info(f"Created {len(time_features)} time-based features")
        else:
            self.logger.warning("Appointment date column not found")
        
        # Handle scheduled time if available
        sched_col = None
        for col in df.columns:
            if 'scheduled' in col.lower():
                sched_col = col
                break
        
        if sched_col:
            df[sched_col] = pd.to_datetime(df[sched_col])
            df['schedule_hour'] = df[sched_col].dt.hour
            df['scheduled_in_business_hours'] = df['schedule_hour'].between(8, 17).astype(int)
            
            self.features_created.extend(['schedule_hour', 'scheduled_in_business_hours'])
        
        return df
    
    def create_patient_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate historical no-show rate per patient.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        
        Returns
        -------
        pd.DataFrame
            DataFrame with patient history features
        """
        self.logger.info("Creating patient history features...")
        
        if 'patientid' not in df.columns:
            self.logger.warning("PatientId column not found")
            return df
        
        if 'no_show' not in df.columns:
            self.logger.warning("No_show column not found")
            return df
        
        # Sort by patient and date to ensure chronological order
        appt_col = None
        for col in df.columns:
            if 'appointment' in col.lower() and 'day' in col.lower():
                appt_col = col
                break
        
        if appt_col:
            df = df.sort_values(['patientid', appt_col])
        
        # Calculate cumulative statistics per patient
        df['patient_total_appointments'] = df.groupby('patientid').cumcount() + 1
        df['patient_previous_noshows'] = df.groupby('patientid')['no_show'].cumsum() - df['no_show']
        df['patient_previous_appointments'] = df['patient_total_appointments'] - 1
        
        # Calculate historical no-show rate
        df['patient_historical_noshow_rate'] = df.apply(
            lambda row: row['patient_previous_noshows'] / row['patient_previous_appointments'] 
            if row['patient_previous_appointments'] > 0 else 0,
            axis=1
        )
        
        # Flag first appointment
        df['is_first_appointment'] = (df['patient_total_appointments'] == 1).astype(int)
        
        # Calculate days since last appointment
        if appt_col:
            df['days_since_last_appointment'] = df.groupby('patientid')[appt_col].diff().dt.days
            df['days_since_last_appointment'].fillna(-1, inplace=True)  # -1 for first appointment
        
        history_features = [
            'patient_total_appointments', 'patient_previous_noshows',
            'patient_previous_appointments', 'patient_historical_noshow_rate',
            'is_first_appointment'
        ]
        if appt_col:
            history_features.append('days_since_last_appointment')
        
        self.features_created.extend(history_features)
        self.logger.info(f"Created {len(history_features)} patient history features")
        
        return df
    
    def create_health_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to health conditions.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        
        Returns
        -------
        pd.DataFrame
            DataFrame with health-related features
        """
        self.logger.info("Creating health-related features...")
        
        health_columns = ['hypertension', 'diabetes', 'alcoholism', 'handicap']
        found_columns = [col for col in health_columns if col in df.columns]
        
        if found_columns:
            # Total number of conditions
            condition_cols = [col for col in found_columns if col != 'handicap']
            if condition_cols:
                df['total_conditions'] = df[condition_cols].sum(axis=1)
                df['has_chronic_condition'] = (df['total_conditions'] > 0).astype(int)
                self.features_created.extend(['total_conditions', 'has_chronic_condition'])
            
            # Disability level categorization
            if 'handicap' in found_columns:
                df['has_disability'] = (df['handicap'] > 0).astype(int)
                df['severe_disability'] = (df['handicap'] >= 3).astype(int)
                self.features_created.extend(['has_disability', 'severe_disability'])
            
            # Risk categories based on conditions
            if 'total_conditions' in df.columns:
                df['health_risk_category'] = pd.cut(
                    df['total_conditions'],
                    bins=[-1, 0, 1, 2, 10],
                    labels=['Healthy', 'Single Condition', 'Multiple Conditions', 'Complex']
                )
                self.features_created.append('health_risk_category')
            
            self.logger.info(f"Created health features from {len(found_columns)} columns")
        else:
            self.logger.warning("No health condition columns found")
        
        return df
    
    def create_socioeconomic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create socioeconomic indicator features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        
        Returns
        -------
        pd.DataFrame
            DataFrame with socioeconomic features
        """
        self.logger.info("Creating socioeconomic features...")
        
        if 'scholarship' in df.columns:
            # Scholarship is a proxy for low income
            df['low_income_indicator'] = df['scholarship']
            self.features_created.append('low_income_indicator')
            
            # Interaction with age
            if 'age_group' in df.columns:
                df['young_low_income'] = (
                    (df['age_group'] == 'Young Adult') & 
                    (df['scholarship'] == 1)
                ).astype(int)
                self.features_created.append('young_low_income')
        
        # Neighborhood-based features (if we have neighborhood stats)
        if 'neighbourhood' in df.columns:
            # Calculate neighborhood-level statistics
            neighborhood_stats = df.groupby('neighbourhood').agg({
                'no_show': 'mean',
                'scholarship': 'mean' if 'scholarship' in df.columns else lambda x: 0,
                'age': 'mean' if 'age' in df.columns else lambda x: 0
            }).rename(columns={
                'no_show': 'neighborhood_noshow_rate',
                'scholarship': 'neighborhood_scholarship_rate',
                'age': 'neighborhood_avg_age'
            })
            
            # Merge back to main dataframe
            df = df.merge(neighborhood_stats, left_on='neighbourhood', right_index=True, how='left')
            
            # Create neighborhood risk category
            df['neighborhood_risk'] = pd.cut(
                df['neighborhood_noshow_rate'],
                bins=[0, 0.18, 0.22, 1.0],
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            )
            
            socio_features = ['neighborhood_noshow_rate', 'neighborhood_scholarship_rate', 
                            'neighborhood_avg_age', 'neighborhood_risk']
            self.features_created.extend(socio_features)
            self.logger.info(f"Created {len(socio_features)} neighborhood-based features")
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key variables.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        
        Returns
        -------
        pd.DataFrame
            DataFrame with interaction features
        """
        self.logger.info("Creating interaction features...")
        
        interactions_created = []
        
        # Young adult + long lead time (high risk combination)
        if 'age_group' in df.columns and 'lead_time_category' in df.columns:
            df['young_long_lead'] = (
                (df['age_group'] == 'Young Adult') & 
                (df['lead_time_category'].isin(['15-30 days', '30+ days']))
            ).astype(int)
            interactions_created.append('young_long_lead')
        
        # Monday + long lead (bad combination)
        if 'is_monday' in df.columns and 'lead_days' in df.columns:
            df['monday_long_lead'] = (
                (df['is_monday'] == 1) & 
                (df['lead_days'] > 14)
            ).astype(int)
            interactions_created.append('monday_long_lead')
        
        # First appointment + young adult
        if 'is_first_appointment' in df.columns and 'age_group' in df.columns:
            df['first_young'] = (
                (df['is_first_appointment'] == 1) & 
                (df['age_group'] == 'Young Adult')
            ).astype(int)
            interactions_created.append('first_young')
        
        # High risk neighborhood + no SMS
        if 'neighborhood_risk' in df.columns and 'sms_received' in df.columns:
            df['high_risk_no_sms'] = (
                (df['neighborhood_risk'] == 'High Risk') & 
                (df['sms_received'] == 0)
            ).astype(int)
            interactions_created.append('high_risk_no_sms')
        
        # Chronic condition + elderly
        if 'has_chronic_condition' in df.columns and 'age_group' in df.columns:
            df['elderly_chronic'] = (
                (df['has_chronic_condition'] == 1) & 
                (df['age_group'].isin(['Middle Age', 'Senior']))
            ).astype(int)
            interactions_created.append('elderly_chronic')
        
        self.features_created.extend(interactions_created)
        self.logger.info(f"Created {len(interactions_created)} interaction features")
        
        return df
    
    @timer
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.
        
        Parameters
        ----------
        df : pd.DataFrame
            Cleaned DataFrame
        
        Returns
        -------
        pd.DataFrame
            DataFrame with all engineered features
        """
        self.logger.info("="*60)
        self.logger.info("Starting feature engineering pipeline...")
        
        initial_columns = len(df.columns)
        
        # Create all features
        df = self.create_lead_time(df)
        df = self.create_age_groups(df)
        df = self.create_lead_time_category(df)
        df = self.create_time_features(df)
        df = self.create_patient_history(df)
        df = self.create_health_features(df)
        df = self.create_socioeconomic_features(df)
        df = self.create_interaction_features(df)
        
        final_columns = len(df.columns)
        new_features = final_columns - initial_columns
        
        # Log summary
        self.logger.info("="*60)
        self.logger.info("FEATURE ENGINEERING SUMMARY:")
        self.logger.info(f"  Initial columns: {initial_columns}")
        self.logger.info(f"  Final columns: {final_columns}")
        self.logger.info(f"  New features created: {new_features}")
        self.logger.info(f"  Feature list: {self.features_created}")
        self.logger.info("="*60)
        
        # Save feature-engineered data if configured
        if self.config['pipeline'].get('save_intermediate', True):
            output_path = self.config['paths']['features_data']
            df.to_csv(output_path, index=False)
            self.logger.info(f"Saved feature-engineered data to {output_path}")
        
        return df
    
    def get_features_created(self) -> List[str]:
        """
        Get list of features created.
        
        Returns
        -------
        list
            List of feature names created
        """
        return self.features_created
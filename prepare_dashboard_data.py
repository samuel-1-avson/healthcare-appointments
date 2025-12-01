# prepare_dashboard_data.py
"""
Prepare data for Looker Studio Dashboard
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def prepare_dashboard_data(input_path: str, output_path: str):
    """
    Prepare data specifically optimized for Looker Studio.
    
    Parameters
    ----------
    input_path : str
        Path to scored appointments data
    output_path : str
        Path for dashboard-ready data
    """
    
    print("📊 PREPARING DATA FOR LOOKER STUDIO")
    print("="*50)
    
    # Load scored data
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} records")
    
    # Select columns for dashboard
    dashboard_cols = [
        # IDs
        'patientid',
        'appointmentid',
        
        # Dates
        'scheduledday',
        'appointmentday',
        
        # Demographics
        'gender',
        'age',
        'age_group',
        
        # Location
        'neighbourhood',
        
        # Health
        'hypertension',
        'diabetes',
        'alcoholism',
        'handicap',
        'has_chronic_condition',
        
        # Scheduling
        'lead_days',
        'lead_time_category',
        'appointment_weekday',
        'appointment_month',
        
        # Interventions
        'sms_received',
        'scholarship',
        
        # Risk Scoring
        'composite_risk_score',
        'risk_tier',
        'risk_percentile',
        
        # Intervention Recommendations
        'sms_reminders_needed',
        'phone_call_required',
        'overbook_percentage',
        'deposit_required',
        
        # Outcome
        'no_show',
        'showed_up'
    ]
    
    # Filter to existing columns
    existing_cols = [col for col in dashboard_cols if col in df.columns]
    df_dashboard = df[existing_cols].copy()
    
    # Create additional dashboard-friendly columns
    
    # 1. Financial Impact
    cost_per_noshow = 150
    df_dashboard['noshow_cost'] = df_dashboard['no_show'] * cost_per_noshow
    
    # 2. Clean date formats
    if 'appointmentday' in df_dashboard.columns:
        df_dashboard['appointmentday'] = pd.to_datetime(df_dashboard['appointmentday'])
        df_dashboard['appointment_date'] = df_dashboard['appointmentday'].dt.date
        df_dashboard['appointment_year_month'] = df_dashboard['appointmentday'].dt.to_period('M').astype(str)
        df_dashboard['appointment_week'] = df_dashboard['appointmentday'].dt.isocalendar().week
    
    # 3. Risk tier with emoji (for display)
    tier_display = {
        'CRITICAL': '🔴 Critical',
        'HIGH': '🟠 High',
        'MEDIUM': '🟡 Medium',
        'LOW': '🟢 Low',
        'MINIMAL': '⭐ Minimal'
    }
    if 'risk_tier' in df_dashboard.columns:
        df_dashboard['risk_tier_display'] = df_dashboard['risk_tier'].map(tier_display)
    
    # 4. Binary to Yes/No (more readable in dashboards)
    binary_cols = ['no_show', 'showed_up', 'sms_received', 'scholarship', 
                   'hypertension', 'diabetes', 'phone_call_required', 'deposit_required']
    
    for col in binary_cols:
        if col in df_dashboard.columns:
            df_dashboard[f'{col}_label'] = df_dashboard[col].map({1: 'Yes', 0: 'No', True: 'Yes', False: 'No'})
    
    # 5. Intervention priority score (for sorting)
    if 'composite_risk_score' in df_dashboard.columns:
        df_dashboard['intervention_priority'] = (
            df_dashboard['composite_risk_score'] * 
            df_dashboard.get('lead_days', 1).fillna(1) / 10
        ).round(2)
    
    # 6. Appointment count (always 1, for aggregations)
    df_dashboard['appointment_count'] = 1
    
    # 7. Age brackets for easier filtering
    if 'age' in df_dashboard.columns:
        df_dashboard['age_bracket'] = pd.cut(
            df_dashboard['age'],
            bins=[0, 18, 30, 45, 60, 100],
            labels=['0-17', '18-29', '30-44', '45-59', '60+']
        )
    
    # 8. Lead time brackets
    if 'lead_days' in df_dashboard.columns:
        df_dashboard['lead_time_bracket'] = pd.cut(
            df_dashboard['lead_days'],
            bins=[-1, 0, 7, 14, 30, 365],
            labels=['Same Day', '1-7 Days', '8-14 Days', '15-30 Days', '30+ Days']
        )
    
    # 9. Clean neighborhood names
    if 'neighbourhood' in df_dashboard.columns:
        df_dashboard['neighbourhood'] = df_dashboard['neighbourhood'].str.strip().str.title()
    
    # 10. Create aggregation helper columns
    df_dashboard['total_conditions'] = df_dashboard[
        ['hypertension', 'diabetes', 'alcoholism']
    ].sum(axis=1) if all(col in df_dashboard.columns for col in ['hypertension', 'diabetes', 'alcoholism']) else 0
    
    # Save dashboard data
    df_dashboard.to_csv(output_path, index=False)
    print(f"\n✅ Dashboard data saved to {output_path}")
    print(f"   Rows: {len(df_dashboard):,}")
    print(f"   Columns: {len(df_dashboard.columns)}")
    
    # Also create summary tables for faster loading
    create_summary_tables(df_dashboard, Path(output_path).parent)
    
    return df_dashboard


def create_summary_tables(df: pd.DataFrame, output_dir: Path):
    """Create pre-aggregated summary tables for dashboard performance."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Daily Summary
    if 'appointmentday' in df.columns:
        daily_summary = df.groupby(df['appointmentday'].dt.date).agg({
            'appointment_count': 'sum',
            'no_show': 'sum',
            'noshow_cost': 'sum',
            'composite_risk_score': 'mean'
        }).reset_index()
        daily_summary.columns = ['date', 'total_appointments', 'no_shows', 'cost', 'avg_risk_score']
        daily_summary['noshow_rate'] = (daily_summary['no_shows'] / daily_summary['total_appointments'] * 100).round(2)
        daily_summary.to_csv(output_dir / 'summary_daily.csv', index=False)
        print(f"   ✅ Daily summary: {len(daily_summary)} rows")
    
    # 2. Neighborhood Summary
    if 'neighbourhood' in df.columns:
        neighborhood_summary = df.groupby('neighbourhood').agg({
            'appointment_count': 'sum',
            'no_show': ['sum', 'mean'],
            'noshow_cost': 'sum',
            'composite_risk_score': 'mean'
        }).reset_index()
        neighborhood_summary.columns = ['neighbourhood', 'total_appointments', 'no_shows', 'noshow_rate', 'total_cost', 'avg_risk_score']
        neighborhood_summary['noshow_rate'] = (neighborhood_summary['noshow_rate'] * 100).round(2)
        neighborhood_summary = neighborhood_summary.sort_values('no_shows', ascending=False)
        neighborhood_summary.to_csv(output_dir / 'summary_neighborhood.csv', index=False)
        print(f"   ✅ Neighborhood summary: {len(neighborhood_summary)} rows")
    
    # 3. Risk Tier Summary
    if 'risk_tier' in df.columns:
        tier_summary = df.groupby('risk_tier').agg({
            'appointment_count': 'sum',
            'no_show': ['sum', 'mean'],
            'noshow_cost': 'sum',
            'phone_call_required': 'sum' if 'phone_call_required' in df.columns else lambda x: 0
        }).reset_index()
        tier_summary.columns = ['risk_tier', 'total_appointments', 'no_shows', 'noshow_rate', 'total_cost', 'phone_calls_needed']
        tier_summary['noshow_rate'] = (tier_summary['noshow_rate'] * 100).round(2)
        tier_summary.to_csv(output_dir / 'summary_risk_tier.csv', index=False)
        print(f"   ✅ Risk tier summary: {len(tier_summary)} rows")
    
    # 4. Age Group Summary
    if 'age_group' in df.columns:
        age_summary = df.groupby('age_group').agg({
            'appointment_count': 'sum',
            'no_show': ['sum', 'mean'],
            'noshow_cost': 'sum'
        }).reset_index()
        age_summary.columns = ['age_group', 'total_appointments', 'no_shows', 'noshow_rate', 'total_cost']
        age_summary['noshow_rate'] = (age_summary['noshow_rate'] * 100).round(2)
        age_summary.to_csv(output_dir / 'summary_age_group.csv', index=False)
        print(f"   ✅ Age group summary: {len(age_summary)} rows")
    
    # 5. Day of Week Summary
    if 'appointment_weekday' in df.columns:
        dow_summary = df.groupby('appointment_weekday').agg({
            'appointment_count': 'sum',
            'no_show': ['sum', 'mean'],
            'noshow_cost': 'sum'
        }).reset_index()
        dow_summary.columns = ['day_of_week', 'total_appointments', 'no_shows', 'noshow_rate', 'total_cost']
        dow_summary['noshow_rate'] = (dow_summary['noshow_rate'] * 100).round(2)
        dow_summary.to_csv(output_dir / 'summary_day_of_week.csv', index=False)
        print(f"   ✅ Day of week summary: {len(dow_summary)} rows")
    
    print("\n✅ All summary tables created!")


# Run data preparation
if __name__ == "__main__":
    prepare_dashboard_data(
        input_path='data/processed/appointments_features.csv',
        output_path='data/dashboard/appointments_dashboard.csv'
    )
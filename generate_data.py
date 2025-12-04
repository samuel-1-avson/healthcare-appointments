"""
============================================================================
HEALTHCARE APPOINTMENTS - DATA GENERATION (LOCAL FILE VERSION)
============================================================================
Run this AFTER downloading the Kaggle dataset manually.
Place KaggleV2-May-2016.csv in data/raw/ folder first!
============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DASHBOARD_DIR = DATA_DIR / "dashboard"
OUTPUTS_DIR = Path("outputs")
SQL_DIR = OUTPUTS_DIR / "sql"

for directory in [RAW_DIR, PROCESSED_DIR, DASHBOARD_DIR, OUTPUTS_DIR, SQL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print("="*70)
print("HEALTHCARE APPOINTMENTS - DATA GENERATION")
print("="*70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# STEP 1: LOAD LOCAL FILE
# ============================================================================

print("📥 STEP 1: Loading data from local file...")

# Try to find the file with different possible names
possible_files = [
    RAW_DIR / "KaggleV2-May-2016.csv",
    RAW_DIR / "appointments_raw.csv",
    RAW_DIR / "noshowappointments.csv",
    Path("KaggleV2-May-2016.csv"),  # In current directory
    Path("data") / "KaggleV2-May-2016.csv"
]

df_raw = None
for file_path in possible_files:
    if file_path.exists():
        try:
            df_raw = pd.read_csv(file_path, encoding='latin-1')
            print(f"   ✅ Loaded from: {file_path}")
            break
        except Exception as e:
            print(f"   ❌ Error reading {file_path}: {e}")

if df_raw is None:
    print("\n❌ ERROR: Could not find the data file!")
    print("\nPlease download from Kaggle and place the file in one of these locations:")
    for f in possible_files[:3]:
        print(f"   - {f}")
    print("\nDownload link: https://www.kaggle.com/datasets/joniarroba/noshowappointments")
    exit(1)

# Save raw data copy
raw_path = RAW_DIR / "appointments_raw.csv"
df_raw.to_csv(raw_path, index=False)
print(f"   💾 Saved raw data: {raw_path}")
print(f"   📊 Shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")



# ============================================================================
# STEP 2: CLEAN DATA
# ============================================================================

print("\n🧹 STEP 2: Cleaning data...")

df = df_raw.copy()

# 2.1 Standardize column names
df.columns = df.columns.str.replace('-', '_')
df.columns = df.columns.str.lower()
df = df.rename(columns={
    'hipertension': 'hypertension',
    'handcap': 'handicap',
    'no_show': 'no_show_original'
})

# 2.2 Convert dates
df['scheduledday'] = pd.to_datetime(df['scheduledday'])
df['appointmentday'] = pd.to_datetime(df['appointmentday'])

# 2.3 Fix age outliers
df['age'] = df['age'].abs()
df.loc[df['age'] > 100, 'age'] = df['age'].median()

# 2.4 Fix no-show encoding (original: 'No'=showed up, 'Yes'=no-show)
df['no_show'] = df['no_show_original'].map({'No': 0, 'Yes': 1})
df['showed_up'] = df['no_show_original'].map({'No': 1, 'Yes': 0})

# 2.5 Remove duplicates
df = df.drop_duplicates(subset=['appointmentid'], keep='first')

print(f"   ✅ Cleaned {len(df):,} records")

# Save cleaned data
cleaned_path = PROCESSED_DIR / "appointments_cleaned.csv"
df.to_csv(cleaned_path, index=False)
print(f"   💾 Saved: {cleaned_path}")

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================

print("\n✨ STEP 3: Engineering features...")

# 3.1 Lead time features
df['lead_days'] = (df['appointmentday'] - df['scheduledday']).dt.days
df.loc[df['lead_days'] < 0, 'lead_days'] = 0

df['lead_time_category'] = pd.cut(
    df['lead_days'],
    bins=[-1, 0, 7, 14, 30, 365],
    labels=['Same Day', '1-7 days', '8-14 days', '15-30 days', '30+ days']
)

# 3.2 Age groups
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 12, 18, 35, 50, 65, 100],
    labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Middle Age', 'Senior']
)

# 3.3 Time features
df['appointment_weekday'] = df['appointmentday'].dt.day_name()
df['appointment_month'] = df['appointmentday'].dt.month_name()
df['appointment_week'] = df['appointmentday'].dt.isocalendar().week
df['appointment_day'] = df['appointmentday'].dt.day

# 3.4 Patient history (simplified - cumulative)
df = df.sort_values(['patientid', 'appointmentday'])
df['patient_total_appointments'] = df.groupby('patientid').cumcount() + 1
df['patient_previous_noshows'] = df.groupby('patientid')['no_show'].cumsum() - df['no_show']
df['patient_previous_appointments'] = df['patient_total_appointments'] - 1
df['patient_historical_noshow_rate'] = np.where(
    df['patient_previous_appointments'] > 0,
    df['patient_previous_noshows'] / df['patient_previous_appointments'],
    0
)
df['is_first_appointment'] = (df['patient_total_appointments'] == 1).astype(int)

# 3.5 Health features
df['has_chronic_condition'] = ((df['hypertension'] == 1) | (df['diabetes'] == 1)).astype(int)
df['total_conditions'] = df['hypertension'] + df['diabetes'] + df['alcoholism']

# 3.6 Neighborhood features
neighborhood_stats = df.groupby('neighbourhood')['no_show'].mean().reset_index()
neighborhood_stats.columns = ['neighbourhood', 'neighborhood_noshow_rate']
df = df.merge(neighborhood_stats, on='neighbourhood', how='left')

df['neighborhood_risk'] = pd.cut(
    df['neighborhood_noshow_rate'],
    bins=[0, 0.18, 0.22, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

# 3.7 Interaction features
df['young_long_lead'] = (
    (df['age_group'] == 'Young Adult') & 
    (df['lead_time_category'].isin(['15-30 days', '30+ days']))
).astype(int)

df['is_monday'] = (df['appointment_weekday'] == 'Monday').astype(int)
df['is_weekend'] = (df['appointment_weekday'].isin(['Saturday', 'Sunday'])).astype(int)

print(f"   ✅ Created {len(df.columns) - len(df_raw.columns)} new features")

# Save feature-engineered data
features_path = PROCESSED_DIR / "appointments_features.csv"
df.to_csv(features_path, index=False)
print(f"   💾 Saved: {features_path}")

# ============================================================================
# STEP 4: RISK SCORING
# ============================================================================

print("\n🎯 STEP 4: Calculating risk scores...")

# 4.1 Patient risk score
def score_patient_history(row):
    if row['patient_previous_appointments'] == 0:
        return 2.5  # New patient
    rate = row['patient_historical_noshow_rate']
    if rate >= 0.5: return 5.0
    elif rate >= 0.3: return 4.0
    elif rate >= 0.15: return 3.0
    elif rate >= 0.05: return 2.0
    else: return 1.0

df['patient_risk_score'] = df.apply(score_patient_history, axis=1)

# 4.2 Lead time risk
def score_lead_time(days):
    if days <= 0: return 1.0
    elif days <= 3: return 2.0
    elif days <= 7: return 3.0
    elif days <= 14: return 4.0
    else: return 5.0

df['lead_time_risk'] = df['lead_days'].apply(score_lead_time)

# 4.3 Day risk
day_scores = {
    'Saturday': 1.0, 'Sunday': 1.5, 'Wednesday': 2.0, 'Thursday': 2.0,
    'Friday': 3.0, 'Tuesday': 3.5, 'Monday': 4.0
}
df['day_risk'] = df['appointment_weekday'].map(day_scores).fillna(2.5)

# 4.4 Age risk
age_scores = {
    'Child': 2.0, 'Teen': 3.0, 'Young Adult': 4.0,
    'Adult': 2.5, 'Middle Age': 2.0, 'Senior': 1.5
}
df['age_risk'] = df['age_group'].map(age_scores).fillna(2.5)

# 4.5 Neighborhood risk score
def score_neighborhood(rate):
    if rate > 0.25: return 5.0
    elif rate > 0.22: return 4.0
    elif rate > 0.20: return 3.0
    elif rate > 0.18: return 2.0
    else: return 1.0

df['neighborhood_risk_score'] = df['neighborhood_noshow_rate'].apply(score_neighborhood)

# 4.6 Health risk (chronic conditions = lower risk)
df['health_risk_score'] = df['has_chronic_condition'].apply(lambda x: 1.5 if x == 1 else 2.5)

# 4.7 Socioeconomic risk
df['socio_risk_score'] = df['scholarship'].apply(lambda x: 3.5 if x == 1 else 2.0)

# 4.8 SMS risk
df['sms_risk_score'] = df.apply(
    lambda row: 3.0 if row['sms_received'] == 1 and row['lead_days'] > 7
    else 4.0 if row['sms_received'] == 0 and row['lead_days'] > 7
    else 2.0,
    axis=1
)

# 4.9 Composite risk score (weighted average)
weights = {
    'patient': 3.0, 'lead_time': 2.5, 'day': 1.5, 'age': 2.0,
    'sms': 1.0, 'health': 1.5, 'socio': 1.5, 'neighborhood': 2.0
}
total_weight = sum(weights.values())

df['composite_risk_score'] = (
    df['patient_risk_score'] * weights['patient'] +
    df['lead_time_risk'] * weights['lead_time'] +
    df['day_risk'] * weights['day'] +
    df['age_risk'] * weights['age'] +
    df['sms_risk_score'] * weights['sms'] +
    df['health_risk_score'] * weights['health'] +
    df['socio_risk_score'] * weights['socio'] +
    df['neighborhood_risk_score'] * weights['neighborhood']
) / total_weight

# 4.10 Risk percentile
df['risk_percentile'] = df['composite_risk_score'].rank(pct=True) * 100

# 4.11 Risk tier
def assign_tier(score):
    if score >= 3.5: return 'CRITICAL'
    elif score >= 3.0: return 'HIGH'
    elif score >= 2.5: return 'MEDIUM'
    elif score >= 2.0: return 'LOW'
    else: return 'MINIMAL'

df['risk_tier'] = df['composite_risk_score'].apply(assign_tier)

tier_display = {
    'CRITICAL': '🔴 Critical',
    'HIGH': '🟠 High',
    'MEDIUM': '🟡 Medium',
    'LOW': '🟢 Low',
    'MINIMAL': '⭐ Minimal'
}
df['risk_tier_display'] = df['risk_tier'].map(tier_display)

# 4.12 Intervention recommendations
intervention_map = {
    'CRITICAL': {'sms': 3, 'phone': True, 'deposit': True, 'overbook': 0.20},
    'HIGH': {'sms': 2, 'phone': True, 'deposit': False, 'overbook': 0.15},
    'MEDIUM': {'sms': 2, 'phone': False, 'deposit': False, 'overbook': 0.10},
    'LOW': {'sms': 1, 'phone': False, 'deposit': False, 'overbook': 0.05},
    'MINIMAL': {'sms': 1, 'phone': False, 'deposit': False, 'overbook': 0.00}
}

df['sms_reminders_needed'] = df['risk_tier'].map(lambda x: intervention_map[x]['sms'])
df['phone_call_required'] = df['risk_tier'].map(lambda x: intervention_map[x]['phone'])
df['deposit_required'] = df['risk_tier'].map(lambda x: intervention_map[x]['deposit'])
df['overbook_percentage'] = df['risk_tier'].map(lambda x: intervention_map[x]['overbook'])

print(f"   ✅ Calculated risk scores for {len(df):,} appointments")
print(f"   📊 Risk tier distribution:")
for tier in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']:
    count = (df['risk_tier'] == tier).sum()
    pct = count / len(df) * 100
    print(f"      {tier_display[tier]}: {count:,} ({pct:.1f}%)")

# Save scored data
scored_path = PROCESSED_DIR / "appointments_scored.csv"
df.to_csv(scored_path, index=False)
print(f"   💾 Saved: {scored_path}")

# ============================================================================
# STEP 5: CREATE DASHBOARD-READY DATA
# ============================================================================

print("\n📊 STEP 5: Creating dashboard-ready data...")

# 5.1 Main dashboard data
df_dashboard = df.copy()

# Add financial columns
df_dashboard['noshow_cost'] = df_dashboard['no_show'] * 150
df_dashboard['appointment_count'] = 1

# Add date components
df_dashboard['appointment_date'] = df_dashboard['appointmentday'].dt.date
df_dashboard['appointment_year_month'] = df_dashboard['appointmentday'].dt.to_period('M').astype(str)

# Add label columns
for col in ['no_show', 'showed_up', 'sms_received', 'scholarship', 
            'hypertension', 'diabetes', 'phone_call_required', 'deposit_required']:
    if col in df_dashboard.columns:
        df_dashboard[f'{col}_label'] = df_dashboard[col].map({1: 'Yes', 0: 'No', True: 'Yes', False: 'No'})

# Save main dashboard data
dashboard_main_path = DASHBOARD_DIR / "appointments_dashboard.csv"
df_dashboard.to_csv(dashboard_main_path, index=False)
print(f"   💾 Saved: {dashboard_main_path}")

# ============================================================================
# STEP 6: CREATE SUMMARY TABLES
# ============================================================================

print("\n📈 STEP 6: Creating summary tables...")

# 6.1 Daily summary
daily_summary = df_dashboard.groupby(df_dashboard['appointmentday'].dt.date).agg({
    'appointment_count': 'sum',
    'no_show': 'sum',
    'noshow_cost': 'sum',
    'composite_risk_score': 'mean'
}).reset_index()
daily_summary.columns = ['date', 'total_appointments', 'no_shows', 'cost', 'avg_risk_score']
daily_summary['noshow_rate'] = (daily_summary['no_shows'] / daily_summary['total_appointments'] * 100).round(2)
daily_summary.to_csv(DASHBOARD_DIR / "summary_daily.csv", index=False)
print(f"   💾 Saved: summary_daily.csv ({len(daily_summary)} rows)")

# 6.2 Neighborhood summary
neighborhood_summary = df_dashboard.groupby('neighbourhood').agg({
    'appointment_count': 'sum',
    'no_show': ['sum', 'mean'],
    'noshow_cost': 'sum',
    'composite_risk_score': 'mean'
}).reset_index()
neighborhood_summary.columns = ['neighbourhood', 'total_appointments', 'no_shows', 'noshow_rate', 'total_cost', 'avg_risk_score']
neighborhood_summary['noshow_rate'] = (neighborhood_summary['noshow_rate'] * 100).round(2)
neighborhood_summary = neighborhood_summary.sort_values('no_shows', ascending=False)
neighborhood_summary.to_csv(DASHBOARD_DIR / "summary_neighborhood.csv", index=False)
print(f"   💾 Saved: summary_neighborhood.csv ({len(neighborhood_summary)} rows)")

# 6.3 Risk tier summary
tier_summary = df_dashboard.groupby('risk_tier').agg({
    'appointment_count': 'sum',
    'no_show': ['sum', 'mean'],
    'noshow_cost': 'sum',
    'phone_call_required': 'sum',
    'deposit_required': 'sum'
}).reset_index()
tier_summary.columns = ['risk_tier', 'total_appointments', 'no_shows', 'noshow_rate', 'total_cost', 'phone_calls_needed', 'deposits_needed']
tier_summary['noshow_rate'] = (tier_summary['noshow_rate'] * 100).round(2)
tier_summary.to_csv(DASHBOARD_DIR / "summary_risk_tier.csv", index=False)
print(f"   💾 Saved: summary_risk_tier.csv ({len(tier_summary)} rows)")

# 6.4 Age group summary
age_summary = df_dashboard.groupby('age_group').agg({
    'appointment_count': 'sum',
    'no_show': ['sum', 'mean'],
    'noshow_cost': 'sum'
}).reset_index()
age_summary.columns = ['age_group', 'total_appointments', 'no_shows', 'noshow_rate', 'total_cost']
age_summary['noshow_rate'] = (age_summary['noshow_rate'] * 100).round(2)
age_summary.to_csv(DASHBOARD_DIR / "summary_age_group.csv", index=False)
print(f"   💾 Saved: summary_age_group.csv ({len(age_summary)} rows)")

# 6.5 Day of week summary
dow_summary = df_dashboard.groupby('appointment_weekday').agg({
    'appointment_count': 'sum',
    'no_show': ['sum', 'mean'],
    'noshow_cost': 'sum'
}).reset_index()
dow_summary.columns = ['day_of_week', 'total_appointments', 'no_shows', 'noshow_rate', 'total_cost']
dow_summary['noshow_rate'] = (dow_summary['noshow_rate'] * 100).round(2)
dow_summary.to_csv(DASHBOARD_DIR / "summary_day_of_week.csv", index=False)
print(f"   💾 Saved: summary_day_of_week.csv ({len(dow_summary)} rows)")

# 6.6 Lead time summary
lead_summary = df_dashboard.groupby('lead_time_category').agg({
    'appointment_count': 'sum',
    'no_show': ['sum', 'mean'],
    'noshow_cost': 'sum'
}).reset_index()
lead_summary.columns = ['lead_time_category', 'total_appointments', 'no_shows', 'noshow_rate', 'total_cost']
lead_summary['noshow_rate'] = (lead_summary['noshow_rate'] * 100).round(2)
lead_summary.to_csv(DASHBOARD_DIR / "summary_lead_time.csv", index=False)
print(f"   💾 Saved: summary_lead_time.csv ({len(lead_summary)} rows)")

# 6.7 SMS effectiveness summary
sms_summary = df_dashboard.groupby('sms_received').agg({
    'appointment_count': 'sum',
    'no_show': ['sum', 'mean'],
    'noshow_cost': 'sum'
}).reset_index()
sms_summary.columns = ['sms_received', 'total_appointments', 'no_shows', 'noshow_rate', 'total_cost']
sms_summary['noshow_rate'] = (sms_summary['noshow_rate'] * 100).round(2)
sms_summary['sms_status'] = sms_summary['sms_received'].map({0: 'No SMS', 1: 'SMS Sent'})
sms_summary.to_csv(DASHBOARD_DIR / "summary_sms.csv", index=False)
print(f"   💾 Saved: summary_sms.csv ({len(sms_summary)} rows)")

# 6.8 Patient segments summary
patient_summary = df_dashboard.groupby('patientid').agg({
    'appointment_count': 'sum',
    'no_show': ['sum', 'mean'],
    'age': 'max',
    'lead_days': 'mean'
}).reset_index()
patient_summary.columns = ['patientid', 'visit_count', 'total_noshows', 'noshow_rate', 'age', 'avg_lead_days']
patient_summary['noshow_rate'] = patient_summary['noshow_rate'] * 100

def categorize_patient(row):
    count = row['visit_count']
    rate = row['noshow_rate']
    
    if count == 1 and rate == 100: return 'Ghost Patient'
    if count == 1 and rate == 0: return 'New Good Patient'
    if count >= 5 and rate == 0: return 'VIP'
    if count >= 3 and rate >= 66: return 'Chronic Problem'
    if count >= 2 and 20 <= rate < 66: return 'Inconsistent'
    if count >= 2 and rate < 20: return 'Reliable Regular'
    return 'Other'

patient_summary['patient_segment'] = patient_summary.apply(categorize_patient, axis=1)

patient_segment_summary = patient_summary['patient_segment'].value_counts().reset_index()
patient_segment_summary.columns = ['patient_segment', 'patient_count']
patient_segment_summary['percentage'] = (patient_segment_summary['patient_count'] / len(patient_summary) * 100).round(2)
patient_segment_summary.to_csv(DASHBOARD_DIR / "summary_patient_segments.csv", index=False)
print(f"   💾 Saved: summary_patient_segments.csv ({len(patient_segment_summary)} rows)")

# 6.9 Behavior Evolution (New)
# Analyze no-show rate by visit number (1st, 2nd, 3rd...)
behavior_df = df_dashboard.copy()
# Ensure we have visit number
if 'patient_total_appointments' not in behavior_df.columns:
    behavior_df = behavior_df.sort_values(['patientid', 'appointmentday'])
    behavior_df['patient_total_appointments'] = behavior_df.groupby('patientid').cumcount() + 1

behavior_summary = behavior_df.groupby('patient_total_appointments').agg({
    'appointment_count': 'sum',
    'no_show': 'mean'
}).reset_index()
behavior_summary.columns = ['visit_number', 'total_appointments', 'noshow_rate']
behavior_summary['noshow_rate'] = (behavior_summary['noshow_rate'] * 100).round(2)
# Filter to first 10 visits for cleaner chart
behavior_summary = behavior_summary[behavior_summary['visit_number'] <= 10]
behavior_summary.to_csv(DASHBOARD_DIR / "summary_behavior.csv", index=False)
print(f"   💾 Saved: summary_behavior.csv ({len(behavior_summary)} rows)")

# ============================================================================
# STEP 7: CREATE SQL ANALYSIS EXPORTS
# ============================================================================

print("\n📋 STEP 7: Creating SQL analysis exports...")

# Serial no-show offenders
serial_noshows = df_dashboard.groupby('patientid').agg({
    'appointment_count': 'sum',
    'no_show': 'sum'
}).reset_index()
serial_noshows.columns = ['patientid', 'total_appointments', 'missed_appointments']
serial_noshows['noshow_rate'] = (serial_noshows['missed_appointments'] / serial_noshows['total_appointments'] * 100).round(2)
serial_noshows = serial_noshows.sort_values('missed_appointments', ascending=False).head(100)
serial_noshows.to_csv(SQL_DIR / "sql_serial_noshows.csv", index=False)
print(f"   💾 Saved: sql_serial_noshows.csv")

# Priority interventions
priority = df_dashboard[df_dashboard['risk_tier'].isin(['HIGH', 'CRITICAL'])].copy()
priority = priority[['appointmentid', 'patientid', 'neighbourhood', 'age_group', 
                     'appointment_weekday', 'lead_days', 'composite_risk_score', 
                     'risk_tier', 'phone_call_required', 'deposit_required']].head(500)
priority.to_csv(SQL_DIR / "sql_priority_interventions.csv", index=False)
print(f"   💾 Saved: sql_priority_interventions.csv")
print("   3. Use summary files for faster dashboard loading")

print("\n" + "="*70)
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

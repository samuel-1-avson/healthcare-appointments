"""
============================================================================
HEALTHCARE APPOINTMENTS - SYNTHETIC DATA GENERATOR
============================================================================
Generates realistic synthetic data matching the Kaggle dataset structure.
Use this if you cannot download the original data.
============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

N_APPOINTMENTS = 110527  # Same as original dataset
N_PATIENTS = 62299       # Same as original dataset

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DASHBOARD_DIR = DATA_DIR / "dashboard"
OUTPUTS_DIR = Path("outputs")
SQL_DIR = OUTPUTS_DIR / "sql"

for directory in [RAW_DIR, PROCESSED_DIR, DASHBOARD_DIR, OUTPUTS_DIR, SQL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print("="*70)
print("HEALTHCARE APPOINTMENTS - SYNTHETIC DATA GENERATION")
print("="*70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# STEP 1: GENERATE RAW DATA
# ============================================================================

print("🔨 STEP 1: Generating synthetic data...")

# Brazilian neighborhoods (from the real dataset)
neighborhoods = [
    'JARDIM CAMBURI', 'MARIA ORTIZ', 'RESISTÊNCIA', 'JARDIM DA PENHA',
    'ITARARÉ', 'CENTRO', 'TABUAZEIRO', 'JESUS DE NAZARETH', 'SANTO ANTÔNIO',
    'BONFIM', 'ANDORINHAS', 'SANTA MARTHA', 'PRAIA DO CANTO', 'MARUÍPE',
    'SÃO PEDRO', 'CONSOLAÇÃO', 'SÃO BENEDITO', 'GURIGICA', 'CARATOÍRA',
    'SANTO ANDRÉ', 'DA PENHA', 'GOIABEIRAS', 'SOLON BORGES', 'BOA VISTA',
    'REPÚBLICA', 'NOVA PALESTINA', 'ILHA DO PRÍNCIPE', 'SÃO CRISTÓVÃO',
    'SANTA CECÍLIA', 'JABOUR', 'CONQUISTA', 'HORTO', 'BARRO VERMELHO',
    'INHANGUETÁ', 'GRANDE VITÓRIA', 'SANTA TEREZA', 'UNIVERSITÁRIO',
    'MORRO DO QUADRO', 'JOANA D´ARC', 'CRUZAMENTO', 'SANTOS DUMONT',
    'MONTE BELO', 'FONTE GRANDE', 'ANTÔNIO HONÓRIO', 'ILHA DAS CAIEIRAS',
    'ROMÃO', 'DO CABRAL', 'PRAIA DO SUÁ', 'ENSEADA DO SUÁ', 'SANTA HELENA',
    'BELA VISTA', 'PIEDADE', 'SÃO JOSÉ', 'PONTAL DE CAMBURI', 'FORTE SÃO JOÃO',
    'DO MOSCOSO', 'ILHA DE SANTA MARIA', 'PARQUE MOSCOSO', 'SANTA CLARA',
    'ARIOVALDO FAVALESSA', 'REDENÇÃO', 'FRADINHOS', 'JUCUTUQUARA',
    'VILA RUBIM', 'SANTA LÚCIA', 'ILHA DO FRADE', 'AEROPORTO', 'COMDUSA',
    'SEGURANÇA DO LAR', 'ESTRELINHA', 'MORADA DE CAMBURI', 'ILHAS OCEÂNICAS',
    'PARQUE INDUSTRIAL', 'DE LOURDES', 'NAZARETH', 'MARIO CYPRESTE',
    'SANTOS REIS', 'BENÉVOLO', 'MATA DA PRAIA', 'SANTOS DUMONT'
]

# Date range (April - June 2016, like original)
start_date = datetime(2016, 4, 29)
end_date = datetime(2016, 6, 8)
date_range = (end_date - start_date).days

# Generate patient IDs (some patients have multiple appointments)
patient_ids = []
for _ in range(N_APPOINTMENTS):
    if random.random() < 0.7:  # 70% chance of being a repeat patient
        patient_ids.append(random.randint(1, N_PATIENTS))
    else:
        patient_ids.append(random.randint(1, N_PATIENTS))

# Generate appointment data
print("   Generating appointments...")

data = {
    'PatientId': patient_ids,
    'AppointmentID': [f'A{5000000 + i}' for i in range(N_APPOINTMENTS)],
    'Gender': np.random.choice(['M', 'F'], size=N_APPOINTMENTS, p=[0.35, 0.65]),
    'Age': np.clip(np.random.normal(40, 22, N_APPOINTMENTS).astype(int), 0, 100),
    'Neighbourhood': np.random.choice(neighborhoods, size=N_APPOINTMENTS, 
                                       p=np.random.dirichlet(np.ones(len(neighborhoods)))),
    'Scholarship': np.random.choice([0, 1], size=N_APPOINTMENTS, p=[0.90, 0.10]),
    'Hipertension': np.random.choice([0, 1], size=N_APPOINTMENTS, p=[0.80, 0.20]),
    'Diabetes': np.random.choice([0, 1], size=N_APPOINTMENTS, p=[0.93, 0.07]),
    'Alcoholism': np.random.choice([0, 1], size=N_APPOINTMENTS, p=[0.97, 0.03]),
    'Handcap': np.random.choice([0, 1, 2, 3, 4], size=N_APPOINTMENTS, p=[0.98, 0.015, 0.003, 0.001, 0.001])
}

# Generate dates
appointment_dates = [start_date + timedelta(days=random.randint(0, date_range)) for _ in range(N_APPOINTMENTS)]
scheduled_dates = [appt - timedelta(days=max(0, int(np.random.exponential(10)))) for appt in appointment_dates]

data['AppointmentDay'] = [d.strftime('%Y-%m-%dT00:00:00Z') for d in appointment_dates]
data['ScheduledDay'] = [d.strftime('%Y-%m-%dT%H:%M:%SZ') for d in scheduled_dates]

# Generate SMS_received (more likely for longer lead times)
lead_days = [(a - s).days for a, s in zip(appointment_dates, scheduled_dates)]
data['SMS_received'] = [1 if (ld > 3 and random.random() < 0.6) else 0 for ld in lead_days]

# Generate No-show based on realistic factors
print("   Generating no-show outcomes based on realistic patterns...")

def calculate_noshow_probability(row_idx):
    """Calculate no-show probability based on various factors."""
    base_prob = 0.20  # 20% base rate
    
    # Age effect
    age = data['Age'][row_idx]
    if 18 <= age <= 35:
        base_prob += 0.05  # Young adults higher risk
    elif age >= 60:
        base_prob -= 0.05  # Seniors lower risk
    
    # Lead time effect
    ld = lead_days[row_idx]
    if ld == 0:
        base_prob -= 0.12  # Same day much lower
    elif ld <= 7:
        base_prob -= 0.05
    elif ld > 14:
        base_prob += 0.05
    
    # Day of week effect
    dow = appointment_dates[row_idx].weekday()
    if dow == 0:  # Monday
        base_prob += 0.02
    elif dow == 5:  # Saturday
        base_prob -= 0.03
    
    # Chronic conditions lower no-show
    if data['Hipertension'][row_idx] or data['Diabetes'][row_idx]:
        base_prob -= 0.03
    
    # Scholarship (low income) higher no-show
    if data['Scholarship'][row_idx]:
        base_prob += 0.04
    
    # Clamp probability
    return max(0.02, min(0.45, base_prob))

noshow_probs = [calculate_noshow_probability(i) for i in range(N_APPOINTMENTS)]
data['No-show'] = ['Yes' if random.random() < p else 'No' for p in noshow_probs]

# Create DataFrame
df_raw = pd.DataFrame(data)

# Save raw data
raw_path = RAW_DIR / "appointments_raw.csv"
df_raw.to_csv(raw_path, index=False)
print(f"   ✅ Generated {N_APPOINTMENTS:,} appointments for {N_PATIENTS:,} patients")
print(f"   💾 Saved: {raw_path}")

# ============================================================================
# STEP 2: CLEAN DATA
# ============================================================================

print("\n🧹 STEP 2: Cleaning data...")

df = df_raw.copy()

# Standardize column names
df.columns = df.columns.str.replace('-', '_').str.lower()
df = df.rename(columns={
    'hipertension': 'hypertension',
    'handcap': 'handicap',
    'no_show': 'no_show_original'
})

# Convert dates
df['scheduledday'] = pd.to_datetime(df['scheduledday'])
df['appointmentday'] = pd.to_datetime(df['appointmentday'])

# Fix no-show encoding
df['no_show'] = df['no_show_original'].map({'No': 0, 'Yes': 1})
df['showed_up'] = df['no_show_original'].map({'No': 1, 'Yes': 0})

print(f"   ✅ Cleaned {len(df):,} records")
print(f"   📊 No-show rate: {df['no_show'].mean()*100:.1f}%")

cleaned_path = PROCESSED_DIR / "appointments_cleaned.csv"
df.to_csv(cleaned_path, index=False)
print(f"   💾 Saved: {cleaned_path}")

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================

print("\n✨ STEP 3: Engineering features...")

# Lead time
df['lead_days'] = (df['appointmentday'] - df['scheduledday']).dt.days
df.loc[df['lead_days'] < 0, 'lead_days'] = 0

df['lead_time_category'] = pd.cut(
    df['lead_days'],
    bins=[-1, 0, 7, 14, 30, 365],
    labels=['Same Day', '1-7 days', '8-14 days', '15-30 days', '30+ days']
)

# Age groups
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 12, 18, 35, 50, 65, 100],
    labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Middle Age', 'Senior']
)

# Time features
df['appointment_weekday'] = df['appointmentday'].dt.day_name()
df['appointment_month'] = df['appointmentday'].dt.month_name()
df['appointment_week'] = df['appointmentday'].dt.isocalendar().week
df['appointment_day'] = df['appointmentday'].dt.day

# Patient history
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

# Health features
df['has_chronic_condition'] = ((df['hypertension'] == 1) | (df['diabetes'] == 1)).astype(int)
df['total_conditions'] = df['hypertension'] + df['diabetes'] + df['alcoholism']

# Neighborhood features
neighborhood_stats = df.groupby('neighbourhood')['no_show'].mean().reset_index()
neighborhood_stats.columns = ['neighbourhood', 'neighborhood_noshow_rate']
df = df.merge(neighborhood_stats, on='neighbourhood', how='left')

df['neighborhood_risk'] = pd.cut(
    df['neighborhood_noshow_rate'],
    bins=[0, 0.18, 0.22, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

# Interaction features
df['young_long_lead'] = (
    (df['age_group'] == 'Young Adult') & 
    (df['lead_time_category'].isin(['15-30 days', '30+ days']))
).astype(int)

df['is_monday'] = (df['appointment_weekday'] == 'Monday').astype(int)
df['is_weekend'] = (df['appointment_weekday'].isin(['Saturday', 'Sunday'])).astype(int)

print(f"   ✅ Created {len(df.columns) - len(df_raw.columns)} new features")

features_path = PROCESSED_DIR / "appointments_features.csv"
df.to_csv(features_path, index=False)
print(f"   💾 Saved: {features_path}")

# ============================================================================
# STEP 4: RISK SCORING
# ============================================================================

print("\n🎯 STEP 4: Calculating risk scores...")

# Patient risk score
def score_patient_history(row):
    if row['patient_previous_appointments'] == 0:
        return 2.5
    rate = row['patient_historical_noshow_rate']
    if rate >= 0.5: return 5.0
    elif rate >= 0.3: return 4.0
    elif rate >= 0.15: return 3.0
    elif rate >= 0.05: return 2.0
    else: return 1.0

df['patient_risk_score'] = df.apply(score_patient_history, axis=1)

# Lead time risk
def score_lead_time(days):
    if days <= 0: return 1.0
    elif days <= 3: return 2.0
    elif days <= 7: return 3.0
    elif days <= 14: return 4.0
    else: return 5.0

df['lead_time_risk'] = df['lead_days'].apply(score_lead_time)

# Day risk
day_scores = {
    'Saturday': 1.0, 'Sunday': 1.5, 'Wednesday': 2.0, 'Thursday': 2.0,
    'Friday': 3.0, 'Tuesday': 3.5, 'Monday': 4.0
}
df['day_risk'] = df['appointment_weekday'].map(day_scores).fillna(2.5)

# Age risk
age_scores = {
    'Child': 2.0, 'Teen': 3.0, 'Young Adult': 4.0,
    'Adult': 2.5, 'Middle Age': 2.0, 'Senior': 1.5
}
df['age_risk'] = df['age_group'].map(age_scores).fillna(2.5)

# Neighborhood risk score
def score_neighborhood(rate):
    if rate > 0.25: return 5.0
    elif rate > 0.22: return 4.0
    elif rate > 0.20: return 3.0
    elif rate > 0.18: return 2.0
    else: return 1.0

df['neighborhood_risk_score'] = df['neighborhood_noshow_rate'].apply(score_neighborhood)

# Health risk
df['health_risk_score'] = df['has_chronic_condition'].apply(lambda x: 1.5 if x == 1 else 2.5)

# Socioeconomic risk
df['socio_risk_score'] = df['scholarship'].apply(lambda x: 3.5 if x == 1 else 2.0)

# SMS risk
df['sms_risk_score'] = df.apply(
    lambda row: 3.0 if row['sms_received'] == 1 and row['lead_days'] > 7
    else 4.0 if row['sms_received'] == 0 and row['lead_days'] > 7
    else 2.0,
    axis=1
)

# Composite risk score
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

df['risk_percentile'] = df['composite_risk_score'].rank(pct=True) * 100

# Risk tier
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

# Intervention recommendations
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

print(f"   ✅ Calculated risk scores")
print(f"   📊 Risk tier distribution:")
for tier in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']:
    count = (df['risk_tier'] == tier).sum()
    pct = count / len(df) * 100
    print(f"      {tier_display[tier]}: {count:,} ({pct:.1f}%)")

scored_path = PROCESSED_DIR / "appointments_scored.csv"
df.to_csv(scored_path, index=False)
print(f"   💾 Saved: {scored_path}")

# ============================================================================
# STEP 5: CREATE DASHBOARD DATA
# ============================================================================

print("\n📊 STEP 5: Creating dashboard-ready data...")

df_dashboard = df.copy()
df_dashboard['noshow_cost'] = df_dashboard['no_show'] * 150
df_dashboard['appointment_count'] = 1
df_dashboard['appointment_date'] = df_dashboard['appointmentday'].dt.date
df_dashboard['appointment_year_month'] = df_dashboard['appointmentday'].dt.to_period('M').astype(str)

for col in ['no_show', 'showed_up', 'sms_received', 'scholarship', 
            'hypertension', 'diabetes', 'phone_call_required', 'deposit_required']:
    if col in df_dashboard.columns:
        df_dashboard[f'{col}_label'] = df_dashboard[col].map({1: 'Yes', 0: 'No', True: 'Yes', False: 'No'})

dashboard_main_path = DASHBOARD_DIR / "appointments_dashboard.csv"
df_dashboard.to_csv(dashboard_main_path, index=False)
print(f"   💾 Saved: {dashboard_main_path}")

# ============================================================================
# STEP 6: CREATE SUMMARY TABLES
# ============================================================================

print("\n📈 STEP 6: Creating summary tables...")

# Daily summary
daily_summary = df_dashboard.groupby(df_dashboard['appointmentday'].dt.date).agg({
    'appointment_count': 'sum',
    'no_show': 'sum',
    'noshow_cost': 'sum',
    'composite_risk_score': 'mean'
}).reset_index()
daily_summary.columns = ['date', 'total_appointments', 'no_shows', 'cost', 'avg_risk_score']
daily_summary['noshow_rate'] = (daily_summary['no_shows'] / daily_summary['total_appointments'] * 100).round(2)
daily_summary.to_csv(DASHBOARD_DIR / "summary_daily.csv", index=False)
print(f"   💾 summary_daily.csv ({len(daily_summary)} rows)")

# Neighborhood summary
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
print(f"   💾 summary_neighborhood.csv ({len(neighborhood_summary)} rows)")

# Risk tier summary
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
print(f"   💾 summary_risk_tier.csv ({len(tier_summary)} rows)")

# Age group summary
age_summary = df_dashboard.groupby('age_group').agg({
    'appointment_count': 'sum',
    'no_show': ['sum', 'mean'],
    'noshow_cost': 'sum'
}).reset_index()
age_summary.columns = ['age_group', 'total_appointments', 'no_shows', 'noshow_rate', 'total_cost']
age_summary['noshow_rate'] = (age_summary['noshow_rate'] * 100).round(2)
age_summary.to_csv(DASHBOARD_DIR / "summary_age_group.csv", index=False)
print(f"   💾 summary_age_group.csv ({len(age_summary)} rows)")

# Day of week summary
dow_summary = df_dashboard.groupby('appointment_weekday').agg({
    'appointment_count': 'sum',
    'no_show': ['sum', 'mean'],
    'noshow_cost': 'sum'
}).reset_index()
dow_summary.columns = ['day_of_week', 'total_appointments', 'no_shows', 'noshow_rate', 'total_cost']
dow_summary['noshow_rate'] = (dow_summary['noshow_rate'] * 100).round(2)
dow_summary.to_csv(DASHBOARD_DIR / "summary_day_of_week.csv", index=False)
print(f"   💾 summary_day_of_week.csv ({len(dow_summary)} rows)")

# Lead time summary
lead_summary = df_dashboard.groupby('lead_time_category').agg({
    'appointment_count': 'sum',
    'no_show': ['sum', 'mean'],
    'noshow_cost': 'sum'
}).reset_index()
lead_summary.columns = ['lead_time_category', 'total_appointments', 'no_shows', 'noshow_rate', 'total_cost']
lead_summary['noshow_rate'] = (lead_summary['noshow_rate'] * 100).round(2)
lead_summary.to_csv(DASHBOARD_DIR / "summary_lead_time.csv", index=False)
print(f"   💾 summary_lead_time.csv ({len(lead_summary)} rows)")

# SMS summary
sms_summary = df_dashboard.groupby('sms_received').agg({
    'appointment_count': 'sum',
    'no_show': ['sum', 'mean'],
    'noshow_cost': 'sum'
}).reset_index()
sms_summary.columns = ['sms_received', 'total_appointments', 'no_shows', 'noshow_rate', 'total_cost']
sms_summary['noshow_rate'] = (sms_summary['noshow_rate'] * 100).round(2)
sms_summary['sms_status'] = sms_summary['sms_received'].map({0: 'No SMS', 1: 'SMS Sent'})
sms_summary.to_csv(DASHBOARD_DIR / "summary_sms.csv", index=False)
print(f"   💾 summary_sms.csv ({len(sms_summary)} rows)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("✅ DATA GENERATION COMPLETE!")
print("="*70)

print("\n📁 FILES CREATED:")
print("\n   📂 data/raw/")
print("      └── appointments_raw.csv")
print("\n   📂 data/processed/")
print("      ├── appointments_cleaned.csv")
print("      ├── appointments_features.csv")
print("      └── appointments_scored.csv")
print("\n   📂 data/dashboard/")
print("      ├── appointments_dashboard.csv (MAIN FILE FOR LOOKER)")
print("      ├── summary_daily.csv")
print("      ├── summary_neighborhood.csv")
print("      ├── summary_risk_tier.csv")
print("      ├── summary_age_group.csv")
print("      ├── summary_day_of_week.csv")
print("      ├── summary_lead_time.csv")
print("      └── summary_sms.csv")

print("\n📊 FINAL STATISTICS:")
print(f"   Total appointments: {len(df):,}")
print(f"   Unique patients: {df['patientid'].nunique():,}")
print(f"   Date range: {df['appointmentday'].min().date()} to {df['appointmentday'].max().date()}")
print(f"   No-show rate: {df['no_show'].mean()*100:.1f}%")
print(f"   Total no-shows: {df['no_show'].sum():,}")
print(f"   Financial impact: ${df['no_show'].sum() * 150:,.0f}")

print("\n🚀 NEXT STEPS:")
print("   1. Upload 'data/dashboard/appointments_dashboard.csv' to Google Sheets")
print("   2. Connect to Looker Studio")
print("   3. Build your dashboard!")

print("\n" + "="*70)
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)
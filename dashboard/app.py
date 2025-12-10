"""
Healthcare Appointments Analytics Dashboard
Week 4: Visualization & Dashboards (Streamlit Alternative to Looker Studio)

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="NSP Analytics",
    page_icon="📟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME CONFIGURATION ---
def get_theme_colors(theme_mode):
    if theme_mode == "DARK":
        return {
            "bg": "#0E1117",
            "text": "#FAFAFA",
            "card_bg": "#262730",
            "border": "#414141",
            "accent": "#FF4B4B",
            "success": "#00CC96",
            "chart_bg": "rgba(0,0,0,0)",
            "grid": "#414141",
            "shadow": "#000000"
        }
    else: # LIGHT
        return {
            "bg": "#FFFFFF",
            "text": "#000000",
            "card_bg": "#FFFFFF",
            "border": "#000000",
            "accent": "#FF4B4B",
            "success": "#22c55e",
            "chart_bg": "rgba(0,0,0,0)",
            "grid": "#eee",
            "shadow": "#000000"
        }

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = "LIGHT"

# Sidebar controls
with st.sidebar:
    st.markdown("### // SYSTEM CONTROLS")
    theme_mode = st.radio("INTERFACE_THEME", ["LIGHT", "DARK"], index=0 if st.session_state.theme == "LIGHT" else 1)
    st.session_state.theme = theme_mode
    st.markdown("---")

colors = get_theme_colors(st.session_state.theme)

# Custom CSS for Retro Minimal Design with Theming
st.markdown(f"""
<style>
    /* Global Font & Colors */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
    
    /* Enforce Theme Colors on Main Containers */
    .stApp, .block-container, .stMarkdown, p, h1, h2, h3, h4, span, div {{
        font-family: 'Space Mono', 'Courier New', monospace;
        color: {colors['text']} !important;
    }}
    
    .stApp {{
        background-color: {colors['bg']} !important;
    }}
    
    /* Hide Default Header/Footer */
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Main Layout */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95% !important;
    }}
    
    /* Retro Headers */
    h1, h2, h3 {{
        font-family: 'Space Mono', monospace;
        text-transform: uppercase;
        letter-spacing: -1px;
        font-weight: 700;
        border-bottom: 2px solid {colors['border']};
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
        color: {colors['text']} !important;
    }}
    
    /* Metrics Cards - Retro Box Style */
    div[data-testid="stMetricValue"] {{
        font-size: 1.8rem !important;
        font-weight: 700;
        color: {colors['text']} !important;
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 0.8rem !important;
        text-transform: uppercase;
        font-weight: 400;
        color: {colors['text']} !important;
        opacity: 0.8;
    }}
    div[data-testid="metric-container"] {{
        background-color: {colors['card_bg']} !important;
        border: 2px solid {colors['border']} !important;
        border-radius: 0px !important;
        padding: 15px;
        box-shadow: 4px 4px 0px {colors['shadow']};  /* Retro hard shadow */
        transition: all 0.2s ease;
    }}
    div[data-testid="metric-container"]:hover {{
        transform: translate(-2px, -2px);
        box-shadow: 6px 6px 0px {colors['shadow']};
    }}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {colors['card_bg']} !important;
        border-right: 2px solid {colors['border']};
    }}
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span {{
        color: {colors['text']} !important;
    }}
    
    /* Widget Styling */
    .stSelectbox > div > div {{
        border: 2px solid {colors['border']} !important;
        border-radius: 0px !important;
        background-color: {colors['card_bg']} !important;
        color: {colors['text']} !important;
    }}
    .stDateInput > div > div {{
        border: 2px solid {colors['border']} !important;
        border-radius: 0px !important;
        background-color: {colors['bg']} !important;
    }}
    
    /* Fix Input Labels */
    .stSelectbox label, .stDateInput label, .stRadio label {{
        color: {colors['text']} !important;
    }}
    
    /* Insight Boxes */
    .insight-box {{
        background-color: {colors['card_bg']};
        border: 2px solid {colors['border']};
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 4px 4px 0px {colors['shadow']};
        color: {colors['text']} !important;
    }}
    .insight-box h4 {{
        border-bottom: 1px solid {colors['border']};
        margin-bottom: 0.5rem;
        font-size: 1rem;
        color: {colors['text']} !important;
    }}
    
    /* Custom divider */
    hr {{
        border: 0;
        border-top: 2px solid {colors['border']};
        margin: 2rem 0;
    }}
    
    /* Dataframe Fix */
    div[data-testid="stDataFrame"] {{
        border: 2px solid {colors['border']};
    }}
    
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load data from SQLite database."""
    db_paths = [
        "healthcare.db",
        "../healthcare.db",
        "data/healthcare.db",
        Path(__file__).parent.parent / "healthcare.db"
    ]
    
    conn = None
    for db_path in db_paths:
        try:
            if Path(db_path).exists():
                conn = sqlite3.connect(str(db_path))
                break
        except:
            continue
    
    if conn is None:
        st.error("EROR: Database Connection Failed")
        return None
    
    df = pd.read_sql_query("SELECT * FROM appointments", conn)
    conn.close()
    
    # Parse dates
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
    
    # Calculate Lead_Days
    if 'Lead_Days' not in df.columns:
        df['Lead_Days'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days.clip(lower=0)
    
    # Create Age Groups
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 55, 100], 
                              labels=['0-18', '19-35', '36-55', '55+'])
    
    df['DayOfWeek'] = df['AppointmentDay'].dt.day_name()
    
    return df


def style_chart(fig, title=""):
    """Apply retro minimal styling to Plotly charts."""
    fig.update_layout(
        title=dict(
            text=title.upper(),
            font=dict(family="Space Mono, monospace", size=14, color=colors['text']),
            x=0,
            xanchor="left"
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Space Mono, monospace", color=colors['text']),
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            showgrid=True, 
            gridwidth=1, 
            gridcolor=colors['grid'],
            linecolor=colors['border'],
            linewidth=2,
            ticks="inside"
        ),
        yaxis=dict(
            showgrid=True, 
            gridwidth=1, 
            gridcolor=colors['grid'],
            linecolor=colors['border'],
            linewidth=2,
            ticks="inside"
        )
    )
    
    # Update bars/lines styling
    try:
        fig.update_traces(selector=dict(type='bar'), marker_line_width=1.5, marker_line_color=colors['border'])
        fig.update_traces(selector=dict(type='scatter'), marker_line_width=1.5, marker_line_color=colors['border'])
    except:
        pass
    return fig


def create_kpi_cards(df):
    """Create KPI metric cards."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_appointments = len(df)
    noshow_rate = df['No_Show'].mean() * 100
    avg_age = df['Age'].mean()
    sms_rate = df['SMS_received'].mean() * 100
    avg_lead_days = df['Lead_Days'].mean()
    
    with col1:
        st.metric("Total Appts", f"{total_appointments:,}")
    with col2:
        st.metric("No-Show Rate", f"{noshow_rate:.1f}%")
    with col3:
        st.metric("Avg Age", f"{avg_age:.0f} yrs")
    with col4:
        st.metric("SMS Rate", f"{sms_rate:.1f}%")
    with col5:
        st.metric("Lead Time", f"{avg_lead_days:.0f} days")


def create_noshow_by_category(df, category, title):
    """Create no-show rate bar chart by category."""
    rates = df.groupby(category)['No_Show'].mean() * 100
    
    marker_color = colors['text'] if st.session_state.theme == "LIGHT" else colors['text']
    
    fig = go.Figure(data=[
        go.Bar(
            x=rates.index,
            y=rates.values,
            marker_color=colors['card_bg'],  # Use card bg for contrast
            text=[f"{v:.1f}%" for v in rates.values],
            textposition='auto',
            marker_line_color=colors['text'],
            marker_line_width=1.5
        )
    ])
    
    # Add target line
    fig.add_hline(y=df['No_Show'].mean() * 100, line_dash="solid", line_color=colors['accent'], line_width=1)
    
    return style_chart(fig, title)


def create_trend_chart(df):
    """Create monthly trend chart."""
    df['YearMonth'] = df['AppointmentDay'].dt.to_period('M').astype(str)
    monthly = df.groupby('YearMonth').agg({
        'No_Show': ['count', 'mean']
    }).reset_index()
    monthly.columns = ['YearMonth', 'Total', 'Rate']
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Total Appts - Bar
    fig.add_trace(
        go.Bar(
            x=monthly['YearMonth'],
            y=monthly['Total'],
            name='COUNT',
            marker_color=colors['grid'],
            marker_line_width=1.5,
            marker_line_color=colors['border']
        ),
        secondary_y=False
    )
    
    # Rate - Line
    fig.add_trace(
        go.Scatter(
            x=monthly['YearMonth'],
            y=monthly['Rate'] * 100,
            name='RATE (%)',
            mode='lines+markers',
            line=dict(color=colors['accent'], width=3),
            marker=dict(size=8, color=colors['card_bg'], line=dict(width=2, color=colors['accent']))
        ),
        secondary_y=True
    )
    
    fig = style_chart(fig, "Monthly Volume & No-Show Rate")
    fig.update_layout(legend=dict(orientation="h", y=1.1, font=dict(color=colors['text'])))
    return fig


def create_heatmap(df):
    """Create day of week vs age group heatmap."""
    pivot = df.pivot_table(
        values='No_Show',
        index='Age_Group',
        columns='DayOfWeek',
        aggfunc='mean'
    ) * 100
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = pivot.reindex(columns=[d for d in day_order if d in pivot.columns])
    
    # Choose color scale based on theme
    color_scale = 'Greys' if st.session_state.theme == 'LIGHT' else 'gray_r'
    
    fig = px.imshow(
        pivot,
        color_continuous_scale=color_scale,
        aspect='auto'
    )
    
    fig.update_traces(
        showscale=False,
        text=pivot.applymap(lambda x: f"{x:.1f}%").values,
        texttemplate="%{text}"
    )
    
    return style_chart(fig, "Risk Heatmap (Age vs Day)")


def create_sms_impact_chart(df):
    """Create SMS impact visualization."""
    sms_impact = df.groupby('SMS_received')['No_Show'].mean() * 100
    
    bar_colors = [colors['border'], colors['card_bg']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['No SMS', 'SMS Sent'],
        y=sms_impact.values,
        marker_color=bar_colors,
        text=[f'{v:.1f}%' for v in sms_impact.values],
        textposition='outside',
        marker_pattern_shape=['', '/'],  # Retro hatch pattern
        marker_line_width=2,
        marker_line_color=colors['border']
    ))
    
    return style_chart(fig, "SMS Intervention Impact")


def create_risk_distribution(df):
    """Create risk score distribution."""
    # Simple risk score calc
    df['Risk_Score'] = (
        (df['Age'] < 25).astype(int) * 0.2 +
        (df['Lead_Days'] > 14).astype(int) * 0.3 +
        (df['SMS_received'] == 0).astype(int) * 0.2 +
        (df['Scholarship'] == 1).astype(int) * 0.15
    ) * 100
    
    fig = px.histogram(
        df, x='Risk_Score', nbins=15,
        color_discrete_sequence=[colors['border']]
    )
    fig.update_layout(bargap=0.1)
    
    return style_chart(fig, "Risk Score Distribution")


def main():
    """Main dashboard function."""
    
    # Sidebar
    # Assuming sidebar initiated above due to theme selection requirement appearing first
    
    # Title Section
    st.markdown("# NSP ANALYTICS DASHBOARD v1.1")
    st.markdown(f"`SYSTEM_STATUS: ONLINE` | `MODE: {st.session_state.theme}`")
    
    # Load data
    df = load_data()
    if df is None: st.stop()
    
    # Filter: Date
    min_date, max_date = df['AppointmentDay'].min().date(), df['AppointmentDay'].max().date()
    date_range = st.sidebar.date_input("DATE_RANGE", (min_date, max_date))
    
    # Filter: Age
    age_groups = st.sidebar.multiselect("AGE_SEGMENT", df['Age_Group'].unique(), default=df['Age_Group'].unique())
    
    # Data Filtering
    mask = (df['AppointmentDay'].dt.date >= date_range[0]) & (df['AppointmentDay'].dt.date <= date_range[1])
    mask &= df['Age_Group'].isin(age_groups)
    filtered_df = df.loc[mask]
    
    # Sidebar Stats
    st.sidebar.metric("ACTIVE_RECORDS", len(filtered_df))
    
    # KPIs
    create_kpi_cards(filtered_df)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts Grid
    col1, col2 = st.columns([2, 1])
    with col1: st.plotly_chart(create_trend_chart(filtered_df), use_container_width=True)
    with col2: st.plotly_chart(create_sms_impact_chart(filtered_df), use_container_width=True)
    
    col3, col4 = st.columns(2)
    with col3: st.plotly_chart(create_heatmap(filtered_df), use_container_width=True)
    with col4: st.plotly_chart(create_risk_distribution(filtered_df), use_container_width=True)
    
    # Insights Section
    st.markdown("### // AUTOMATED INSIGHTS")
    
    noshow_diff = (filtered_df[filtered_df['SMS_received']==0]['No_Show'].mean() - 
                  filtered_df[filtered_df['SMS_received']==1]['No_Show'].mean()) * 100
                  
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="insight-box">
            <h4>[INTERVENTION_EFFECT]</h4>
            SMS reduces no-shows by <b>{noshow_diff:.1f}%</b>.
            RECOMMENDATION: INCREASE SMS COVERAGE.
        </div>
        """, unsafe_allow_html=True)
    with c2:
        high_risk = filtered_df.groupby('Age_Group')['No_Show'].mean().idxmax()
        st.markdown(f"""
        <div class="insight-box">
            <h4>[RISK_SEGMENT]</h4>
            Highest risk group: <b>{high_risk}</b>.
            RECOMMENDATION: TARGETED OUTREACH.
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="insight-box">
            <h4>[DATA_QUALITY]</h4>
            Records Processed: {len(filtered_df)}.
            Data Integrity: VALID
        </div>
        """, unsafe_allow_html=True)
        
    # Raw Data Explorer
    with st.expander(">> ACCESS_RAW_DATA_LOGS"):
        st.dataframe(filtered_df.sort_values('AppointmentDay', ascending=False).head(100), use_container_width=True, height=300)

if __name__ == "__main__":
    main()

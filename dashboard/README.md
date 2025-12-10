# Healthcare Analytics Dashboard

Interactive Streamlit dashboard for healthcare appointment no-show analytics.

## Features

- 📊 **KPI Cards**: Total appointments, no-show rate, average age, SMS rate, lead time
- 🔍 **Interactive Filters**: Date range, age groups, SMS status, neighborhoods
- 📈 **Visualizations**:
  - Monthly appointment trends with no-show rate overlay
  - SMS impact comparison
  - No-show rate by age group and day of week
  - Heatmap (Age Group × Day of Week)
  - Risk score distribution
- 💡 **Key Insights**: Automated recommendations

## Quick Start

```bash
# Install dependencies
pip install -r dashboard/requirements.txt

# Run dashboard
streamlit run dashboard/app.py
```

The dashboard will open at http://localhost:8501

## Screenshots

The dashboard includes:
1. Top KPI metrics bar
2. Interactive sidebar filters
3. 6 interactive Plotly charts
4. Insights and recommendations section
5. Raw data explorer

## Week 4 Deliverable

This dashboard serves as the **Week 4: Visualization & Dashboards** deliverable,
providing an alternative to Google Looker Studio using Python-based tools.

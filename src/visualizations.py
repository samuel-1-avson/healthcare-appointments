"""
Visualization Module
====================
Functions to create charts and dashboards for the analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple, Union
import logging
from pathlib import Path

from .utils import timer


class Visualizer:
    """Create visualizations for appointment analysis."""
    
    def __init__(self, config: dict):
        """
        Initialize Visualizer with configuration.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("healthcare_pipeline.Visualizer")
        self.figures_created = []
        
        # Set style
        plt.style.use(self.config['visualization'].get('style', 'seaborn-v0_8-darkgrid'))
        sns.set_palette(self.config['visualization'].get('color_palette', 'husl'))
        
        # Create output directory
        self.figures_dir = Path(self.config['paths']['figures_dir'])
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def save_figure(self, fig: plt.Figure, filename: str) -> None:
        """
        Save figure to outputs folder.
        
        Parameters
        ----------
        fig : plt.Figure
            Figure to save
        filename : str
            Filename (without path)
        """
        filepath = self.figures_dir / filename
        dpi = self.config['visualization'].get('dpi', 150)
        
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        self.logger.info(f"Saved figure to {filepath}")
        self.figures_created.append(str(filepath))
    
    @timer
    def plot_noshow_rate_by_category(
        self, 
        df: pd.DataFrame, 
        category: str,
        title: Optional[str] = None,
        save: bool = True
    ) -> plt.Figure:
        """
        Bar chart of no-show rate by category.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data with no_show and category column
        category : str
            Column name to group by
        title : str, optional
            Chart title
        save : bool
            Whether to save the figure
        
        Returns
        -------
        plt.Figure
            The figure object
        """
        if category not in df.columns:
            self.logger.warning(f"Column {category} not found")
            return None
        
        # Calculate no-show rates
        rates = df.groupby(category)['no_show'].agg(['mean', 'count'])
        rates['rate_pct'] = rates['mean'] * 100
        rates = rates.sort_values('rate_pct', ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config['visualization']['figure_size'])
        
        # Color based on rate
        colors = ['#e74c3c' if x > 25 else '#f39c12' if x > 20 else '#2ecc71' 
                  for x in rates['rate_pct']]
        
        # Plot bars
        bars = ax.bar(range(len(rates)), rates['rate_pct'], color=colors)
        
        # Customize
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels(rates.index, rotation=45, ha='right')
        ax.set_ylabel('No-Show Rate (%)')
        ax.set_title(title or f'No-Show Rate by {category}', fontweight='bold')
        ax.axhline(y=20.2, color='gray', linestyle='--', alpha=0.5, label='Overall Average')
        
        # Add value labels
        for bar, rate in zip(bars, rates['rate_pct']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{rate:.1f}%', ha='center', fontsize=9)
        
        # Add count as subtitle
        for i, (idx, row) in enumerate(rates.iterrows()):
            ax.text(i, -2, f'n={row["count"]:,}', ha='center', fontsize=8, color='gray')
        
        plt.tight_layout()
        
        if save:
            filename = f'noshow_by_{category.lower().replace(" ", "_")}.png'
            self.save_figure(fig, filename)
        
        return fig
    
    def plot_risk_distribution(
        self,
        df: pd.DataFrame,
        save: bool = True
    ) -> plt.Figure:
        """
        Plot distribution of risk scores and tiers.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with risk scores
        save : bool
            Whether to save the figure
        
        Returns
        -------
        plt.Figure
            The figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Risk score distribution
        ax1 = axes[0, 0]
        if 'composite_risk_score' in df.columns:
            ax1.hist(df['composite_risk_score'], bins=50, edgecolor='black', alpha=0.7)
            ax1.axvline(df['composite_risk_score'].mean(), color='red', 
                       linestyle='--', label=f'Mean: {df["composite_risk_score"].mean():.2f}')
            ax1.set_xlabel('Composite Risk Score')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Risk Score Distribution', fontweight='bold')
            ax1.legend()
        
        # 2. Risk tier pie chart
        ax2 = axes[0, 1]
        if 'risk_tier' in df.columns:
            tier_counts = df['risk_tier'].value_counts()
            tier_colors = {
                'CRITICAL': '#e74c3c',
                'HIGH': '#e67e22',
                'MEDIUM': '#f1c40f',
                'LOW': '#2ecc71',
                'MINIMAL': '#3498db'
            }
            colors = [tier_colors.get(tier, '#95a5a6') for tier in tier_counts.index]
            
            wedges, texts, autotexts = ax2.pie(
                tier_counts.values,
                labels=tier_counts.index,
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            ax2.set_title('Risk Tier Distribution', fontweight='bold')
        
        # 3. Risk score by actual no-show
        ax3 = axes[1, 0]
        if 'composite_risk_score' in df.columns and 'no_show' in df.columns:
            df.boxplot(column='composite_risk_score', by='no_show', ax=ax3)
            ax3.set_xlabel('No-Show Status')
            ax3.set_ylabel('Risk Score')
            ax3.set_title('Risk Score by Actual No-Show', fontweight='bold')
            ax3.set_xticklabels(['Showed Up', 'No-Show'])
            plt.sca(ax3)
            plt.xticks([1, 2], ['Showed Up', 'No-Show'])
        
        # 4. No-show rate by risk tier
        ax4 = axes[1, 1]
        if 'risk_tier' in df.columns and 'no_show' in df.columns:
            tier_rates = df.groupby('risk_tier')['no_show'].mean() * 100
            tier_order = ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            tier_rates = tier_rates.reindex(tier_order)
            
            bars = ax4.bar(range(len(tier_rates)), tier_rates.values, 
                          color=[tier_colors[tier] for tier in tier_order])
            ax4.set_xticks(range(len(tier_rates)))
            ax4.set_xticklabels(tier_order, rotation=45)
            ax4.set_ylabel('No-Show Rate (%)')
            ax4.set_title('Actual No-Show Rate by Risk Tier', fontweight='bold')
            
            for bar, rate in zip(bars, tier_rates.values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{rate:.1f}%', ha='center', fontsize=9)
        
        plt.suptitle('Risk Analysis Dashboard', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            self.save_figure(fig, 'risk_analysis_dashboard.png')
        
        return fig
    
    def plot_trend_over_time(
        self,
        df: pd.DataFrame,
        save: bool = True
    ) -> plt.Figure:
        """
        Plot no-show trend over time.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with appointment dates
        save : bool
            Whether to save the figure
        
        Returns
        -------
        plt.Figure
            The figure object
        """
        # Find date column
        date_col = None
        for col in df.columns:
            if 'appointment' in col.lower() and 'day' in col.lower():
                date_col = col
                break
        
        if date_col is None or 'no_show' not in df.columns:
            self.logger.warning("Required columns for trend analysis not found")
            return None
        
        # Ensure datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Daily trend
        daily_trend = df.groupby(df[date_col].dt.date)['no_show'].agg(['mean', 'count'])
        daily_trend['rate_pct'] = daily_trend['mean'] * 100
        
        # Weekly trend
        df['week'] = df[date_col].dt.to_period('W')
        weekly_trend = df.groupby('week')['no_show'].agg(['mean', 'count'])
        weekly_trend['rate_pct'] = weekly_trend['mean'] * 100
        weekly_trend.index = weekly_trend.index.to_timestamp()
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Daily trend
        ax1 = axes[0]
        ax1.plot(daily_trend.index, daily_trend['rate_pct'], 
                marker='o', markersize=4, linewidth=1, alpha=0.7)
        ax1.axhline(y=20.2, color='red', linestyle='--', alpha=0.5, label='Overall Average')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('No-Show Rate (%)')
        ax1.set_title('Daily No-Show Rate Trend', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Weekly trend with volume
        ax2 = axes[0]
        ax2_twin = ax2.twinx()
        
        ax2.bar(weekly_trend.index, weekly_trend['count'], alpha=0.3, color='gray', label='Volume')
        ax2_twin.plot(weekly_trend.index, weekly_trend['rate_pct'], 
                     color='red', marker='o', linewidth=2, label='No-Show Rate')
        
        ax2.set_xlabel('Week')
        ax2.set_ylabel('Number of Appointments', color='gray')
        ax2_twin.set_ylabel('No-Show Rate (%)', color='red')
        ax2.set_title('Weekly Trend with Volume', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        
        # Add legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        if save:
            self.save_figure(fig, 'noshow_trend_analysis.png')
        
        return fig
    
    def plot_factor_importance(
        self,
        importance_dict: Optional[Dict] = None,
        df: Optional[pd.DataFrame] = None,
        save: bool = True
    ) -> plt.Figure:
        """
        Horizontal bar chart of factor importance.
        
        Parameters
        ----------
        importance_dict : dict, optional
            Dictionary of factor: importance
        df : pd.DataFrame, optional
            DataFrame to calculate importance from
        save : bool
            Whether to save the figure
        
        Returns
        -------
        plt.Figure
            The figure object
        """
        # If no importance dict provided, calculate from data
        if importance_dict is None and df is not None:
            importance_dict = self._calculate_factor_importance(df)
        
        if importance_dict is None:
            self.logger.warning("No importance data available")
            return None
        
        # Sort by importance
        sorted_factors = dict(sorted(importance_dict.items(), 
                                   key=lambda x: x[1], reverse=True))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot horizontal bars
        y_pos = np.arange(len(sorted_factors))
        bars = ax.barh(y_pos, list(sorted_factors.values()))
        
        # Color gradient
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(sorted_factors)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(sorted_factors.keys()))
        ax.set_xlabel('Importance Score')
        ax.set_title('Risk Factor Importance', fontweight='bold', fontsize=14)
        
        # Add value labels
        for i, v in enumerate(sorted_factors.values()):
            ax.text(v + 0.1, i, f'{v:.2f}', va='center')
        
        plt.tight_layout()
        
        if save:
            self.save_figure(fig, 'factor_importance.png')
        
        return fig
    
    def _calculate_factor_importance(self, df: pd.DataFrame) -> Dict:
        """Calculate factor importance from data."""
        importance = {}
        
        # Check correlation with no-show for available features
        if 'no_show' in df.columns:
            # Numeric features
            numeric_features = [
                'lead_days', 'age', 'patient_historical_noshow_rate',
                'composite_risk_score', 'patient_total_appointments'
            ]
            for feature in numeric_features:
                if feature in df.columns:
                    corr = df[feature].corr(df['no_show'])
                    importance[feature] = abs(corr)
            
            # Categorical features (using variance ratio)
            categorical_features = [
                'age_group', 'appointment_weekday', 'risk_tier',
                'lead_time_category', 'neighborhood_risk'
            ]
            for feature in categorical_features:
                if feature in df.columns:
                    # Calculate variance ratio
                    group_means = df.groupby(feature)['no_show'].mean()
                    variance_ratio = group_means.std()
                    importance[feature] = variance_ratio
        
        return importance
    
    @timer
    def create_summary_dashboard(
        self,
        df: pd.DataFrame,
        save: bool = True
    ) -> plt.Figure:
        """
        Create comprehensive summary dashboard.
        
        Parameters
        ----------
        df : pd.DataFrame
            Complete DataFrame with all features
        save : bool
            Whether to save the figure
        
        Returns
        -------
        plt.Figure
            The figure object
        """
        self.logger.info("Creating summary dashboard...")
        
        # Create large figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Define grid
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Overall no-show rate (pie)
        ax1 = fig.add_subplot(gs[0, 0])
        if 'no_show' in df.columns:
            no_show_counts = df['no_show'].value_counts()
            colors = ['#2ecc71', '#e74c3c']
            ax1.pie(no_show_counts.values, labels=['Showed Up', 'No-Show'],
                   autopct='%1.1f%%', colors=colors, startangle=90)
            ax1.set_title('Overall Attendance', fontweight='bold')
        
        # 2. Risk tier distribution (pie)
        ax2 = fig.add_subplot(gs[0, 1])
        if 'risk_tier' in df.columns:
            tier_counts = df['risk_tier'].value_counts()
            tier_colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']
            ax2.pie(tier_counts.values, labels=tier_counts.index,
                   autopct='%1.0f%%', colors=tier_colors[:len(tier_counts)],
                   startangle=90)
            ax2.set_title('Risk Tier Distribution', fontweight='bold')
        
        # 3. No-show by day of week
        ax3 = fig.add_subplot(gs[0, 2])
        if 'appointment_weekday' in df.columns:
            dow_rates = df.groupby('appointment_weekday')['no_show'].mean() * 100
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            dow_rates = dow_rates.reindex(day_order)
            
            bars = ax3.bar(range(len(dow_rates)), dow_rates.values)
            ax3.set_xticks(range(len(dow_rates)))
            ax3.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])
            ax3.set_ylabel('No-Show Rate (%)')
            ax3.set_title('By Day of Week', fontweight='bold')
            ax3.axhline(y=20.2, color='red', linestyle='--', alpha=0.3)
        
        # 4. No-show by age group
        ax4 = fig.add_subplot(gs[0, 3])
        if 'age_group' in df.columns:
            age_rates = df.groupby('age_group')['no_show'].mean() * 100
            age_order = ['Child', 'Teen', 'Young Adult', 'Adult', 'Middle Age', 'Senior']
            age_rates = age_rates.reindex(age_order)
            
            bars = ax4.bar(range(len(age_rates)), age_rates.values, color='#3498db')
            ax4.set_xticks(range(len(age_rates)))
            ax4.set_xticklabels(age_order, rotation=45, ha='right')
            ax4.set_ylabel('No-Show Rate (%)')
            ax4.set_title('By Age Group', fontweight='bold')
        
        # 5. Lead time impact
        ax5 = fig.add_subplot(gs[1, 0:2])
        if 'lead_time_category' in df.columns:
            lead_rates = df.groupby('lead_time_category')['no_show'].mean() * 100
            lead_order = ['Same Day', '1-7 days', '8-14 days', '15-30 days', '30+ days']
            lead_rates = lead_rates.reindex(lead_order)
            
            bars = ax5.barh(range(len(lead_rates)), lead_rates.values)
            ax5.set_yticks(range(len(lead_rates)))
            ax5.set_yticklabels(lead_order)
            ax5.set_xlabel('No-Show Rate (%)')
            ax5.set_title('Lead Time Impact', fontweight='bold')
            
            for bar, rate in zip(bars, lead_rates.values):
                ax5.text(rate + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{rate:.1f}%', va='center')
        
        # 6. Risk score distribution
        ax6 = fig.add_subplot(gs[1, 2:4])
        if 'composite_risk_score' in df.columns:
            ax6.hist(df['composite_risk_score'], bins=50, edgecolor='black', alpha=0.7)
            ax6.axvline(df['composite_risk_score'].mean(), color='red',
                       linestyle='--', label=f'Mean: {df["composite_risk_score"].mean():.2f}')
            ax6.set_xlabel('Composite Risk Score')
            ax6.set_ylabel('Frequency')
            ax6.set_title('Risk Score Distribution', fontweight='bold')
            ax6.legend()
        
        # 7. SMS effectiveness
        ax7 = fig.add_subplot(gs[2, 0])
        if 'sms_received' in df.columns:
            sms_rates = df.groupby('sms_received')['no_show'].mean() * 100
            labels = ['No SMS', 'SMS Sent']
            colors_sms = ['#2ecc71', '#e74c3c']
            
            bars = ax7.bar(labels, sms_rates.values, color=colors_sms)
            ax7.set_ylabel('No-Show Rate (%)')
            ax7.set_title('SMS Effectiveness (?)', fontweight='bold')
            
            for bar, rate in zip(bars, sms_rates.values):
                ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{rate:.1f}%', ha='center', fontweight='bold')
        
        # 8. Patient history impact
        ax8 = fig.add_subplot(gs[2, 1])
        if 'is_first_appointment' in df.columns:
            first_appt_rates = df.groupby('is_first_appointment')['no_show'].mean() * 100
            labels = ['Return Patient', 'First Visit']
            
            bars = ax8.bar(labels, first_appt_rates.values, color='#9b59b6')
            ax8.set_ylabel('No-Show Rate (%)')
            ax8.set_title('First vs Return Patients', fontweight='bold')
            
            for bar, rate in zip(bars, first_appt_rates.values):
                ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{rate:.1f}%', ha='center')
        
        # 9. Chronic conditions
        ax9 = fig.add_subplot(gs[2, 2])
        if 'has_chronic_condition' in df.columns:
            chronic_rates = df.groupby('has_chronic_condition')['no_show'].mean() * 100
            labels = ['No Chronic', 'Has Chronic']
            colors_chronic = ['#e74c3c', '#2ecc71']
            
            bars = ax9.bar(labels, chronic_rates.values, color=colors_chronic)
            ax9.set_ylabel('No-Show Rate (%)')
            ax9.set_title('Chronic Conditions', fontweight='bold')
            
            for bar, rate in zip(bars, chronic_rates.values):
                ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{rate:.1f}%', ha='center')
        
        # 10. Intervention needs
        ax10 = fig.add_subplot(gs[2, 3])
        if 'risk_tier' in df.columns:
            intervention_data = {
                'Phone Calls': df[df['phone_call_required'] == True].shape[0] if 'phone_call_required' in df.columns else 0,
                'Deposits': df[df['deposit_required'] == True].shape[0] if 'deposit_required' in df.columns else 0,
                'Double SMS': df[df['sms_reminders_needed'] >= 2].shape[0] if 'sms_reminders_needed' in df.columns else 0
            }
            
            bars = ax10.bar(intervention_data.keys(), intervention_data.values(), color='#e67e22')
            ax10.set_ylabel('Number of Appointments')
            ax10.set_title('Interventions Needed', fontweight='bold')
            ax10.tick_params(axis='x', rotation=45)
        
        # 11. Key metrics text
        ax11 = fig.add_subplot(gs[3, :2])
        ax11.axis('off')
        
        # Calculate metrics
        total_appts = len(df)
        no_show_rate = df['no_show'].mean() * 100 if 'no_show' in df.columns else 0
        high_risk = df[df['risk_tier'].isin(['HIGH', 'CRITICAL'])].shape[0] if 'risk_tier' in df.columns else 0
        cost_impact = df['no_show'].sum() * 150 if 'no_show' in df.columns else 0
        
        metrics_text = f"""
        KEY METRICS SUMMARY
        ═══════════════════════════════════
        
        📊 Total Appointments: {total_appts:,}
        📉 No-Show Rate: {no_show_rate:.1f}%
        🚨 High/Critical Risk: {high_risk:,} ({high_risk/total_appts*100:.1f}%)
        💰 Financial Impact: ${cost_impact:,.0f}
        
        TOP RISK FACTORS:
        • Lead time >14 days
        • Young adults (18-35)
        • Monday appointments
        • First-time patients
        """
        
        ax11.text(0.05, 0.95, metrics_text, transform=ax11.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 12. Recommendations text
        ax12 = fig.add_subplot(gs[3, 2:])
        ax12.axis('off')
        
        recommendations_text = """
        RECOMMENDED INTERVENTIONS
        ═══════════════════════════════════
        
        IMMEDIATE ACTIONS:
        ✓ Cap booking at 14 days maximum
        ✓ Send SMS to ALL patients
        ✓ Expand same-day capacity 20%
        
        HIGH-RISK PROTOCOLS:
        ✓ Phone confirmation 48hrs before
        ✓ Overbook by 15-20%
        ✓ Require deposits for chronic no-shows
        
        EXPECTED OUTCOME:
        • Reduce no-show rate to 15%
        • Recover $3.3M annually
        • Improve patient access
        """
        
        ax12.text(0.05, 0.95, recommendations_text, transform=ax12.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
        
        # Main title
        fig.suptitle('Healthcare Appointments No-Show Analysis Dashboard',
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        if save:
            self.save_figure(fig, 'summary_dashboard.png')
        
        self.logger.info(f"Created summary dashboard with {len(fig.axes)} panels")
        
        return fig
    
    def get_figures_created(self) -> List[str]:
        """
        Get list of figures created.
        
        Returns
        -------
        list
            List of figure filenames created
        """
        return self.figures_created
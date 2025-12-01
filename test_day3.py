# test_day3.py
"""Test script for Day 3 modules."""

import pandas as pd
import matplotlib.pyplot as plt
from src.utils import load_config, setup_logging
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.feature_engineer import FeatureEngineer
from src.risk_scorer import RiskScorer
from src.visualizations import Visualizer


def main():
    """Test the complete pipeline including risk scoring and visualizations."""
    
    print("\n" + "="*60)
    print("TESTING DAY 3: RISK SCORING & VISUALIZATIONS")
    print("="*60)
    
    # Setup
    config = load_config("config/config.yaml")
    logger = setup_logging(config['logging']['level'])
    
    # Load and process data
    print("\n1. LOADING AND PROCESSING DATA...")
    loader = DataLoader(config)
    df = loader.load(source="auto")
    
    cleaner = DataCleaner(config)
    df = cleaner.clean_pipeline(df)
    
    engineer = FeatureEngineer(config)
    df = engineer.engineer_all_features(df)
    print(f"   ✅ Data ready: {df.shape}")
    
    # Risk scoring
    print("\n2. CALCULATING RISK SCORES...")
    scorer = RiskScorer(config)
    df = scorer.score_pipeline(df)
    
    # Display risk summary
    risk_summary = scorer.get_risk_summary(df)
    print("\n   Risk Score Statistics:")
    print(f"   • Min: {risk_summary['risk_score_stats']['min']:.2f}")
    print(f"   • Max: {risk_summary['risk_score_stats']['max']:.2f}")
    print(f"   • Mean: {risk_summary['risk_score_stats']['mean']:.2f}")
    
    print("\n   Risk Tier Distribution:")
    for tier, count in risk_summary['tier_distribution'].items():
        pct = count / risk_summary['total_appointments'] * 100
        print(f"   • {tier}: {count:,} ({pct:.1f}%)")
    
    # Create visualizations
    print("\n3. CREATING VISUALIZATIONS...")
    visualizer = Visualizer(config)
    
    # Create individual charts
    fig1 = visualizer.plot_noshow_rate_by_category(df, 'age_group', 
                                                   title='No-Show Rate by Age Group')
    
    fig2 = visualizer.plot_risk_distribution(df)
    
    fig3 = visualizer.create_summary_dashboard(df)
    
    print(f"   ✅ Created {len(visualizer.get_figures_created())} visualizations")
    
    # Display sample of high-risk appointments
    print("\n4. HIGH-RISK APPOINTMENTS SAMPLE:")
    high_risk = df[df['risk_tier'] == 'CRITICAL'].head()
    if len(high_risk) > 0:
        print(high_risk[['risk_tier_display', 'composite_risk_score', 
                        'intervention_description']].to_string())
    
    print("\n✅ DAY 3 MODULES WORKING CORRECTLY!")
    print("="*60)
    
    return df


if __name__ == "__main__":
    df = main()
    
    # Optional: show plots
    # plt.show()
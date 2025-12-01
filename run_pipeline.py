#!/usr/bin/env python
"""
Main Pipeline Script
====================
Run the complete healthcare appointments no-show analysis pipeline.

Usage:
    python run_pipeline.py [options]
    
Examples:
    python run_pipeline.py                    # Run with default config
    python run_pipeline.py --config custom.yaml
    python run_pipeline.py --source url --verbose
    python run_pipeline.py --skip-viz --output results/
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.abspath('src'))

from src.utils import (
    load_config, 
    setup_logging, 
    print_pipeline_banner,
    print_pipeline_summary,
    calculate_statistics,
    create_directories,
    timer
)
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.feature_engineer import FeatureEngineer
from src.risk_scorer import RiskScorer
from src.visualizations import Visualizer


class Pipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config_path: str = "config/config.yaml", **kwargs):
        """
        Initialize pipeline with configuration.
        
        Parameters
        ----------
        config_path : str
            Path to configuration file
        **kwargs : dict
            Override configuration parameters
        """
        self.config = load_config(config_path)
        
        # Override config with command line arguments
        for key, value in kwargs.items():
            if value is not None:
                if '.' in key:
                    # Handle nested keys like 'paths.output_dir'
                    keys = key.split('.')
                    d = self.config
                    for k in keys[:-1]:
                        d = d.setdefault(k, {})
                    d[keys[-1]] = value
                else:
                    self.config[key] = value
        
        # Setup logging
        self.logger = setup_logging(
            self.config['logging']['level'],
            self.config['logging'].get('file')
        )
        
        # Create directories
        create_directories(self.config)
        
        # Initialize components
        self.loader = DataLoader(self.config)
        self.cleaner = DataCleaner(self.config)
        self.engineer = FeatureEngineer(self.config)
        self.scorer = RiskScorer(self.config)
        self.visualizer = Visualizer(self.config)
        
        # Pipeline state
        self.df = None
        self.results = {}
        
    @timer
    def load_data(self, source: str = "auto") -> None:
        """Load data from specified source."""
        self.logger.info("="*60)
        self.logger.info("STEP 1: DATA LOADING")
        self.logger.info("="*60)
        
        self.df = self.loader.load(source=source)
        
        # Store initial stats
        self.results['initial_stats'] = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'memory_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        self.logger.info(f"Loaded {self.results['initial_stats']['rows']:,} rows")
        
    @timer
    def clean_data(self) -> None:
        """Clean and standardize data."""
        self.logger.info("="*60)
        self.logger.info("STEP 2: DATA CLEANING")
        self.logger.info("="*60)
        
        if self.df is None:
            raise ValueError("No data loaded. Run load_data() first.")
        
        self.df = self.cleaner.clean_pipeline(self.df)
        
        # Store cleaning report
        self.results['cleaning_report'] = self.cleaner.get_cleaning_report()
        
    @timer
    def engineer_features(self) -> None:
        """Create engineered features."""
        self.logger.info("="*60)
        self.logger.info("STEP 3: FEATURE ENGINEERING")
        self.logger.info("="*60)
        
        if self.df is None:
            raise ValueError("No data loaded. Run load_data() first.")
        
        self.df = self.engineer.engineer_all_features(self.df)
        
        # Store features created
        self.results['features_created'] = self.engineer.get_features_created()
        self.results['feature_count'] = len(self.results['features_created'])
        
    @timer
    def calculate_risk_scores(self) -> None:
        """Calculate risk scores and assign tiers."""
        self.logger.info("="*60)
        self.logger.info("STEP 4: RISK SCORING")
        self.logger.info("="*60)
        
        if self.df is None:
            raise ValueError("No data loaded. Run load_data() first.")
        
        self.df = self.scorer.score_pipeline(self.df)
        
        # Store risk summary
        self.results['risk_summary'] = self.scorer.get_risk_summary(self.df)
        
    @timer
    def create_visualizations(self) -> None:
        """Generate all visualizations."""
        self.logger.info("="*60)
        self.logger.info("STEP 5: VISUALIZATIONS")
        self.logger.info("="*60)
        
        if self.df is None:
            raise ValueError("No data loaded. Run load_data() first.")
        
        # Create visualizations
        self.visualizer.plot_noshow_rate_by_category(
            self.df, 'age_group', title='No-Show Rate by Age Group'
        )
        
        self.visualizer.plot_noshow_rate_by_category(
            self.df, 'appointment_weekday', title='No-Show Rate by Day of Week'
        )
        
        self.visualizer.plot_noshow_rate_by_category(
            self.df, 'risk_tier', title='No-Show Rate by Risk Tier'
        )
        
        self.visualizer.plot_risk_distribution(self.df)
        
        self.visualizer.plot_trend_over_time(self.df)
        
        # Factor importance
        importance = {
            'Lead Time': 0.25,
            'Patient History': 0.22,
            'Age Group': 0.15,
            'Neighborhood': 0.12,
            'Day of Week': 0.10,
            'SMS Status': 0.08,
            'Health Conditions': 0.08
        }
        self.visualizer.plot_factor_importance(importance_dict=importance)
        
        # Summary dashboard
        self.visualizer.create_summary_dashboard(self.df)
        
        # Store figures created
        self.results['figures_created'] = self.visualizer.get_figures_created()
        
    @timer
    def generate_reports(self) -> None:
        """Generate analysis reports."""
        self.logger.info("="*60)
        self.logger.info("STEP 6: REPORT GENERATION")
        self.logger.info("="*60)
        
        if self.df is None:
            raise ValueError("No data loaded. Run load_data() first.")
        
        # Calculate final statistics
        final_stats = calculate_statistics(self.df)
        self.results['final_stats'] = final_stats
        
        # Generate executive summary
        summary = self._generate_executive_summary()
        
        # Save summary
        summary_path = Path(self.config['paths']['reports_dir']) / 'executive_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        self.logger.info(f"Executive summary saved to {summary_path}")
        
        # Save results JSON
        results_path = Path(self.config['paths']['reports_dir']) / 'pipeline_results.json'
        with open(results_path, 'w') as f:
            # Convert non-serializable objects
            results_clean = {
                k: v if not hasattr(v, 'to_dict') else v.to_dict() 
                for k, v in self.results.items()
            }
            json.dump(results_clean, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {results_path}")
        
    def _generate_executive_summary(self) -> str:
        """Generate executive summary text."""
        stats = self.results.get('final_stats', {})
        risk = self.results.get('risk_summary', {})
        
        summary = f"""
{'='*70}
HEALTHCARE APPOINTMENTS NO-SHOW ANALYSIS
Executive Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*70}

DATA OVERVIEW
-------------
Total Appointments: {stats.get('total_appointments', 0):,}
Unique Patients: {stats.get('unique_patients', 0):,}
Analysis Period: {stats.get('date_range', {}).get('start')} to {stats.get('date_range', {}).get('end')}
No-Show Rate: {stats.get('no_show_rate', 0)*100:.1f}%
Total No-Shows: {stats.get('total_no_shows', 0):,}

FINANCIAL IMPACT
---------------
Cost per No-Show: ${self.config['business']['cost_per_noshow']}
Total Cost (Period): ${stats.get('total_no_shows', 0) * self.config['business']['cost_per_noshow']:,}
Projected Annual Cost: ${stats.get('total_no_shows', 0) * self.config['business']['cost_per_noshow'] * 4:,}

RISK ANALYSIS
------------
Risk Tiers:
"""
        
        if 'tier_distribution' in risk:
            for tier, count in sorted(risk['tier_distribution'].items()):
                pct = count / stats.get('total_appointments', 1) * 100
                summary += f"  {tier}: {count:,} ({pct:.1f}%)\n"
        
        summary += f"""
High-Risk Appointments: {risk.get('high_risk_count', 0):,}
Phone Calls Needed: {risk.get('interventions_needed', {}).get('phone_calls', 0):,}
Deposits Required: {risk.get('interventions_needed', {}).get('deposits', 0):,}

KEY FINDINGS
-----------
1. Lead time is the strongest predictor of no-shows
   - Same day: 7% no-show rate
   - 30+ days: 23% no-show rate

2. Young adults (18-35) have highest no-show rates (24%)
   - Compared to seniors at 16%

3. Monday has worst performance (22.3% no-show)
   - Saturday performs best (17.3%)

4. SMS paradox identified: selection bias in current strategy
   - SMS recipients have higher no-show rates
   - Need universal SMS coverage

5. Serial no-show pattern: 15% of patients cause 45% of no-shows
   - Targeted interventions for repeat offenders

RECOMMENDATIONS
--------------
IMMEDIATE ACTIONS:
✓ Cap advance booking at 14 days maximum
✓ Expand same-day appointment capacity by 20%
✓ Send SMS reminders to ALL patients

SHORT-TERM (1 month):
✓ Implement phone confirmation for high-risk appointments
✓ Require deposits for chronic no-show patients
✓ Deploy overbooking strategy (10-20% based on risk)

MEDIUM-TERM (3 months):
✓ Launch mobile clinics in high-risk neighborhoods
✓ Extend evening hours for working patients
✓ Implement patient risk scoring in booking system

EXPECTED OUTCOMES
----------------
Target No-Show Rate: {self.config['business']['target_noshow_rate']*100:.0f}%
Potential Reduction: {(stats.get('no_show_rate', 0.202) - self.config['business']['target_noshow_rate'])*100:.1f} percentage points
Annual Savings Opportunity: ${(stats.get('no_show_rate', 0.202) - self.config['business']['target_noshow_rate']) * stats.get('total_appointments', 0) * 4 * self.config['business']['cost_per_noshow']:,.0f}

{'='*70}
END OF EXECUTIVE SUMMARY
{'='*70}
"""
        return summary
    
    @timer
    def run(self, 
           source: str = "auto",
           skip_viz: bool = False,
           skip_reports: bool = False) -> None:
        """
        Run the complete pipeline.
        
        Parameters
        ----------
        source : str
            Data source ('auto', 'csv', 'url', 'database')
        skip_viz : bool
            Skip visualization generation
        skip_reports : bool
            Skip report generation
        """
        print_pipeline_banner()
        
        self.logger.info("Starting Healthcare Appointments Analysis Pipeline")
        self.logger.info(f"Configuration: {self.config['project']['name']} v{self.config['project']['version']}")
        
        start_time = datetime.now()
        
        try:
            # Run pipeline steps
            self.load_data(source=source)
            self.clean_data()
            self.engineer_features()
            self.calculate_risk_scores()
            
            if not skip_viz:
                self.create_visualizations()
            
            if not skip_reports:
                self.generate_reports()
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Print summary
            if self.results.get('final_stats'):
                print_pipeline_summary(self.results['final_stats'])
            
            self.logger.info("="*60)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info(f"Total execution time: {execution_time:.2f} seconds")
            self.logger.info("="*60)
            
            # Save processed data
            if self.config['pipeline'].get('save_intermediate', True):
                output_path = Path(self.config['paths']['features_data'].replace('.csv', '_final.csv'))
                self.df.to_csv(output_path, index=False)
                self.logger.info(f"Final dataset saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def get_dataframe(self):
        """Get the processed dataframe."""
        return self.df
    
    def get_results(self):
        """Get the pipeline results."""
        return self.results


def main():
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Healthcare Appointments No-Show Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                           # Run with defaults
  python run_pipeline.py --config custom.yaml      # Use custom config
  python run_pipeline.py --source url --verbose    # Load from URL with verbose output
  python run_pipeline.py --skip-viz                # Skip visualizations for faster run
  python run_pipeline.py --output results/         # Specify output directory
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--source', '-s',
        choices=['auto', 'csv', 'url', 'database'],
        default='auto',
        help='Data source type (default: auto)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output directory (overrides config)'
    )
    
    parser.add_argument(
        '--skip-viz',
        action='store_true',
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '--skip-reports',
        action='store_true',
        help='Skip report generation'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output (DEBUG level logging)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet output (WARNING level logging)'
    )
    
    args = parser.parse_args()
    
    # Prepare kwargs for pipeline
    kwargs = {}
    if args.output:
        kwargs['paths.outputs_dir'] = args.output
    if args.verbose:
        kwargs['logging.level'] = 'DEBUG'
    elif args.quiet:
        kwargs['logging.level'] = 'WARNING'
    
    # Initialize and run pipeline
    pipeline = Pipeline(args.config, **kwargs)
    pipeline.run(
        source=args.source,
        skip_viz=args.skip_viz,
        skip_reports=args.skip_reports
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
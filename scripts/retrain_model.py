#!/usr/bin/env python
"""
Automated Model Retraining Script
==================================
This script orchestrates the model retraining process by:
1. Loading latest data from the database
2. Exporting to CSV format for training
3. Running the training pipeline
4. Logging results to MLflow

Usage:
    python scripts/retrain_model.py
    python scripts/retrain_model.py --since-days 90
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.database import get_db_session
from src.api.models import AppointmentPrediction
import train_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RetrainModel")


def load_data_from_db(since_days: int = 90) -> pd.DataFrame:
    """
    Load prediction data from database for retraining.
    
    Parameters
    ----------
    since_days : int
        Number of days to look back for data
        
    Returns
    -------
    pd.DataFrame
        Training data
    """
    logger.info(f"Loading data from last {since_days} days...")
    
    cutoff_date = datetime.utcnow() - timedelta(days=since_days)
    
    session = next(get_db_session())
    try:
        # Query predictions from database
        predictions = session.query(AppointmentPrediction)\
            .filter(AppointmentPrediction.created_at >= cutoff_date)\
            .all()
        
        logger.info(f"Loaded {len(predictions)} prediction records")
        
        if len(predictions) < 100:
            logger.warning(f"Only {len(predictions)} records found. This may be insufficient for retraining.")
        
        # Convert to DataFrame
        data = []
        for pred in predictions:
            # Reconstruct input features from stored data
            # This assumes we store the input features in the database
            # Adjust based on your actual database schema
            data.append({
                'patient_id': pred.patient_id,
                'appointment_id': pred.appointment_id,
                'age': pred.age,
                'gender': pred.gender,
                'scholarship': pred.scholarship,
                'hypertension': pred.hypertension,
                'diabetes': pred.diabetes,
                'alcoholism': pred.alcoholism,
                'handcap': pred.handcap,
                'sms_received': pred.sms_received,
                'scheduled_day': pred.scheduled_day,
                'appointment_day': pred.appointment_day,
                'neighbourhood': pred.neighbourhood,
                'no_show': pred.actual_outcome,  # Actual outcome for training
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Created DataFrame with shape: {df.shape}")
        
        return df
        
    finally:
        session.close()


def export_data_for_training(df: pd.DataFrame, output_path: str) -> None:
    """
    Export data to CSV format expected by training pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to export
    output_path : str
        Path to save CSV
    """
    logger.info(f"Exporting data to {output_path}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    logger.info(f"Exported {len(df)} rows to {output_path}")


def run_retraining(data_path: str, config_path: str = "config/ml_config.yaml") -> dict:
    """
    Run the training pipeline.
    
    Parameters
    ----------
    data_path : str
        Path to training data CSV
    config_path : str
        Path to ML configuration
        
    Returns
    -------
    dict
        Training results including best model and metrics
    """
    logger.info("Starting training pipeline...")
    
    # Create training pipeline
    pipeline = train_model.TrainingPipeline(
        config_path=config_path,
        data_path=data_path,
        output_dir=None,  # Use default timestamped directory
        verbose=False
    )
    
    # Run training
    results = pipeline.run(
        tune=False,  # Use baseline models for automated retraining
        evaluate=True,
        interpret=False,  # Skip interpretation for automated runs
        model_names=None
    )
    
    logger.info(f"Training complete! Best model: {results['best_model']}")
    logger.info(f"Results saved to: {results['output_dir']}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Automated model retraining"
    )
    parser.add_argument(
        "--since-days",
        type=int,
        default=90,
        help="Number of days to look back for training data"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/retraining/latest_data.csv",
        help="Path to export training data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/ml_config.yaml",
        help="Path to ML configuration"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 60)
        logger.info("AUTOMATED MODEL RETRAINING")
        logger.info("=" * 60)
        logger.info(f"Start time: {datetime.now()}")
        
        # Step 1: Load data from database
        df = load_data_from_db(since_days=args.since_days)
        
        # Step 2: Export data
        export_data_for_training(df, args.data_path)
        
        # Step 3: Run training
        results = run_retraining(args.data_path, args.config)
        
        logger.info("=" * 60)
        logger.info("RETRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Experiment ID: {results['experiment_id']}")
        logger.info(f"Best Model: {results['best_model']}")
        logger.info(f"MLflow Run ID: {results.get('mlflow_run_id', 'N/A')}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

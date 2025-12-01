# test_integration.py
"""Integration tests for the complete pipeline."""

import pytest
import pandas as pd
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

from run_pipeline import Pipeline


def test_full_pipeline_execution():
    """Test the complete pipeline runs without errors."""
    
    # Initialize pipeline with test config
    pipeline = Pipeline()
    
    # Run pipeline
    pipeline.run(source='auto', skip_viz=True, skip_reports=True)
    
    # Check results
    assert pipeline.df is not None
    assert len(pipeline.df) > 0
    assert 'composite_risk_score' in pipeline.df.columns
    assert 'risk_tier' in pipeline.df.columns
    
    # Check results dictionary
    results = pipeline.get_results()
    assert 'initial_stats' in results
    assert 'cleaning_report' in results
    assert 'features_created' in results
    assert 'risk_summary' in results


def test_pipeline_with_small_data():
    """Test pipeline with a small sample dataset."""
    
    # Create small test dataset
    test_data = pd.DataFrame({
        'PatientId': range(100),
        'AppointmentID': [f'A{i}' for i in range(100)],
        'Gender': ['M', 'F'] * 50,
        'Age': list(range(20, 70)) * 2,
        'No-show': ['Yes', 'No'] * 50,
        'ScheduledDay': '2016-04-29',
        'AppointmentDay': '2016-05-01',
        'Neighbourhood': ['Centro'] * 100,
        'Scholarship': [0, 1] * 50,
        'Hypertension': [0, 1] * 50,
        'Diabetes': [0] * 100,
        'Alcoholism': [0] * 100,
        'Handcap': [0] * 100,
        'SMS_received': [0, 1] * 50
    })
    
    # Save test data
    test_path = 'test_data.csv'
    test_data.to_csv(test_path, index=False)
    
    try:
        # Run pipeline
        pipeline = Pipeline()
        pipeline.config['paths']['raw_data'] = test_path
        pipeline.run(source='csv', skip_viz=True, skip_reports=True)
        
        # Verify results
        assert len(pipeline.df) > 0
        assert 'risk_tier' in pipeline.df.columns
        
    finally:
        # Cleanup
        if Path(test_path).exists():
            Path(test_path).unlink()


if __name__ == "__main__":
    test_full_pipeline_execution()
    test_pipeline_with_small_data()
    print("✅ All integration tests passed!")
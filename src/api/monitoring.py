"""
Model Monitoring
================
Drift detection and monitoring utilities.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Try to import Evidently, but make it optional
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metrics import DataDriftTable, DatasetDriftMetric
    EVIDENTLY_AVAILABLE = True
except ImportError:
    logger.warning("Evidently not available - drift detection disabled")
    EVIDENTLY_AVAILABLE = False


class DriftMonitor:
    """
    Data drift monitor using Evidently.
    
    Compares incoming data against a reference dataset
    to detect distribution shifts.
    """
    
    def __init__(self):
        self.reference_data: Optional[pd.DataFrame] = None
        self._cached_metrics: Optional[Dict[str, Any]] = None
        self._last_analysis: Optional[datetime] = None
        
        # Feature columns to monitor
        self.numerical_features = [
            'age', 'lead_days', 'patient_total_appointments',
            'patient_historical_noshow_rate'
        ]
        self.categorical_features = [
            'gender', 'scholarship', 'hypertension', 'diabetes',
            'alcoholism', 'handicap', 'sms_received'
        ]
    
    def set_reference_data(self, data: pd.DataFrame) -> None:
        """Set the reference dataset for drift comparison."""
        self.reference_data = data.copy()
        self._cached_metrics = None
        logger.info(f"Reference data set with {len(data)} records")
    
    def calculate_drift(
        self, 
        current_data: pd.DataFrame,
        cache_result: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate drift between current and reference data.
        
        Parameters
        ----------
        current_data : pd.DataFrame
            Current data to compare
        cache_result : bool
            Whether to cache the result
        
        Returns
        -------
        dict
            Drift metrics
        """
        if self.reference_data is None:
            return {
                "status": "no_reference",
                "message": "No reference data set. Call set_reference_data first.",
                "drift_detected": False
            }
        
        if not EVIDENTLY_AVAILABLE:
            return self._calculate_simple_drift(current_data)
        
        try:
            # Column mapping
            column_mapping = ColumnMapping(
                numerical_features=self.numerical_features,
                categorical_features=self.categorical_features
            )
            
            # Ensure columns exist
            available_num = [c for c in self.numerical_features if c in current_data.columns]
            available_cat = [c for c in self.categorical_features if c in current_data.columns]
            
            column_mapping = ColumnMapping(
                numerical_features=available_num,
                categorical_features=available_cat
            )
            
            # Create drift report
            report = Report(metrics=[
                DatasetDriftMetric(),
                DataDriftTable()
            ])
            
            report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=column_mapping
            )
            
            # Extract metrics
            result_dict = report.as_dict()
            
            metrics = {
                "status": "success",
                "drift_detected": result_dict["metrics"][0]["result"]["dataset_drift"],
                "dataset_drift_score": result_dict["metrics"][0]["result"]["drift_share"],
                "num_drifted_features": result_dict["metrics"][0]["result"]["number_of_drifted_columns"],
                "total_features": result_dict["metrics"][0]["result"]["number_of_columns"],
                "feature_drift": {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Per-feature drift
            if "metrics" in result_dict and len(result_dict["metrics"]) > 1:
                drift_table = result_dict["metrics"][1]["result"]
                if "drift_by_columns" in drift_table:
                    for col, col_data in drift_table["drift_by_columns"].items():
                        metrics["feature_drift"][col] = {
                            "drifted": col_data.get("drift_detected", False),
                            "p_value": col_data.get("drift_score", None)
                        }
            
            if cache_result:
                self._cached_metrics = metrics
                self._last_analysis = datetime.utcnow()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Drift calculation failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "drift_detected": False
            }
    
    def _calculate_simple_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Simple drift detection without Evidently."""
        if self.reference_data is None:
            return {"status": "no_reference", "drift_detected": False}
        
        drifted_features = []
        feature_drift = {}
        
        # Simple KS test for numerical features
        from scipy import stats
        
        for col in self.numerical_features:
            if col in current_data.columns and col in self.reference_data.columns:
                try:
                    ref_vals = self.reference_data[col].dropna()
                    cur_vals = current_data[col].dropna()
                    
                    if len(ref_vals) > 0 and len(cur_vals) > 0:
                        stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
                        drifted = p_value < 0.05
                        
                        feature_drift[col] = {
                            "drifted": drifted,
                            "p_value": float(p_value)
                        }
                        
                        if drifted:
                            drifted_features.append(col)
                except Exception as e:
                    logger.warning(f"Could not calculate drift for {col}: {e}")
        
        metrics = {
            "status": "success",
            "method": "simple_ks_test",
            "drift_detected": len(drifted_features) > 0,
            "num_drifted_features": len(drifted_features),
            "drifted_features": drifted_features,
            "feature_drift": feature_drift,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._cached_metrics = metrics
        return metrics
    
    def get_cached_metrics(self) -> Optional[Dict[str, Any]]:
        """Get cached drift metrics."""
        return self._cached_metrics


# Singleton instance
_drift_monitor: Optional[DriftMonitor] = None


def get_drift_monitor() -> DriftMonitor:
    """Get drift monitor singleton."""
    global _drift_monitor
    if _drift_monitor is None:
        _drift_monitor = DriftMonitor()
    return _drift_monitor
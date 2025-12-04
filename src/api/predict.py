"""
Prediction Logic Module
=======================
Core prediction logic for the no-show prediction API.

This module handles:
- Model loading and caching
- Feature preprocessing
- Prediction generation
- Explanation generation (optional)
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from functools import lru_cache
import uuid
import io

import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.pipeline import Pipeline

from .config import get_settings, RiskTierConfig
from .schemas import (
    AppointmentFeatures,
    PredictionResponse,
    BatchPredictionResponse,
    RiskAssessment,
    InterventionRecommendation,
    PredictionExplanation,
    FeatureContribution,
    create_risk_assessment,
    create_intervention
)


logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Raised when model cannot be loaded."""
    pass


class PredictionError(Exception):
    """Raised when prediction fails."""
    pass


class NoShowPredictor:
    """
    No-Show Prediction Engine.
    
    This class manages the ML model and provides prediction functionality.
    It handles:
    - Model and preprocessor loading
    - Feature preparation
    - Prediction generation
    - Result formatting
    
    Attributes
    ----------
    model : sklearn estimator
        Loaded ML model
    preprocessor : object
        Loaded preprocessor
    feature_names : list
        List of feature names expected by the model
    metadata : dict
        Model metadata (version, metrics, etc.)
    
    Example
    -------
    >>> predictor = NoShowPredictor()
    >>> predictor.load_model()
    >>> result = predictor.predict(appointment_features)
    """
    
    def __init__(self):
        """Initialize the predictor."""
        self.settings = get_settings()
        self.model = None
        self.explainer = None
        self.preprocessor = None
        self.feature_names: List[str] = []
        self.input_features: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self._is_loaded = False
        
        logger.info("NoShowPredictor initialized")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded and self.model is not None
    
    def load_model(self) -> None:
        """
        Load model and preprocessor from disk.
        
        Raises
        ------
        ModelLoadError
            If model or preprocessor cannot be loaded
        """
        logger.info("Loading model and preprocessor...")
        
        model_path = Path(self.settings.model_path)
        preprocessor_path = Path(self.settings.preprocessor_path)
        metadata_path = Path(self.settings.metadata_path)
        
        # Load model
        if not model_path.exists():
            raise ModelLoadError(f"Model file not found: {model_path}")
        
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            if isinstance(self.model, Pipeline):
                logger.info("Model is a Pipeline")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")
        
        # Load preprocessor
        if not preprocessor_path.exists():
            logger.warning(f"Preprocessor not found: {preprocessor_path}")
            self.preprocessor = None
        else:
            try:
                preprocessor_data = joblib.load(preprocessor_path)
                # Handle different preprocessor formats
                if isinstance(preprocessor_data, dict):
                    self.preprocessor = preprocessor_data.get('preprocessor')
                    self.feature_names = preprocessor_data.get('feature_names', [])
                    
                    # Extract input features expected by the preprocessor
                    valid_numeric = preprocessor_data.get('valid_numeric', [])
                    valid_categorical = preprocessor_data.get('valid_categorical', [])
                    valid_binary = preprocessor_data.get('valid_binary', [])
                    self.input_features = valid_numeric + valid_categorical + valid_binary
                    
                    logger.info(f"Loaded {len(self.input_features)} input features from preprocessor")
                else:
                    self.preprocessor = preprocessor_data
                    if hasattr(self.preprocessor, 'feature_names_'):
                        self.feature_names = self.preprocessor.feature_names_
                
                logger.info(f"Preprocessor loaded from {preprocessor_path}")
                logger.info(f"Preprocessor type: {type(self.preprocessor)}")
            except Exception as e:
                logger.warning(f"Failed to load preprocessor: {e}")
                self.preprocessor = None
        
        # Load metadata
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Metadata loaded from {metadata_path}")
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                self.metadata = {}
        
        # Load feature importance
        feature_importance_path = model_path.parent / "feature_importance.json"
        self.feature_importance = []
        if feature_importance_path.exists():
            try:
                with open(feature_importance_path, 'r') as f:
                    self.feature_importance = json.load(f)
                logger.info(f"Feature importance loaded from {feature_importance_path}")
            except Exception as e:
                logger.warning(f"Failed to load feature importance: {e}")
                
        # Load SHAP explainer
        explainer_path = model_path.parent / "shap_explainer.joblib"
        if explainer_path.exists():
            try:
                self.explainer = joblib.load(explainer_path)
                logger.info(f"SHAP explainer loaded from {explainer_path}")
            except Exception as e:
                logger.warning(f"Failed to load SHAP explainer: {e}")
                self.explainer = None
        else:
            logger.warning("SHAP explainer not found.")
            self.explainer = None

        # Set default metadata
        if not self.metadata:
            self.metadata = {
                'model_name': self.settings.model_name,
                'model_version': self.settings.model_version,
                'model_type': type(self.model).__name__ if self.model else 'Unknown'
            }
        
        self._is_loaded = True
        logger.info("Model loading complete!")
    
    def _prepare_features(
        self, 
        appointment: AppointmentFeatures
    ) -> pd.DataFrame:
        """
        Prepare features for prediction.
        
        Parameters
        ----------
        appointment : AppointmentFeatures
            Input appointment data
        
        Returns
        -------
        pd.DataFrame
            Prepared feature dataframe
        """
        # Convert to dictionary
        data = appointment.model_dump()
        logger.info(f"Preparing features for data: {data}")
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # --- 1. Basic Features ---
        df['age'] = data['age']
        df['lead_days'] = data['lead_days']
        df['scholarship'] = data['scholarship']
        df['hypertension'] = data['hypertension']
        df['diabetes'] = data['diabetes']
        df['alcoholism'] = data['alcoholism']
        df['handicap'] = data['handicap']
        df['sms_received'] = data['sms_received']
        df['gender'] = data['gender']
        
        # --- 2. Derived Time Features ---
        # Schedule hour (default to 9 AM if not provided)
        df['schedule_hour'] = 9 
        
        # Weekday features
        df['appointment_weekday'] = data.get('appointment_weekday') or self.settings.default_weekday
        weekday = df['appointment_weekday'].iloc[0]
        df['is_monday'] = int(weekday == 'Monday')
        df['is_friday'] = int(weekday == 'Friday')
        df['is_weekend'] = int(weekday in ['Saturday', 'Sunday'])
        
        # Lead time category
        lead_days = data['lead_days']
        if lead_days <= 0:
            df['lead_time_category'] = 'Same Day'
        elif lead_days <= 7:
            df['lead_time_category'] = '1-7 days'
        elif lead_days <= 14:
            df['lead_time_category'] = '8-14 days'
        elif lead_days <= 30:
            df['lead_time_category'] = '15-30 days'
        else:
            df['lead_time_category'] = '30+ days'
            
        # --- 3. Patient History Features ---
        # Robust None handling
        patient_total = data.get('patient_total_appointments')
        if patient_total is None:
            patient_total = 1
        df['patient_total_appointments'] = patient_total
        
        patient_hist_rate = data.get('patient_historical_noshow_rate')
        if patient_hist_rate is None:
            patient_hist_rate = 0.2
        df['patient_historical_noshow_rate'] = patient_hist_rate
        
        is_first = data.get('is_first_appointment')
        if is_first is None:
            is_first = 1
        df['is_first_appointment'] = is_first
        
        # Calculate previous noshows based on rate and total
        # rate = prev_noshows / (total - 1) if total > 1 else 0
        total_prev = max(0, patient_total - 1)
        df['patient_previous_noshows'] = int(total_prev * patient_hist_rate)
        
        # --- 4. Neighborhood Features (Defaults) ---
        # In a real system, these would be looked up from a database
        df['neighborhood_noshow_rate'] = 0.2  # Global average
        df['neighborhood_avg_age'] = 37.0     # Global average
        df['neighborhood_risk'] = 'Medium'
        
        # --- 5. Age Group ---
        age = data['age']
        if age < 12:
            df['age_group'] = 'Child'
        elif age < 18:
            df['age_group'] = 'Teen'
        elif age < 35:
            df['age_group'] = 'Young Adult'
        elif age < 50:
            df['age_group'] = 'Adult'
        elif age < 65:
            df['age_group'] = 'Middle Age'
        else:
            df['age_group'] = 'Senior'
            
        # --- 6. Health Risk ---
        df['total_conditions'] = (
            data.get('hypertension', 0) + 
            data.get('diabetes', 0) + 
            data.get('alcoholism', 0)
        )
        df['has_chronic_condition'] = (df['total_conditions'] > 0).astype(int)
        df['has_disability'] = (df['handicap'] > 0).astype(int)
        
        if df['total_conditions'].iloc[0] >= 2:
            df['health_risk_category'] = 'High'
        elif df['total_conditions'].iloc[0] == 1:
            df['health_risk_category'] = 'Medium'
        else:
            df['health_risk_category'] = 'Low'
            
        # --- 7. Interaction Features ---
        # young_long_lead: Young Adult + lead_days > 14
        df['young_long_lead'] = int(df['age_group'].iloc[0] == 'Young Adult' and lead_days > 14)
        
        # monday_long_lead: Monday + lead_days > 14
        df['monday_long_lead'] = int(df['is_monday'].iloc[0] == 1 and lead_days > 14)
        
        # first_young: First appointment + (Child or Teen or Young Adult)
        is_young = df['age_group'].iloc[0] in ['Child', 'Teen', 'Young Adult']
        df['first_young'] = int(df['is_first_appointment'].iloc[0] == 1 and is_young)
        
        # high_risk_no_sms: High historical rate (>0.5) + No SMS
        # Ensure patient_hist_rate is float
        high_hist_risk = float(patient_hist_rate) > 0.5
        df['high_risk_no_sms'] = int(high_hist_risk and df['sms_received'].iloc[0] == 0)
        
        # elderly_chronic: Senior + Chronic Condition
        df['elderly_chronic'] = int(df['age_group'].iloc[0] == 'Senior' and df['has_chronic_condition'].iloc[0] == 1)
        
        # Ensure column order matches model expectation (optional but good for debugging)
        # The ColumnTransformer will handle order by name usually, but good to have all present.
        
        return df
    
    def _transform_features(self, df: pd.DataFrame) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform features using preprocessor.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw feature dataframe
        
        Returns
        -------
        Union[np.ndarray, pd.DataFrame]
            Transformed features or raw dataframe if model is Pipeline
        """
        # Determine expected features
        expected_features = []
        
        # Priority 1: Use feature names from preprocessor if available (most reliable)
        if hasattr(self.preprocessor, 'feature_names_in_'):
            expected_features = list(self.preprocessor.feature_names_in_)
            logger.info(f"Using {len(expected_features)} features from preprocessor.feature_names_in_")
            
        # Priority 2: Use input_features from metadata
        elif self.input_features:
            expected_features = self.input_features
            
        # Priority 3: Legacy fallback for specific 30-feature model
        elif hasattr(self.preprocessor, 'n_features_in_') and self.preprocessor.n_features_in_ == 30:
             # Hardcoded list of 30 features from inspection
            expected_features = [
                'age', 'lead_days', 'patient_total_appointments', 'patient_historical_noshow_rate', 
                'days_since_last_appointment', 'neighborhood_noshow_rate', 'neighborhood_scholarship_rate', 
                'neighborhood_avg_age', 'gender', 'appointment_weekday', 'lead_time_category', 
                'age_group', 'neighborhood_risk', 'health_risk_category', 'scholarship', 'hypertension', 
                'diabetes', 'alcoholism', 'handicap', 'sms_received', 'is_monday', 'is_friday', 
                'is_weekend', 'is_month_start', 'is_month_end', 'scheduled_in_business_hours', 
                'is_first_appointment', 'has_chronic_condition', 'has_disability', 'severe_disability'
            ]
        
        if expected_features:
            logger.info(f"Filtering features. Expected: {len(expected_features)}, Available: {len(df.columns)}")
            # Ensure all expected features are present
            missing_cols = [col for col in expected_features if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing expected features: {missing_cols}")
                # Add missing columns with default values (0) to prevent crash
                for col in missing_cols:
                    df[col] = 0
            
            # Select only expected features in correct order
            X_input = df[expected_features]
        else:
            logger.warning("No expected features defined! Passing all features.")
            X_input = df

        # Check if model is a Pipeline
        if isinstance(self.model, Pipeline):
            logger.info("Model is a Pipeline, skipping manual transformation.")
            return X_input
        
        # Otherwise, transform manually
        if self.preprocessor is not None:
            try:
                return self.preprocessor.transform(X_input)
            except Exception as e:
                logger.error(f"Transform failed: {e}")
                raise e
        
        # Fallback to manual numeric selection if no preprocessor and not pipeline
        logger.warning("No preprocessor and not a pipeline. Using manual numeric selection.")
        numeric_cols = [
            'age', 'lead_days', 'scholarship', 'hypertension', 
            'diabetes', 'alcoholism', 'handicap', 'sms_received',
            'total_conditions', 'has_chronic_condition', 'has_disability',
            'is_first_appointment', 'patient_historical_noshow_rate',
            'patient_total_appointments', 'is_monday', 'is_friday', 'is_weekend'
        ]
        available_cols = [c for c in numeric_cols if c in df.columns]
        return df[available_cols].values
    
    def predict(
        self,
        appointment: AppointmentFeatures,
        threshold: Optional[float] = None,
        include_explanation: bool = False
    ) -> PredictionResponse:
        """
        Make a prediction for a single appointment.
        
        Parameters
        ----------
        appointment : AppointmentFeatures
            Appointment features
        threshold : float, optional
            Classification threshold (default from settings)
        include_explanation : bool
            Whether to include feature explanations
        
        Returns
        -------
        PredictionResponse
            Complete prediction response
        
        Raises
        ------
        PredictionError
            If prediction fails
        """
        try:
            # Prepare features
            df = self._prepare_features(appointment)
            X = self._transform_features(df)
            
            # Get prediction
            if hasattr(self.model, 'predict_proba'):
                probability = self.model.predict_proba(X)[0, 1]
            else:
                probability = float(self.model.predict(X)[0])
            
            prediction = int(probability >= (threshold or self.settings.default_threshold))
            
            # Create risk assessment
            risk = create_risk_assessment(probability)
            
            # Create intervention recommendation
            intervention = create_intervention(probability, risk.tier)
            
            # Generate explanation if requested
            explanation = None
            if include_explanation:
                explanation = self._generate_explanation(X, df, probability)
            
            # Create response
            response = PredictionResponse(
                prediction=prediction,
                probability=round(probability, 4),
                risk=risk,
                intervention=intervention,
                explanation=explanation,
                model_version=self.metadata.get('model_version', '1.0.0'),
                prediction_id=str(uuid.uuid4())[:8],
                timestamp=datetime.utcnow()
            )
            
            # Log prediction
            if self.settings.log_predictions:
                logger.info(
                    f"Prediction: prob={probability:.3f}, "
                    f"tier={risk.tier}, pred={prediction}"
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise PredictionError(f"Prediction failed: {e}")
    
    def predict_batch(
        self,
        appointments: List[AppointmentFeatures],
        threshold: Optional[float] = None,
        include_explanations: bool = False
    ) -> BatchPredictionResponse:
        """
        Make predictions for multiple appointments.
        
        Parameters
        ----------
        appointments : list
            List of appointment features
        threshold : float, optional
            Classification threshold
        include_explanations : bool
            Whether to include explanations
        
        Returns
        -------
        BatchPredictionResponse
            Batch prediction response
        """
        import time
        start_time = time.time()
        
        predictions = []
        risk_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "MINIMAL": 0}
        
        for appointment in appointments:
            pred = self.predict(
                appointment,
                threshold=threshold,
                include_explanation=include_explanations
            )
            predictions.append(pred)
            risk_counts[pred.risk.tier] += 1
        
        processing_time = (time.time() - start_time) * 1000
        
        # Calculate summary
        probabilities = [p.probability for p in predictions]
        summary = {
            "total": len(predictions),
            "predicted_noshows": sum(1 for p in predictions if p.prediction == 1),
            "predicted_shows": sum(1 for p in predictions if p.prediction == 0),
            "avg_probability": round(np.mean(probabilities), 4),
            "risk_distribution": risk_counts,
            "threshold_used": threshold or self.settings.default_threshold
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary,
            processing_time_ms=round(processing_time, 2)
        )
    
    def _generate_explanation(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        df: pd.DataFrame,
        probability: float
    ) -> PredictionExplanation:
        """
        Generate explanation for a prediction.
        
        Uses feature importance from the model to explain predictions.
        For tree-based models, uses feature_importances_.
        For linear models, uses coefficients.
        """
        try:
            # Use SHAP if available
            if self.explainer is not None:
                try:
                    # Calculate SHAP values
                    # Note: TreeExplainer expects transformed features
                    shap_values = self.explainer.shap_values(X)
                    
                    # Handle different SHAP output formats (list for classification vs array)
                    if isinstance(shap_values, list):
                        # Binary classification, take the positive class (index 1)
                        shap_values = shap_values[1]
                    
                    # If single prediction, get first row
                    if len(shap_values.shape) > 1:
                        shap_values = shap_values[0]
                        
                    # Get feature names
                    if self.feature_names:
                        feature_names = self.feature_names
                    else:
                        feature_names = [f'feature_{i}' for i in range(len(shap_values))]
                        
                    contributions = []
                    limit = min(len(feature_names), len(shap_values))
                    
                    for i in range(limit):
                        name = feature_names[i]
                        shap_val = float(shap_values[i])
                        
                        # Skip zero contributions
                        if abs(shap_val) < 0.001:
                            continue
                            
                        # Direction: positive SHAP pushes towards class 1 (No-Show)
                        direction = 'positive' if shap_val > 0 else 'negative'
                        
                        contributions.append(
                            FeatureContribution(
                                feature=name,
                                value=X[0, i] if i < X.shape[1] else 0, # Approximate value
                                contribution=round(shap_val, 4),
                                direction=direction
                            )
                        )
                        
                    # Sort by absolute contribution
                    contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
                    
                    # Split into risk and protective factors
                    top_risk = [c for c in contributions if c.direction == 'positive'][:5]
                    top_protective = [c for c in contributions if c.direction == 'negative'][:5]
                    
                    return PredictionExplanation(
                        top_risk_factors=top_risk,
                        top_protective_factors=top_protective,
                        summary=self._generate_summary_text(probability, df)
                    )
                    
                except Exception as e:
                    logger.warning(f"SHAP calculation failed: {e}. Falling back to simple explanation.")

            # Fallback to simple explanation (existing logic)
            # Handle Pipeline model
            model = self.model
            if isinstance(model, Pipeline):
                # Try to get the classifier step
                if hasattr(model, 'steps'):
                    model = model.steps[-1][1]
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                # Return simple explanation without detailed feature contributions
                return PredictionExplanation(
                    top_risk_factors=[],
                    top_protective_factors=[],
                    summary=self._generate_summary_text(probability, df)
                )
            
            # Get feature names
            if self.feature_names:
                feature_names = self.feature_names
            else:
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            # Handle X if it's DataFrame (Pipeline case)
            if isinstance(X, pd.DataFrame):
                # We need transformed values for contribution calculation?
                # Or just use raw values?
                # For simplicity, use raw values if available, or 0
                # But feature importance maps to TRANSFORMED features (e.g. one-hot)
                # This is tricky with Pipeline.
                # For now, we'll just use the available X values if they match length
                pass
            
            # Create feature contributions (simplified)
            # In production, use SHAP for accurate contributions
            contributions = []
            
            # Limit to min length to avoid index error
            limit = min(len(feature_names), len(importances))
            
            # If X is DataFrame, we can't easily map to transformed features without transforming
            # So we might skip detailed contribution if X is DataFrame
            if isinstance(X, pd.DataFrame):
                 # Fallback to summary only for now to avoid errors
                 return PredictionExplanation(
                    top_risk_factors=[],
                    top_protective_factors=[],
                    summary=self._generate_summary_text(probability, df)
                )

            for i in range(limit):
                name = feature_names[i]
                imp = importances[i]
                
                # Determine direction based on feature value and probability
                direction = 'positive' if probability > 0.5 else 'negative'
                contributions.append(
                    FeatureContribution(
                        feature=name,
                        value=X[0, i] if i < X.shape[1] else 0,
                        contribution=round(float(imp), 4),
                        direction=direction
                    )
                )
            
            # Sort by absolute contribution
            contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
            
            # Split into risk and protective factors
            top_risk = [c for c in contributions if c.direction == 'positive'][:5]
            top_protective = [c for c in contributions if c.direction == 'negative'][:5]
            
            return PredictionExplanation(
                top_risk_factors=top_risk,
                top_protective_factors=top_protective,
                summary=self._generate_summary_text(probability, df)
            )
            
        except Exception as e:
            logger.warning(f"Failed to generate explanation: {e}")
            return PredictionExplanation(
                top_risk_factors=[],
                top_protective_factors=[],
                summary=self._generate_summary_text(probability, df)
            )
    
    def _generate_summary_text(
        self, 
        probability: float, 
        df: pd.DataFrame
    ) -> str:
        """Generate human-readable summary."""
        tier = RiskTierConfig.get_tier(probability)
        
        data = df.iloc[0]
        factors = []
        
        # Check key risk factors
        if data.get('lead_days', 0) > 14:
            factors.append("long lead time")
        if data.get('age_group') == 'Young Adult':
            factors.append("young adult age group")
        if data.get('is_first_appointment', 0) == 1:
            factors.append("first-time patient")
        if data.get('sms_received', 1) == 0:
            factors.append("no SMS reminder sent")
        
        # Protective factors
        if data.get('has_chronic_condition', 0) == 1:
            factors.append("chronic condition (typically lower no-show)")
        if data.get('lead_days', 0) <= 3:
            factors.append("short lead time (lower risk)")
        
        factor_text = ", ".join(factors) if factors else "standard risk profile"
        
        return (
            f"This appointment has a {probability:.1%} probability of no-show, "
            f"classified as {tier} risk. Key factors: {factor_text}."
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        return {
            "metadata": self.metadata,
            "feature_importance": getattr(self, 'feature_importance', []),
            "input_features": self.input_features,
            "model_type": type(self.model).__name__
        }


def get_predictor() -> NoShowPredictor:
    """
    Get the singleton predictor instance.
    
    Returns
    -------
    NoShowPredictor
        Predictor instance
    """
    global _predictor
    
    if _predictor is None:
        _predictor = NoShowPredictor()
        try:
            _predictor.load_model()
        except ModelLoadError as e:
            logger.error(f"Failed to load model: {e}")
            # Return unloaded predictor - will fail gracefully on predict
    
    return _predictor


def reload_predictor() -> NoShowPredictor:
    """
    Reload the predictor with fresh model.
    
    Returns
    -------
    NoShowPredictor
        New predictor instance
    """
    global _predictor
    _predictor = NoShowPredictor()
    _predictor.load_model()
    return _predictor

_predictor = None
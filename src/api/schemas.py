"""
API Schemas Module
==================
Pydantic models for request/response validation.

These schemas ensure:
- Input validation with clear error messages
- Response structure consistency
- Automatic OpenAPI documentation
- Type safety throughout the API
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


# ==================== ENUMS ====================

class Gender(str, Enum):
    """Gender options."""
    MALE = "M"
    FEMALE = "F"
    OTHER = "O"


class RiskTier(str, Enum):
    """Risk tier levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    MINIMAL = "MINIMAL"


class Weekday(str, Enum):
    """Days of the week."""
    MONDAY = "Monday"
    TUESDAY = "Tuesday"
    WEDNESDAY = "Wednesday"
    THURSDAY = "Thursday"
    FRIDAY = "Friday"
    SATURDAY = "Saturday"
    SUNDAY = "Sunday"


# ==================== INPUT SCHEMAS ====================

class AppointmentBase(BaseModel):
    """Base appointment information."""
    
    age: int = Field(
        ...,
        ge=0,
        le=120,
        description="Patient age in years",
        examples=[35]
    )
    gender: Gender = Field(
        ...,
        description="Patient gender (M/F/O)",
        examples=["F"]
    )
    
    class Config:
        use_enum_values = True


class AppointmentFeatures(AppointmentBase):
    """
    Complete appointment features for prediction.
    
    This schema defines all features required for making a no-show prediction.
    Required fields must be provided; optional fields have sensible defaults.
    """
    
    # Medical conditions (binary)
    scholarship: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Whether patient is enrolled in welfare program (0/1)",
        examples=[0]
    )
    hypertension: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Whether patient has hypertension (0/1)",
        examples=[0]
    )
    diabetes: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Whether patient has diabetes (0/1)",
        examples=[0]
    )
    alcoholism: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Whether patient has alcoholism (0/1)",
        examples=[0]
    )
    handicap: int = Field(
        default=0,
        ge=0,
        le=4,
        description="Handicap level (0-4)",
        examples=[0]
    )
    
    # Appointment details
    sms_received: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Whether patient received SMS reminder (0/1)",
        examples=[1]
    )
    lead_days: int = Field(
        ...,
        ge=0,
        le=365,
        description="Days between scheduling and appointment",
        examples=[7]
    )
    
    # Optional features
    neighbourhood: Optional[str] = Field(
        default=None,
        description="Patient neighbourhood",
        examples=["JARDIM CAMBURI"]
    )
    appointment_weekday: Optional[Weekday] = Field(
        default=None,
        description="Day of the week for appointment",
        examples=["Monday"]
    )
    appointment_month: Optional[str] = Field(
        default=None,
        description="Month of appointment",
        examples=["May"]
    )
    
    # Patient history (optional - for returning patients)
    patient_historical_noshow_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Patient's historical no-show rate",
        examples=[0.2]
    )
    patient_total_appointments: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total previous appointments",
        examples=[5]
    )
    is_first_appointment: Optional[int] = Field(
        default=None,
        ge=0,
        le=1,
        description="Whether this is patient's first appointment (0/1)",
        examples=[0]
    )
    
    @field_validator('gender', mode='before')
    @classmethod
    def validate_gender(cls, v):
        """Accept various gender formats."""
        if isinstance(v, str):
            v = v.upper()
            if v in ['M', 'MALE', 'MAN']:
                return 'M'
            elif v in ['F', 'FEMALE', 'WOMAN']:
                return 'F'
            elif v in ['O', 'OTHER']:
                return 'O'
        return v
    
    @field_validator('appointment_weekday', mode='before')
    @classmethod
    def validate_weekday(cls, v):
        """Accept various weekday formats."""
        if isinstance(v, str):
            return v.capitalize()
        return v
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "age": 35,
                "gender": "F",
                "scholarship": 0,
                "hypertension": 0,
                "diabetes": 0,
                "alcoholism": 0,
                "handicap": 0,
                "sms_received": 1,
                "lead_days": 7,
                "neighbourhood": "JARDIM CAMBURI",
                "appointment_weekday": "Monday"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    
    appointments: List[AppointmentFeatures] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of appointments to predict"
    )
    include_explanations: bool = Field(
        default=False,
        description="Whether to include feature importance explanations"
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Custom classification threshold"
    )


# ==================== OUTPUT SCHEMAS ====================

class RiskAssessment(BaseModel):
    """Risk assessment details."""
    
    tier: RiskTier = Field(description="Risk tier classification")
    probability: float = Field(
        ge=0.0,
        le=1.0,
        description="No-show probability"
    )
    confidence: str = Field(description="Confidence level description")
    color: str = Field(description="Hex color code for UI")
    emoji: str = Field(description="Emoji indicator")
    
    class Config:
        use_enum_values = True


class InterventionRecommendation(BaseModel):
    """Recommended intervention for the appointment."""
    
    action: str = Field(description="Primary recommended action")
    sms_reminders: int = Field(description="Number of SMS reminders")
    phone_call: bool = Field(description="Whether phone call is recommended")
    priority: str = Field(description="Scheduling priority")
    notes: Optional[str] = Field(default=None, description="Additional notes")


class FeatureContribution(BaseModel):
    """Feature contribution to prediction."""
    
    feature: str = Field(description="Feature name")
    value: Any = Field(description="Feature value")
    contribution: float = Field(description="Contribution to prediction")
    direction: str = Field(description="'positive' increases no-show risk, 'negative' decreases")


class PredictionExplanation(BaseModel):
    """Explanation of prediction."""
    
    top_risk_factors: List[FeatureContribution] = Field(
        description="Features increasing no-show risk"
    )
    top_protective_factors: List[FeatureContribution] = Field(
        description="Features decreasing no-show risk"
    )
    summary: str = Field(description="Human-readable explanation")


class PredictionResponse(BaseModel):
    """
    Single prediction response.
    
    Contains the prediction result, risk assessment, and optional explanation.
    """
    
    # Core prediction
    prediction: int = Field(
        description="Binary prediction (0=show, 1=no-show)"
    )
    probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability of no-show"
    )
    
    # Risk assessment
    risk: RiskAssessment = Field(description="Risk assessment details")
    
    # Intervention
    intervention: InterventionRecommendation = Field(
        description="Recommended intervention"
    )
    
    # Optional explanation
    explanation: Optional[PredictionExplanation] = Field(
        default=None,
        description="Prediction explanation (if requested)"
    )
    
    # Metadata
    model_version: str = Field(description="Model version used")
    prediction_id: Optional[str] = Field(
        default=None,
        description="Unique prediction ID for tracking"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Prediction timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 0,
                "probability": 0.25,
                "risk": {
                    "tier": "LOW",
                    "probability": 0.25,
                    "confidence": "Moderate",
                    "color": "#2ecc71",
                    "emoji": "🟢"
                },
                "intervention": {
                    "action": "Standard SMS reminder",
                    "sms_reminders": 1,
                    "phone_call": False,
                    "priority": "normal",
                    "notes": None
                },
                "model_version": "1.0.0",
                "timestamp": "2024-01-20T15:30:00Z"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    
    predictions: List[PredictionResponse] = Field(
        description="List of predictions"
    )
    summary: Dict[str, Any] = Field(
        description="Summary statistics"
    )
    processing_time_ms: float = Field(
        description="Processing time in milliseconds"
    )


# ==================== HEALTH & INFO SCHEMAS ====================

class HealthStatus(str, Enum):
    """API health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ModelInfo(BaseModel):
    """Model information."""
    
    name: str = Field(description="Model name")
    version: str = Field(description="Model version")
    type: str = Field(description="Model type (e.g., RandomForestClassifier)")
    features: int = Field(description="Number of input features")
    trained_at: Optional[str] = Field(default=None, description="Training timestamp")
    metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Model performance metrics"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: HealthStatus = Field(description="Overall API status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Check timestamp"
    )
    version: str = Field(description="API version")
    model_loaded: bool = Field(description="Whether model is loaded")
    model_info: Optional[ModelInfo] = Field(
        default=None,
        description="Model information"
    )
    system: Optional[Dict[str, Any]] = Field(
        default=None,
        description="System information"
    )
    
    class Config:
        use_enum_values = True


class APIInfo(BaseModel):
    """API information response."""
    
    name: str
    version: str
    description: str
    documentation_url: str
    endpoints: List[Dict[str, str]]


# ==================== ERROR SCHEMAS ====================

class ErrorDetail(BaseModel):
    """Error detail."""
    
    field: Optional[str] = Field(default=None, description="Field with error")
    message: str = Field(description="Error message")
    type: str = Field(description="Error type")


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    details: Optional[List[ErrorDetail]] = Field(
        default=None,
        description="Detailed error information"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    path: Optional[str] = Field(default=None, description="Request path")


# ==================== UTILITY FUNCTIONS ====================

def create_risk_assessment(probability: float) -> RiskAssessment:
    """
    Create a RiskAssessment from a probability.
    
    Parameters
    ----------
    probability : float
        No-show probability (0-1)
    
    Returns
    -------
    RiskAssessment
        Complete risk assessment
    """
    from .config import RiskTierConfig
    
    tier_info = RiskTierConfig.get_tier_info(probability)
    
    # Determine confidence
    if probability < 0.2 or probability > 0.8:
        confidence = "High"
    elif probability < 0.35 or probability > 0.65:
        confidence = "Moderate"
    else:
        confidence = "Low"
    
    return RiskAssessment(
        tier=tier_info["tier"],
        probability=probability,
        confidence=confidence,
        color=tier_info["color"],
        emoji=tier_info["emoji"]
    )


def create_intervention(probability: float, tier: str) -> InterventionRecommendation:
    """
    Create intervention recommendation based on risk.
    
    Parameters
    ----------
    probability : float
        No-show probability
    tier : str
        Risk tier
    
    Returns
    -------
    InterventionRecommendation
        Recommended intervention
    """
    interventions = {
        "CRITICAL": InterventionRecommendation(
            action="Immediate phone call + deposit required",
            sms_reminders=3,
            phone_call=True,
            priority="high",
            notes="Consider offering telehealth alternative"
        ),
        "HIGH": InterventionRecommendation(
            action="Phone call + double SMS reminder",
            sms_reminders=2,
            phone_call=True,
            priority="high",
            notes="Confirm appointment 24h before"
        ),
        "MEDIUM": InterventionRecommendation(
            action="Double SMS reminder",
            sms_reminders=2,
            phone_call=False,
            priority="normal",
            notes="Send reminder 48h and 24h before"
        ),
        "LOW": InterventionRecommendation(
            action="Standard SMS reminder",
            sms_reminders=1,
            phone_call=False,
            priority="normal",
            notes=None
        ),
        "MINIMAL": InterventionRecommendation(
            action="No additional intervention needed",
            sms_reminders=1,
            phone_call=False,
            priority="low",
            notes="Reliable patient - consider priority scheduling"
        )
    }
    
    return interventions.get(tier, interventions["MEDIUM"])
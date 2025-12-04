"""
Database Models
===============
SQLAlchemy models for the application.
"""

from sqlalchemy import Boolean, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.sql import func
from .database import Base
from .security.encryption import EncryptedString

class User(Base):
    """User model for authentication."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(EncryptedString, unique=True, index=True, nullable=True)
    full_name = Column(EncryptedString, nullable=True)
    hashed_password = Column(String)
    disabled = Column(Boolean, default=False)
    role = Column(String, default="user")


class LLMUsage(Base):
    """Track LLM API usage and costs."""
    __tablename__ = "llm_usage"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    model = Column(String)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    total_tokens = Column(Integer)
    cost = Column(Float)  # USD
    endpoint = Column(String)
    request_id = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)


class UserQuota(Base):
    """User quotas for LLM usage."""
    __tablename__ = "user_quotas"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True)
    monthly_budget_usd = Column(Float, default=10.0)
    monthly_request_limit = Column(Integer, default=1000)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class AppointmentPrediction(Base):
    """Store prediction history."""
    __tablename__ = "appointment_predictions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(EncryptedString, index=True)
    appointment_id = Column(String, index=True)
    prediction = Column(String)  # 'show' or 'no-show'
    probability = Column(Float)
    created_at = Column(DateTime, server_default=func.now(), index=True)
    # Store input features for drift detection
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    scholarship = Column(Boolean, nullable=True)
    hypertension = Column(Boolean, nullable=True)
    diabetes = Column(Boolean, nullable=True)
    alcoholism = Column(Boolean, nullable=True)
    handcap = Column(Integer, nullable=True)
    sms_received = Column(Boolean, nullable=True)
    scheduled_day = Column(DateTime, nullable=True)
    appointment_day = Column(DateTime, nullable=True)
    neighbourhood = Column(EncryptedString, nullable=True)
    actual_outcome = Column(String, nullable=True)  # For evaluation


class LLMFeedback(Base):
    """Store user feedback on LLM responses."""
    __tablename__ = "llm_feedback"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, index=True)
    endpoint = Column(String, index=True)
    rating = Column(Integer)  # 1-5 scale
    comment = Column(String, nullable=True)
    user_id = Column(String, nullable=True, index=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)


class AuditLog(Base):
    """Audit log for compliance and security."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=True)
    action = Column(String, index=True)  # create, read, update, delete
    resource_type = Column(String, index=True)
    resource_id = Column(String, nullable=True)
    details = Column(Text, nullable=True)  # JSON string
    ip_address = Column(String, nullable=True)
    timestamp = Column(DateTime, server_default=func.now(), index=True)


class PatientConsent(Base):
    """Track patient consent for data processing."""
    __tablename__ = "patient_consents"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(EncryptedString, index=True)
    consent_type = Column(String, index=True)  # data_processing, sms, etc.
    granted = Column(Boolean, default=False)
    timestamp = Column(DateTime, server_default=func.now())
    expires_at = Column(DateTime, nullable=True)


class Settings(Base):
    """Application settings."""
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=True) # Optional: if we want per-user settings later
    cost_per_no_show = Column(Float, default=50.0)
    risk_threshold_high = Column(Float, default=0.8)
    risk_threshold_medium = Column(Float, default=0.5)
    notifications_enabled = Column(Boolean, default=True)
    theme = Column(String, default="dark")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


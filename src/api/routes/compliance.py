# src/api/routes/compliance.py
"""
Compliance API Routes
=====================
Endpoints for managing patient consent and viewing audit logs.
"""

import logging
from typing import Optional, List
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ..database import get_db_session
from ..models import PatientConsent, AuditLog, User
from ..auth import get_current_active_user
from ..security.audit import AuditLogger

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/compliance", tags=["Compliance"])


# Schemas
class ConsentCreate(BaseModel):
    """Request to create consent record."""
    patient_id: str
    consent_type: str = Field(..., description="Type of consent (e.g., 'data_processing', 'sms')")
    granted: bool = True
    expires_in_days: Optional[int] = 365


class ConsentResponse(BaseModel):
    """Consent record response."""
    id: int
    patient_id: str
    consent_type: str
    granted: bool
    timestamp: datetime
    expires_at: Optional[datetime]


class AuditLogResponse(BaseModel):
    """Audit log entry response."""
    id: int
    user_id: Optional[str]
    action: str
    resource_type: str
    resource_id: Optional[str]
    details: Optional[str]
    ip_address: Optional[str]
    timestamp: datetime


@router.post("/consent", response_model=ConsentResponse)
async def record_consent(
    consent: ConsentCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """
    Record patient consent.
    """
    expires_at = None
    if consent.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=consent.expires_in_days)
    
    db_consent = PatientConsent(
        patient_id=consent.patient_id,
        consent_type=consent.consent_type,
        granted=consent.granted,
        expires_at=expires_at
    )
    
    db.add(db_consent)
    db.commit()
    db.refresh(db_consent)
    
    # Audit this action
    AuditLogger.log_event(
        db,
        user_id=str(current_user.id),
        action="create_consent",
        resource_type="patient_consent",
        resource_id=str(db_consent.id),
        details={"patient_id": consent.patient_id, "type": consent.consent_type}
    )
    
    return ConsentResponse(
        id=db_consent.id,
        patient_id=db_consent.patient_id,
        consent_type=db_consent.consent_type,
        granted=db_consent.granted,
        timestamp=db_consent.timestamp,
        expires_at=db_consent.expires_at
    )


@router.get("/audit", response_model=List[AuditLogResponse])
async def view_audit_logs(
    limit: int = 100,
    resource_type: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """
    View audit logs (Admin only).
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can view audit logs"
        )
    
    query = db.query(AuditLog).order_by(AuditLog.timestamp.desc())
    
    if resource_type:
        query = query.filter(AuditLog.resource_type == resource_type)
        
    logs = query.limit(limit).all()
    
    return [
        AuditLogResponse(
            id=log.id,
            user_id=log.user_id,
            action=log.action,
            resource_type=log.resource_type,
            resource_id=log.resource_id,
            details=log.details,
            ip_address=log.ip_address,
            timestamp=log.timestamp
        )
        for log in logs
    ]


@router.post("/forget-patient")
async def forget_patient(
    patient_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """
    GDPR 'Right to be Forgotten' - Anonymize patient data.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can perform data deletion"
        )
    
    # In a real system, this would update multiple tables to anonymize data
    # For now, we'll log the request
    
    AuditLogger.log_event(
        db,
        user_id=str(current_user.id),
        action="forget_patient",
        resource_type="patient",
        resource_id=patient_id,
        details={"status": "requested"}
    )
    
    return {"message": f"Patient {patient_id} data deletion requested and logged."}

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
from ..database import get_db
from ..models import User, AppointmentPrediction, AuditLog
from ..schemas import UserProfile, UserStats, UserActivity

router = APIRouter(prefix="/users", tags=["Users"])

@router.get("/profile", response_model=UserProfile)
def get_user_profile(db: Session = Depends(get_db)):
    """Get current user profile with stats."""
    # In a real app, we'd get the current user from auth context
    # For now, we'll return a default admin profile with real system stats
    
    try:
        # Calculate stats
        total_predictions = db.query(AppointmentPrediction).count()
        
        # Get recent activity from AuditLog
        recent_logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(5).all()
        activities = []
        for log in recent_logs:
            activities.append(UserActivity(
                action=f"{log.action} {log.resource_type}",
                time=log.timestamp.strftime("%H:%M"),
                icon="Activity" # Default icon
            ))
            
        # If no logs, add some defaults
        if not activities:
            activities = [
                UserActivity(action="System started", time="Just now", icon="Activity")
            ]

        return UserProfile(
            name="Dr. Admin",
            role="Administrator",
            email="admin@hospital.org",
            location="New York, NY",
            join_date="January 2024",
            avatar="DA",
            stats=UserStats(
                predictions=total_predictions,
                alerts_resolved=12, # Placeholder as we don't track resolution yet
                uptime="99.9%"
            ),
            activities=activities
        )
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

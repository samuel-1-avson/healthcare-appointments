from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import Settings
from ..schemas import SettingsSchema

router = APIRouter(prefix="/settings", tags=["Settings"])

@router.get("/", response_model=SettingsSchema)
def get_settings(db: Session = Depends(get_db)):
    """Get application settings."""
    settings = db.query(Settings).first()
    if not settings:
        # Create default settings
        settings = Settings()
        db.add(settings)
        db.commit()
        db.refresh(settings)
    return settings

@router.put("/", response_model=SettingsSchema)
def update_settings(settings_in: SettingsSchema, db: Session = Depends(get_db)):
    """Update application settings."""
    settings = db.query(Settings).first()
    if not settings:
        settings = Settings()
        db.add(settings)
    
    for key, value in settings_in.dict().items():
        setattr(settings, key, value)
    
    db.commit()
    db.refresh(settings)
    return settings

"""
Database Module
===============
SQLAlchemy setup with graceful fallback.
"""

import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

# Deferred import to avoid circular imports
_engine = None
_SessionLocal = None
Base = declarative_base()


def _get_engine():
    """Get or create database engine."""
    global _engine
    
    if _engine is not None:
        return _engine
    
    from .config import get_settings
    settings = get_settings()
    
    database_url = settings.database_url
    
    # Check if it's a placeholder
    if not database_url or "your-" in database_url.lower():
        logger.warning("DATABASE_URL not configured - using SQLite")
        database_url = "sqlite:///./noshow_dev.db"
    
    try:
        if database_url.startswith("sqlite"):
            _engine = create_engine(
                database_url,
                connect_args={"check_same_thread": False}
            )
        else:
            _engine = create_engine(database_url)
        
        # Test connection
        with _engine.connect() as conn:
            conn.execute("SELECT 1")
        
        logger.info(f"Database connected: {database_url.split('@')[-1] if '@' in database_url else database_url}")
        
    except Exception as e:
        logger.warning(f"Database connection failed: {e}")
        logger.warning("Falling back to SQLite")
        database_url = "sqlite:///./noshow_dev.db"
        _engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False}
        )
    
    return _engine


def _get_session_local():
    """Get SessionLocal factory."""
    global _SessionLocal
    
    if _SessionLocal is None:
        engine = _get_engine()
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    return _SessionLocal


def get_db():
    """
    Dependency for getting database session.
    """
    SessionLocal = _get_session_local()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Alias for backward compatibility
get_db_session = get_db


def init_db():
    """Initialize database tables."""
    engine = _get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized")
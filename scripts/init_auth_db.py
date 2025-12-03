"""
Initialize Authentication Database
==================================
Creates the database tables and a default admin user using SQLAlchemy.
"""

import sys
import os
from pathlib import Path

# Add project root to python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.api.database import engine, SessionLocal, Base
from src.api.models import User
from src.api.auth import get_password_hash

def init_db():
    """Initialize the database."""
    print(f"Initializing database...")
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    print("Tables created.")
    
    # Create session
    db = SessionLocal()
    
    try:
        # Check if admin exists
        admin = db.query(User).filter(User.username == "admin").first()
        
        if not admin:
            print("Creating default admin user...")
            admin_user = User(
                username="admin",
                email="admin@example.com",
                full_name="System Administrator",
                hashed_password=get_password_hash("admin123"),
                role="admin",
                disabled=False
            )
            db.add(admin_user)
            db.commit()
            print("Admin user created (password: admin123)")
        else:
            print("Admin user already exists.")
            
    except Exception as e:
        print(f"Error initializing database: {e}")
        db.rollback()
    finally:
        db.close()
        print("Database initialization complete.")

if __name__ == "__main__":
    init_db()

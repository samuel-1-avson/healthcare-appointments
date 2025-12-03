"""
Audit Logging
=============
Audit trail for compliance and security.
"""

import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class AuditLogger:
    """Static class for audit logging."""
    
    @staticmethod
    def log_event(
        db: Session,
        user_id: Optional[str],
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ) -> None:
        """
        Log an audit event.
        
        Parameters
        ----------
        db : Session
            Database session
        user_id : str, optional
            User performing the action
        action : str
            Action type (create, read, update, delete, etc.)
        resource_type : str
            Type of resource being accessed
        resource_id : str, optional
            ID of specific resource
        details : dict, optional
            Additional details as JSON
        ip_address : str, optional
            Client IP address
        """
        from ..models import AuditLog
        
        try:
            log_entry = AuditLog(
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                details=json.dumps(details) if details else None,
                ip_address=ip_address,
                timestamp=datetime.utcnow()
            )
            
            db.add(log_entry)
            db.commit()
            
            logger.debug(
                f"Audit: user={user_id} action={action} "
                f"resource={resource_type}/{resource_id}"
            )
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            db.rollback()
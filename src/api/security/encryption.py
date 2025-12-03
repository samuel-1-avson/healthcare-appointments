"""
Field-Level Encryption
======================
Custom SQLAlchemy type for encrypted fields.
"""

import os
import base64
import logging
from typing import Any, Optional

from sqlalchemy import TypeDecorator, String
from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

# Get or generate encryption key
def get_encryption_key() -> bytes:
    """Get encryption key from environment or generate one."""
    key = os.environ.get("ENCRYPTION_KEY")
    if key:
        return key.encode()
    else:
        # For development only - generate a key
        # In production, this MUST be set via environment variable
        logger.warning("ENCRYPTION_KEY not set - using generated key (NOT FOR PRODUCTION)")
        return Fernet.generate_key()

_fernet: Optional[Fernet] = None

def get_fernet() -> Fernet:
    """Get Fernet instance (singleton)."""
    global _fernet
    if _fernet is None:
        _fernet = Fernet(get_encryption_key())
    return _fernet


class EncryptedString(TypeDecorator):
    """
    SQLAlchemy type for encrypted string storage.
    
    Encrypts data before storing and decrypts when retrieving.
    Uses Fernet symmetric encryption.
    """
    
    impl = String
    cache_ok = True
    
    def __init__(self, length: int = 500):
        super().__init__()
        self.impl = String(length)
    
    def process_bind_param(self, value: Any, dialect) -> Optional[str]:
        """Encrypt value before storing."""
        if value is None:
            return None
        
        try:
            fernet = get_fernet()
            encrypted = fernet.encrypt(str(value).encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            # Fallback to storing plain (not recommended for production)
            return str(value)
    
    def process_result_value(self, value: Any, dialect) -> Optional[str]:
        """Decrypt value when retrieving."""
        if value is None:
            return None
        
        try:
            fernet = get_fernet()
            decoded = base64.urlsafe_b64decode(value.encode())
            decrypted = fernet.decrypt(decoded)
            return decrypted.decode()
        except InvalidToken:
            # Value might not be encrypted (migration case)
            logger.warning("Could not decrypt value - returning as-is")
            return value
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return value
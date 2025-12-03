# tests/test_security.py
"""
Security Module Tests
=====================
Unit tests for encryption, audit logging, and compliance features.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.api.security.encryption import Encryptor
from src.api.security.audit import AuditLogger
from src.api.models import AuditLog, PatientConsent

# --- Encryption Tests ---

def test_encryption_decryption():
    """Test that data can be encrypted and decrypted correctly."""
    original_text = "Sensitive Patient Data"
    
    # Encrypt
    encrypted = Encryptor.encrypt(original_text)
    assert encrypted != original_text
    assert isinstance(encrypted, str)
    
    # Decrypt
    decrypted = Encryptor.decrypt(encrypted)
    assert decrypted == original_text

def test_encryption_none():
    """Test encryption of None values."""
    assert Encryptor.encrypt(None) is None
    assert Encryptor.decrypt(None) is None

# --- Audit Logging Tests ---

def test_audit_logging():
    """Test that audit logs are created correctly."""
    mock_db = MagicMock()
    
    AuditLogger.log_event(
        db=mock_db,
        user_id="user123",
        action="read",
        resource_type="patient",
        resource_id="pat456",
        details={"reason": "checkup"},
        ip_address="127.0.0.1"
    )
    
    # Verify add was called
    assert mock_db.add.called
    
    # Verify log entry content
    log_entry = mock_db.add.call_args[0][0]
    assert isinstance(log_entry, AuditLog)
    assert log_entry.user_id == "user123"
    assert log_entry.action == "read"
    assert log_entry.resource_type == "patient"
    assert "checkup" in log_entry.details

# --- Compliance Tests ---

def test_patient_consent_model():
    """Test PatientConsent model creation."""
    consent = PatientConsent(
        patient_id="pat789",
        consent_type="sms_notifications",
        granted=True,
        timestamp=datetime.utcnow()
    )
    
    assert consent.patient_id == "pat789"
    assert consent.granted is True
    assert consent.consent_type == "sms_notifications"

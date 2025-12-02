"""
API Configuration Module
========================
Centralized configuration management for the API service.

Uses pydantic-settings for:
- Environment variable loading
- Configuration validation
- Default values
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class Settings(BaseSettings):
    """
    API Settings with environment variable support.
    
    Environment variables override config file values.
    Prefix: NOSHOW_
    
    Example:
        NOSHOW_DEBUG=true
        NOSHOW_MODEL_PATH=/path/to/model.joblib
    """
    
    model_config = SettingsConfigDict(
        env_prefix="NOSHOW_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # API Settings
    api_title: str = "Healthcare No-Show Prediction API"
    api_description: str = "ML-powered API for predicting appointment no-shows"
    api_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    debug: bool = False
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    
    # Model Settings
    model_path: str = "models/production/model.joblib"
    preprocessor_path: str = "models/production/preprocessor.joblib"
    metadata_path: str = "models/production/model_metadata.json"
    model_name: str = "random_forest"
    model_version: str = "1.0.0"
    default_threshold: float = 0.5
    
    # Logging
    log_level: str = "INFO"
    log_requests: bool = True
    log_predictions: bool = True
    
    # CORS
    cors_enabled: bool = True
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000", "http://localhost:8000"]
    
    # Feature defaults
    default_neighbourhood: str = "Unknown"
    default_weekday: str = "Monday"
    
    @field_validator('model_path', 'preprocessor_path')
    @classmethod
    def validate_paths(cls, v: str) -> str:
        """Validate that paths are strings (existence checked at runtime)."""
        return v
    
    @property
    def model_exists(self) -> bool:
        """Check if model file exists."""
        return Path(self.model_path).exists()
    
    @property
    def preprocessor_exists(self) -> bool:
        """Check if preprocessor file exists."""
        return Path(self.preprocessor_path).exists()


class RiskTierConfig:
    """Risk tier configuration."""
    
    TIERS = {
        "CRITICAL": {
            "min_probability": 0.7,
            "color": "#e74c3c",
            "emoji": "🔴",
            "intervention": "Immediate phone call + deposit required"
        },
        "HIGH": {
            "min_probability": 0.5,
            "color": "#e67e22", 
            "emoji": "🟠",
            "intervention": "Phone call + double SMS reminder"
        },
        "MEDIUM": {
            "min_probability": 0.3,
            "color": "#f1c40f",
            "emoji": "🟡",
            "intervention": "Double SMS reminder"
        },
        "LOW": {
            "min_probability": 0.15,
            "color": "#2ecc71",
            "emoji": "🟢",
            "intervention": "Standard SMS reminder"
        },
        "MINIMAL": {
            "min_probability": 0.0,
            "color": "#27ae60",
            "emoji": "⭐",
            "intervention": "No additional intervention needed"
        }
    }
    
    @classmethod
    def get_tier(cls, probability: float) -> str:
        """Get risk tier based on probability."""
        for tier_name, tier_config in cls.TIERS.items():
            if probability >= tier_config["min_probability"]:
                return tier_name
        return "MINIMAL"
    
    @classmethod
    def get_tier_info(cls, probability: float) -> Dict[str, Any]:
        """Get full tier information."""
        tier_name = cls.get_tier(probability)
        tier_config = cls.TIERS[tier_name]
        return {
            "tier": tier_name,
            "probability": probability,
            **tier_config
        }


def load_yaml_config(config_path: str = "config/api_config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file
    
    Returns
    -------
    dict
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        return {}
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache for singleton pattern - settings are loaded once
    and reused throughout the application lifecycle.
    
    Returns
    -------
    Settings
        API settings instance
    """
    return Settings()


def get_risk_tier_config() -> RiskTierConfig:
    """Get risk tier configuration."""
    return RiskTierConfig()
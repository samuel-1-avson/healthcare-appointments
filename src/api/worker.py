# src/api/worker.py
"""
Celery Worker Configuration
===========================
Initializes the Celery application for background task processing.
"""

import os
from celery import Celery

# Get configuration from environment
BROKER_URL = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@rabbitmq:5672//")
BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

# Initialize Celery app
celery_app = Celery(
    "healthcare_worker",
    broker=BROKER_URL,
    backend=BACKEND_URL,
    include=["src.api.tasks"]
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    
    # Celery Beat schedule for periodic tasks
    beat_schedule={
        "retrain-model-weekly": {
            "task": "src.api.tasks.retrain_model",
            "schedule": float(os.getenv("RETRAIN_SCHEDULE_SECONDS", 7 * 24 * 60 * 60)),  # Default: weekly
            "args": (90,),  # Look back 90 days
        },
    },
)


if __name__ == "__main__":
    celery_app.start()

"""
Celery Tasks
============
Background tasks for async predictions.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Try to import Celery, but make it optional
try:
    from celery import Celery
    
    # Create Celery app
    celery_app = Celery(
        'noshow',
        broker='redis://localhost:6379/0',
        backend='redis://localhost:6379/0'
    )
    
    celery_app.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
    )
    
    CELERY_AVAILABLE = True
    
except ImportError:
    logger.warning("Celery not available - async tasks disabled")
    CELERY_AVAILABLE = False
    celery_app = None


def _get_predictor():
    """Get predictor instance (lazy import to avoid circular imports)."""
    from .predict import get_predictor
    return get_predictor()


def _create_appointment(data: Dict[str, Any]):
    """Create AppointmentFeatures from dict."""
    from .schemas import AppointmentFeatures
    return AppointmentFeatures(**data)


if CELERY_AVAILABLE:
    @celery_app.task(name='predict_noshow')
    def predict_noshow_task(appointment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Background task for single prediction.
        
        Parameters
        ----------
        appointment_data : dict
            Appointment features as dictionary
        
        Returns
        -------
        dict
            Prediction result
        """
        predictor = _get_predictor()
        appointment = _create_appointment(appointment_data)
        result = predictor.predict(appointment)
        return result.model_dump()


    @celery_app.task(name='batch_predict')
    def batch_predict_task(appointments_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Background task for batch prediction.
        
        Parameters
        ----------
        appointments_data : list
            List of appointment features as dictionaries
        
        Returns
        -------
        dict
            Batch prediction result
        """
        predictor = _get_predictor()
        appointments = [_create_appointment(data) for data in appointments_data]
        result = predictor.predict_batch(appointments)
        return result.model_dump()

else:
    # Dummy tasks when Celery is not available
    class DummyTask:
        """Dummy task that raises an error."""
        
        def delay(self, *args, **kwargs):
            raise RuntimeError(
                "Async tasks require Celery. "
                "Install with: pip install celery redis"
            )
    
    predict_noshow_task = DummyTask()
    batch_predict_task = DummyTask()
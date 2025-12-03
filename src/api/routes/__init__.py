# src/api/routes/__init__.py
"""
API Routes Package
==================
FastAPI route definitions organized by functionality.

Routes:
- health: Health check and status endpoints
- predictions: Prediction endpoints
- model_info: Model information endpoints
"""

import logging
import traceback
from fastapi import APIRouter

print("DEBUG: Loading api.routes package...")

from .health import router as health_router
from .predictions import router as predictions_router
from .model_info import router as model_info_router
from .monitoring import router as monitoring_router
from .feedback import router as feedback_router
from .compliance import router as compliance_router

logger = logging.getLogger(__name__)

# Import LLM routes
print("DEBUG: Attempting to import LLM routes...")
try:
    from .llm_routes import router as llm_router
    LLM_AVAILABLE = True
    print("DEBUG: [OK] LLM routes imported successfully")
    logger.info(" LLM routes imported successfully")
except Exception as e:
    print(f"DEBUG: [FAIL] Failed to import LLM routes: {e}")
    traceback.print_exc()
    logger.error(f" Failed to import LLM routes: {e}")
    logger.error(traceback.format_exc())
    LLM_AVAILABLE = False

# Import RAG routes
print("DEBUG: Attempting to import RAG routes...")
try:
    from .rag_routes import router as rag_router
    RAG_AVAILABLE = True
    print("DEBUG: [OK] RAG routes imported successfully")
    logger.info(" RAG routes imported successfully")
except Exception as e:
    print(f"DEBUG: [FAIL] Failed to import RAG routes: {e}")
    traceback.print_exc()
    logger.error(f" Failed to import RAG routes: {e}")
    logger.error(traceback.format_exc())
    RAG_AVAILABLE = False

# Create main router
api_router = APIRouter()

# Include sub-routers
api_router.include_router(
    health_router,
    tags=["Health"]
)

api_router.include_router(
    predictions_router,
    prefix="/predict",
    tags=["Predictions"]
)

api_router.include_router(
    model_info_router,
    prefix="/model",
    tags=["Model Info"]
)

if LLM_AVAILABLE:
    api_router.include_router(
        llm_router,
        tags=["LLM"]
    )
else:
    print("DEBUG: LLM routes NOT available, skipping include_router")

if RAG_AVAILABLE:
    api_router.include_router(
        rag_router,
        tags=["RAG"]
    )
else:
    print("DEBUG: RAG routes NOT available, skipping include_router")

# Include monitoring router
api_router.include_router(
    monitoring_router,
    tags=["Monitoring"]
)

# Include feedback router
api_router.include_router(
    feedback_router,
    tags=["Feedback"]
)

# Include compliance router
api_router.include_router(
    compliance_router,
    tags=["Compliance"]
)

__all__ = ["api_router", "health_router"]
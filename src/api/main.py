"""
FastAPI Application
===================
Main FastAPI application factory and configuration.

This module creates and configures the FastAPI application with:
- CORS middleware
- Exception handlers
- Route registration
- OpenAPI documentation
- Startup/shutdown events
"""
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from prometheus_fastapi_instrumentator import Instrumentator

from .config import get_settings, Settings
from .routes import api_router
from .routes.auth import router as auth_router
from .predict import get_predictor, NoShowPredictor
from .schemas import ErrorResponse
from .logging_config import setup_logging
from .cache import RedisClient


# Configure logging
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logger.info("=" * 50)
    logger.info("Starting Healthcare No-Show Prediction API")
    logger.info("=" * 50)
    
    settings = get_settings()
    logger.info(f"API Version: {settings.api_version}")
    logger.info(f"Debug Mode: {settings.debug}")
    
    # Connect to Redis (optional - don't crash if unavailable)
    try:
        redis_client = RedisClient()
        redis_client.connect()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.warning(f"Redis connection failed (continuing without cache): {e}")
    
    # Load model
    try:
        predictor = get_predictor()
        if predictor.is_loaded:
            logger.info("Model loaded successfully")
            logger.info(f"   Model: {predictor.metadata.get('model_name', 'Unknown')}")
            logger.info(f"   Version: {predictor.metadata.get('model_version', '1.0.0')}")
        else:
            logger.warning("Model not loaded - predictions will fail")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    
    logger.info("=" * 50)
    logger.info("API Ready!")
    logger.info(f"Docs: http://{settings.host}:{settings.port}/docs")
    logger.info("=" * 50)
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")
    try:
        RedisClient().close()
    except:
        pass
    logger.info("Goodbye!")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns
    -------
    FastAPI
        Configured FastAPI application
    """
    settings = get_settings()
    
    # Create app
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Initialize Prometheus metrics
    Instrumentator().instrument(app).expose(app)
    
    # Add CORS middleware
    if settings.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests with timing."""
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log request
        if settings.log_requests:
            logger.info(
                f"{request.method} {request.url.path} "
                f"- {response.status_code} "
                f"- {duration*1000:.2f}ms"
            )
        
        # Add timing header
        response.headers["X-Process-Time"] = str(duration)
        
        return response
    
    # Exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, 
        exc: RequestValidationError
    ):
        """Handle validation errors with detailed messages."""
        errors = []
        for error in exc.errors():
            errors.append({
                "field": " -> ".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })
        
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation Error",
                "message": "Invalid input data",
                "details": errors,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP Error",
                "message": exc.detail,
                "status_code": exc.status_code,
                "timestamp": datetime.utcnow().isoformat(),
                "path": str(request.url.path)
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        
        settings = get_settings()
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": str(exc) if settings.debug else "An unexpected error occurred",
                "timestamp": datetime.utcnow().isoformat(),
                "path": str(request.url.path)
            }
        )
    
    # Include routers
    app.include_router(api_router, prefix=settings.api_prefix)
    app.include_router(auth_router, prefix=settings.api_prefix)
    
    # Include health router at root for monitoring
    from .routes.health import router as health_router
    app.include_router(health_router)
    
    # Root redirect to docs
    @app.get("/", include_in_schema=False)
    async def root_redirect():
        """Redirect root to API docs."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/docs")
    
    return app


# Create application instance
app = create_app()


# ==================== CLI ENTRY POINT ====================

def run_server(
    host: str = None,
    port: int = None,
    reload: bool = False,
    workers: int = 1
):
    """
    Run the API server.
    
    Parameters
    ----------
    host : str
        Server host
    port : int
        Server port
    reload : bool
        Enable auto-reload (development)
    workers : int
        Number of worker processes
    """
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "src.api.main:app",
        host=host or settings.host,
        port=port or settings.port,
        reload=reload,
        workers=workers,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    run_server(reload=True)
# Dockerfile
# ============================================================
# Healthcare No-Show Prediction API - Complete Dockerfile
# ============================================================
# Multi-stage build for optimized production image with LLM support
#
# Build:
#   docker build -t noshow-api:latest .
#   docker build --target development -t noshow-api:dev .
#
# Run:
#   docker run -p 8000:8000 --env-file .env noshow-api:latest
#
# ============================================================

# ==================== STAGE 1: Base Dependencies ====================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean


# ==================== STAGE 2: Builder ====================
FROM base as builder

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements files
COPY requirements/ /requirements/

# Install dependencies in order
# 1. Base dependencies (pandas, numpy, sklearn)
RUN pip install -r /requirements/base.txt

# 2. API dependencies (fastapi, uvicorn)
RUN pip install -r /requirements/api.txt

# 3. LLM dependencies (langchain, openai, faiss)
RUN pip install -r /requirements/llm.txt


# ==================== STAGE 3: Development ====================
FROM base as development

# Labels
LABEL stage="development"

ENV APP_HOME=/app \
    PYTHONPATH=/app

WORKDIR ${APP_HOME}

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install development dependencies
COPY requirements/dev.txt /requirements/dev.txt
RUN pip install -r /requirements/dev.txt

# Create directories
RUN mkdir -p models/production data/documents data/vector_store logs evals

# Default command for development (with reload)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# ==================== STAGE 4: Production ====================
FROM python:3.11-slim as production

# Labels
LABEL maintainer="Healthcare Analytics Team" \
    version="1.0.0" \
    description="Healthcare No-Show Prediction API with LLM Support"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    APP_HOME=/app \
    APP_USER=appuser \
    PYTHONPATH=/app \
    # API settings
    NOSHOW_HOST=0.0.0.0 \
    NOSHOW_PORT=8000 \
    NOSHOW_WORKERS=2 \
    NOSHOW_LOG_LEVEL=INFO \
    # Paths
    NOSHOW_MODEL_PATH=/app/models/production/model.joblib \
    NOSHOW_PREPROCESSOR_PATH=/app/models/production/preprocessor.joblib

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 ${APP_USER} && \
    useradd --uid 1000 --gid ${APP_USER} --shell /bin/bash --create-home ${APP_USER}

# Set work directory
WORKDIR ${APP_HOME}

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=${APP_USER}:${APP_USER} src/ ${APP_HOME}/src/
COPY --chown=${APP_USER}:${APP_USER} config/ ${APP_HOME}/config/
COPY --chown=${APP_USER}:${APP_USER} prompts/ ${APP_HOME}/prompts/

# DEBUG: List src directory to verify copy
RUN ls -R ${APP_HOME}/src

# Explicitly copy llm to ensure it exists
COPY --chown=${APP_USER}:${APP_USER} src/llm/ ${APP_HOME}/src/llm/

# Create directories for runtime data
RUN mkdir -p \
    ${APP_HOME}/models/production \
    ${APP_HOME}/data/documents \
    ${APP_HOME}/data/vector_store \
    ${APP_HOME}/data/embeddings_cache \
    ${APP_HOME}/logs \
    ${APP_HOME}/evals/results \
    && chown -R ${APP_USER}:${APP_USER} ${APP_HOME}

# Copy startup script
COPY --chown=${APP_USER}:${APP_USER} scripts/docker-entrypoint.sh ${APP_HOME}/
RUN chmod +x ${APP_HOME}/docker-entrypoint.sh

# Switch to non-root user
USER ${APP_USER}

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint and default command
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["api"]
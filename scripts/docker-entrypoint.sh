#!/bin/bash
# scripts/docker-entrypoint.sh
# ============================================================
# Docker entrypoint script for Healthcare API
# ============================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================
# Initialization
# ============================================================

log_info "Starting Healthcare No-Show Prediction API"
log_info "Python version: $(python --version)"

# Check for required environment variables
if [ -z "$OPENAI_API_KEY" ] && [ "$1" = "api" ]; then
    log_warn "OPENAI_API_KEY not set. LLM features may not work."
fi

# Check for model files
if [ -f "$NOSHOW_MODEL_PATH" ]; then
    log_info "ML Model found: $NOSHOW_MODEL_PATH"
else
    log_warn "ML Model not found at $NOSHOW_MODEL_PATH"
fi

# Check for documents (RAG)
DOC_COUNT=$(find /app/data/documents -name "*.md" 2>/dev/null | wc -l)
log_info "Found $DOC_COUNT policy documents for RAG"

# ============================================================
# Command Handling
# ============================================================

case "$1" in
    api)
        log_info "Starting API server..."
        log_info "Host: ${NOSHOW_HOST:-0.0.0.0}"
        log_info "Port: ${NOSHOW_PORT:-8000}"
        log_info "Workers: ${NOSHOW_WORKERS:-2}"
        
        exec gunicorn src.api.main:app \
            --bind ${NOSHOW_HOST:-0.0.0.0}:${NOSHOW_PORT:-8000} \
            --workers ${NOSHOW_WORKERS:-2} \
            --worker-class uvicorn.workers.UvicornWorker \
            --access-logfile - \
            --error-logfile - \
            --capture-output \
            --log-level ${NOSHOW_LOG_LEVEL:-info}
        ;;
    
    dev)
        log_info "Starting development server with reload..."
        exec uvicorn src.api.main:app \
            --host ${NOSHOW_HOST:-0.0.0.0} \
            --port ${NOSHOW_PORT:-8000} \
            --reload \
            --reload-dir /app/src
        ;;
    
    init-rag)
        log_info "Initializing RAG vector store..."
        exec python -c "
from src.llm.rag import load_policy_documents, VectorStoreManager
docs = load_policy_documents('/app/data/documents')
print(f'Loaded {len(docs)} documents')
manager = VectorStoreManager()
manager.create_from_documents(docs)
manager.save('default')
print('Vector store created!')
"
        ;;
    
    test)
        log_info "Running tests..."
        exec pytest tests/ -v
        ;;
    
    shell)
        log_info "Starting Python shell..."
        exec python
        ;;
    
    bash)
        log_info "Starting bash shell..."
        exec /bin/bash
        ;;
    
    *)
        # Run whatever command was passed
        exec "$@"
        ;;
esac
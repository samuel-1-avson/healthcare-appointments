#!/bin/bash
# ============================================================
# Docker Run Script
# ============================================================

set -e

# Configuration
CONTAINER_NAME="noshow-api"
IMAGE_NAME="noshow-api:latest"
PORT="${API_PORT:-8000}"
MODEL_DIR="${MODEL_DIR:-$(pwd)/models/production}"

echo "=============================================="
echo "Starting Docker Container"
echo "=============================================="
echo "Container: ${CONTAINER_NAME}"
echo "Image: ${IMAGE_NAME}"
echo "Port: ${PORT}"
echo "Model Dir: ${MODEL_DIR}"
echo ""

# Stop existing container if running
docker stop ${CONTAINER_NAME} 2>/dev/null || true
docker rm ${CONTAINER_NAME} 2>/dev/null || true

# Run container
docker run -d \
    --name ${CONTAINER_NAME} \
    --restart unless-stopped \
    -p ${PORT}:8000 \
    -v ${MODEL_DIR}:/app/models/production:ro \
    -v $(pwd)/logs:/app/logs \
    -e NOSHOW_LOG_LEVEL=INFO \
    ${IMAGE_NAME}

echo ""
echo "✅ Container started!"
echo ""

# Show container status
docker ps --filter "name=${CONTAINER_NAME}"

echo ""
echo "📋 View logs: docker logs -f ${CONTAINER_NAME}"
echo "🌐 API docs: http://localhost:${PORT}/docs"
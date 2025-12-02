#!/bin/bash
# ============================================================
# Docker Build Script
# ============================================================

set -e

# Configuration
IMAGE_NAME="noshow-api"
IMAGE_TAG="${1:-latest}"
REGISTRY="${DOCKER_REGISTRY:-}"

echo "=============================================="
echo "Building Docker Image"
echo "=============================================="
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""

# Build image
docker build \
    --target production \
    --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
    --tag "${IMAGE_NAME}:latest" \
    --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
    --build-arg VERSION="${IMAGE_TAG}" \
    .

echo ""
echo "✅ Build complete!"
echo ""

# Show image info
docker images "${IMAGE_NAME}"

# Optional: Push to registry
if [ -n "${REGISTRY}" ]; then
    echo ""
    echo "Pushing to registry: ${REGISTRY}"
    docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    docker push "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    echo "✅ Push complete!"
fi
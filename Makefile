# ============================================================
# Makefile for Healthcare No-Show Prediction API
# ============================================================

.PHONY: help install dev test lint format build run stop logs clean deploy

# Default target
help:
	@echo "Healthcare No-Show Prediction API"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Development:"
	@echo "  install     Install dependencies"
	@echo "  dev         Start development server"
	@echo "  test        Run tests"
	@echo "  lint        Run linter"
	@echo "  format      Format code"
	@echo ""
	@echo "Docker:"
	@echo "  build       Build Docker image"
	@echo "  run         Run Docker container"
	@echo "  stop        Stop Docker container"
	@echo "  logs        View container logs"
	@echo "  clean       Remove containers and images"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy-dev  Deploy development environment"
	@echo "  deploy-prod Deploy production environment"
	@echo ""

# ==================== Development ====================

install:
	pip install -r requirements.txt
	pip install -r requirements-api.txt

dev:
	python serve_model.py --reload --debug

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

# ==================== Docker ====================

build:
	docker build -t noshow-api:latest .

run:
	docker run -d --name noshow-api -p 8000:8000 \
		-v $(PWD)/models/production:/app/models/production:ro \
		noshow-api:latest

stop:
	docker stop noshow-api || true
	docker rm noshow-api || true

logs:
	docker logs -f noshow-api

clean:
	docker stop noshow-api || true
	docker rm noshow-api || true
	docker rmi noshow-api:latest || true
	docker system prune -f

# ==================== Docker Compose ====================

compose-up:
	docker-compose up -d

compose-down:
	docker-compose down

compose-logs:
	docker-compose logs -f

compose-build:
	docker-compose build

# ==================== Deployment ====================

deploy-dev:
	docker-compose -f docker-compose.yaml -f docker-compose.dev.yaml up -d

deploy-prod:
	docker-compose -f docker-compose.yaml -f docker-compose.prod.yaml up -d

# ==================== Model ====================

prepare-model:
	python scripts/prepare_production_model.py

# ==================== Utilities ====================

health-check:
	curl -s http://localhost:8000/health | python -m json.tool

predict-test:
	curl -s -X POST "http://localhost:8000/api/v1/predict" \
		-H "Content-Type: application/json" \
		-d '{"age": 35, "gender": "F", "lead_days": 7, "sms_received": 1}' \
		| python -m json.tool
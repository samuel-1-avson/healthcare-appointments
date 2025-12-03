# Healthcare No-Show Prediction System

> **An intelligent, production-ready ML system for predicting and preventing medical appointment no-shows using machine learning and AI-powered interventions.**

---

## Table of Contents
- [System Overview](#1-system-overview)
- [Architecture Deep Dive](#2-architecture-deep-dive)
- [Data Flow & Request Lifecycle](#3-data-flow--request-lifecycle)
- [Component Details](#4-component-details)
- [API Reference](#5-api-reference)
- [Project Structure](#6-project-structure)
- [Configuration Guide](#7-configuration-guide)
- [Deployment & Operations](#8-deployment--operations)
- [Development Guide](#9-development-guide)

---

## 1. System Overview

### Purpose
The **Healthcare No-Show Prediction System** is a comprehensive ML platform designed to predict patient appointment no-shows and provide actionable intervention recommendations. The system combines traditional machine learning with modern LLM capabilities to deliver:
- **Predictive Analytics**: Real-time risk assessment for individual and batch appointments
- **Explainable AI**: Human-readable explanations for every prediction
- **Intelligent Interventions**: Context-aware recommendations for reducing no-shows
- **Conversational Interface**: RAG-powered AI assistant for policy questions and insights

### Key Capabilities
| Feature | Description | Technology |
|---------|-------------|------------|
| **Real-time Predictions** | Sub-100ms prediction latency | FastAPI + scikit-learn |
| **Batch Processing** | Process 1000s of appointments asynchronously | Celery + RabbitMQ |
| **AI Explanations** | Natural language explanations for predictions | LangChain + GPT-4 |
| **RAG Q&A** | Semantic search over clinic policies | FAISS + OpenAI Embeddings |
| **Risk Tiers** | 5-tier classification (CRITICAL → MINIMAL) | Custom risk scoring |
| **Observability** | Real-time metrics & monitoring | Prometheus + Grafana |

---

## 2. Architecture Deep Dive

### High-Level Architecture

```mermaid
graph TB
    subgraph "User Layer"
        UI[React Frontend<br/>TailwindCSS + Recharts]
    end

    subgraph "API Gateway"
        NGINX[Nginx Load Balancer<br/>3 API Replicas]
    end

    subgraph "Application Layer"
        API1[FastAPI Instance 1]
        API2[FastAPI Instance 2]
        API3[FastAPI Instance 3]
        WORKER[Celery Worker<br/>Async Tasks]
    end

    subgraph "AI/ML Layer"
        MODEL[ML Model<br/>RandomForest/XGBoost]
        LLM[LLM Client<br/>OpenAI/Anthropic]
        RAG[RAG Pipeline<br/>FAISS Vector Store]
    end

    subgraph "Data Layer"
        POSTGRES[(PostgreSQL<br/>Patient Data)]
        REDIS[(Redis Cache<br/>Sessions/Results)]
        RABBITMQ[RabbitMQ<br/>Task Queue]
        VECTOR[(Vector Store<br/>Embeddings)]
    end

    subgraph "Observability"
        PROM[Prometheus<br/>Metrics]
        GRAF[Grafana<br/>Dashboards]
        MLFLOW[MLflow<br/>Model Registry]
    end

    UI --> NGINX
    NGINX --> API1 & API2 & API3
    API1 & API2 & API3 --> MODEL
    API1 & API2 & API3 --> LLM
    API1 & API2 & API3 --> RAG
    API1 & API2 & API3 --> REDIS
    API1 & API2 & API3 --> POSTGRES
    API1 & API2 & API3 --> RABBITMQ
    WORKER --> RABBITMQ
    WORKER --> MODEL
    RAG --> VECTOR
    API1 --> PROM
    PROM --> GRAF
    MODEL -.Registry.-> MLFLOW
```

### Microservices Breakdown

#### 1. Frontend Service
- **Technology**: React 19 + Vite + TypeScript
- **Port**: 3000 (Docker) / 5173 (Dev)
- **Responsibilities**:
  - User authentication & session management
  - Form validation & data entry
  - Real-time prediction visualization
  - Chat interface for AI assistant
  - Dashboard rendering (charts, metrics)
  
#### 2. API Service (3 Replicas)
- **Technology**: FastAPI + Uvicorn
- **Port**: 8000 (internal), exposed via Nginx on 8080
- **Responsibilities**:
  - RESTful endpoints for predictions
  - LLM chat orchestration
  - Authentication & authorization (JWT)
  - Request validation with Pydantic schemas
  - Prometheus metrics export

#### 3. Worker Service
- **Technology**: Celery with RabbitMQ broker
- **Responsibilities**:
  - Asynchronous batch predictions
  - Model retraining tasks
  - Report generation
  - Email/SMS notifications

#### 4. Database Services
- **PostgreSQL**: Stores appointment records, user data, audit logs
- **Redis**: Caches prediction results, LLM sessions, rate limiting
- **RabbitMQ**: Message broker for task distribution

#### 5. AI/ML Services
- **ML Model**: Scikit-learn pipeline (production/model.joblib)
- **LLM API**: OpenAI GPT-4o-mini / Anthropic Claude
- **Vector Store**: FAISS for document embeddings

---

## 3. Data Flow & Request Lifecycle

### Single Prediction Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Nginx
    participant API
    participant Model
    participant LLM
    participant Redis
    participant DB

    User->>Frontend: Fill prediction form
    Frontend->>Frontend: Validate inputs (age, gender, etc.)
    Frontend->>Nginx: POST /api/v1/predict
    Nginx->>API: Route to healthy instance
    
    API->>Redis: Check cache (patient_id + features hash)
    alt Cache Hit
        Redis-->>API: Return cached result
        API-->>Frontend: PredictionResponse (cached)
    else Cache Miss
        API->>Model: Prepare features (30 dims)
        Model->>Model: Preprocess (scale, encode)
        Model->>Model: predict_proba()
        Model-->>API: probability [0.0-1.0]
        API->>API: Calculate risk tier
        API->>LLM: Generate explanation (optional)
        LLM-->>API: Human-readable text
        API->>Redis: Cache result (TTL: 1h)
        API->>DB: Log prediction (audit)
        API-->>Frontend: PredictionResponse + Explanation
    end
    
    Frontend->>Frontend: Render result card
    Frontend->>User: Display risk + interventions
```

### Chat/RAG Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant API
    participant Agent
    participant VectorDB
    participant LLM
    participant Redis

    User->>Frontend: "What's the no-show policy?"
    Frontend->>API: POST /api/v1/llm/chat
    API->>Redis: Get session history (session_id)
    API->>Agent: Invoke orchestrator
    Agent->>Agent: Classify intent (policy_query)
    Agent->>VectorDB: Semantic search (embeddings)
    VectorDB-->>Agent: Top 3 relevant docs
    Agent->>LLM: Generate response (context + history)
    LLM-->>Agent: Answer with citations
    Agent->>Redis: Save message to history
    Agent-->>API: ChatResponse
    API-->>Frontend: Display message
    Frontend->>User: Render formatted answer
```

---

## 4. Component Details

### 4.1 Backend API (`src/api/`)

#### Core Modules

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `main.py` | Application factory | `create_app()`, lifespan events |
| `config.py` | Settings management | Environment vars, risk tiers |
| `predict.py` | ML inference engine | `NoShowPredictor.predict()` |
| `schemas.py` | Pydantic models | Request/response validation |
| `auth.py` | JWT authentication | Token generation, verification |
| `cache.py` | Redis client | `RedisClient.get()`, `set()` |

#### API Routes

**1. `/api/v1/predict` (Predictions)**
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions (up to 1000)
- `POST /predict/quick` - Fast prediction with minimal inputs
- `GET /predict/thresholds` - Risk tier configuration
- `GET /predict/task/{task_id}` - Async task status

**2. `/api/v1/llm` (LLM Services)**
- `POST /llm/chat` - Conversational AI (orchestrator)
- `POST /llm/chat/agent` - Tool-using agent
- `POST /llm/explain` - Generate prediction explanation
- `POST /llm/intervention` - Get intervention recommendations
- `POST /llm/predict-and-explain` - Combined endpoint
- `GET /llm/sessions` - List active chat sessions
- `DELETE /llm/sessions/{id}` - Clear session

**3. `/api/v1/rag` (RAG Management)**
- `POST /rag/ingest` - Add documents to vector store
- `POST /rag/query` - Direct vector search
- `GET /rag/documents` - List indexed documents
- `DELETE /rag/documents/{id}` - Remove document

**4. `/api/v1/auth` (Authentication)**
- `POST /auth/token` - Get JWT token
- `POST /auth/refresh` - Refresh token
- `GET /auth/me` - Current user info

**5. `/health` (Monitoring)**
- `GET /health` - Liveness probe
- `GET /health/ready` - Readiness probe
- `GET /metrics` - Prometheus metrics

### 4.2 ML Engine (`src/api/predict.py`)

#### NoShowPredictor Class

**Responsibilities:**
- Model loading and singleton management
- Feature engineering (30 features from 14 inputs)
- Preprocessing pipeline execution
- Risk tier assignment
- Explanation generation

**Feature Engineering:**
```python
Input Features (14):
  - age, gender, scholarship, hypertension, diabetes, alcoholism
  - handicap, sms_received, lead_days, weekday, hour
  - neighbourhood, prev_noshow_rate, avg_lead_time

Engineered Features (16):
  - age_group, is_weekend, is_morning/afternoon/evening
  - lead_time_category, high_risk_neighbourhood
  - chronic_disease_count, multiple_conditions
  - sms_effective (interaction)
  - age_lead_interaction, etc.

Total: 30 features
```

**Risk Tiers:**
```python
CRITICAL:  0.70+ → Phone call + deposit required
HIGH:      0.50+ → Phone call + double SMS
MEDIUM:    0.30+ → Double SMS reminder
LOW:       0.15+ → Standard SMS
MINIMAL:   0.00+ → No extra intervention
```

### 4.3 LLM System (`src/llm/`)

#### Architecture Components

**1. LLM Client (`client.py`)**
- Multi-provider support (OpenAI, Anthropic)
- Automatic retry with exponential backoff
- Token counting and cost tracking
- Response caching

**2. RAG Pipeline (`rag/`)**
- **Document Loader**: Ingests PDFs, DOCX, MD files
- **Chunking Strategy**: Recursive character splitter (500 chars, 50 overlap)
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Store**: FAISS (local) or Pinecone (prod)
- **Retriever**: Similarity search with MMR re-ranking

**3. Chains (`chains/`)**
- `RiskExplanationChain`: Generates natural language explanations
- `InterventionChain`: Recommends actions based on risk
- `PolicyChain`: Answers policy questions from docs

**4. Agents (`agents/`)**
- `HealthcareAgent`: Tool-using agent with prediction API access
- Tools: `PredictNoShow`, `ExplainRisk`, `SearchPolicies`

**5. Evaluation (`evaluation/`)**
- Ragas metrics: faithfulness, answer relevancy, context precision
- Hallucination detection
- Regression testing suite

### 4.4 Frontend (`frontend/src/`)

#### Component Hierarchy

```
App.tsx (Root)
├── Layout.tsx (Navigation + Theme)
│   ├── Header (Logo, Tab Navigation, Dark Mode Toggle)
│   └── Main Content Area
│
├── PredictionForm.tsx (Patient Data Entry)
│   ├── Form validation (React Hook Form)
│   ├── Dropdown menus (neighborhoods, weekdays)
│   └── Submit button with loading state
│
├── PredictionResult.tsx (Risk Display)
│   ├── Risk tier badge (animated gradient)
│   ├── Probability gauge (circular progress)
│   ├── Intervention card
│   └── Feature contributions chart
│
├── ChatAssistant.tsx (LLM Interface)
│   ├── Message history (scrollable)
│   ├── Input field with auto-resize
│   ├── Typing indicator
│   └── Tool call badges
│
├── ModelDashboard.tsx (Analytics)
│   ├── Metrics cards (accuracy, AUC-ROC, F1)
│   ├── Confusion matrix heatmap
│   ├── Feature importance bar chart
│   └── Calibration curve
│
└── BatchUpload.tsx (CSV Upload)
    ├── Drag-and-drop zone (react-dropzone)
    ├── Preview table
    └── Download results button
```

#### State Management
- **Local State**: `useState` for form inputs, loading states
- **API Client**: Axios with base URL configuration
- **Caching**: No external library (relies on backend caching)

---

## 5. API Reference

### Prediction Endpoints

#### POST `/api/v1/predict`

**Request:**
```json
{
  "age": 35,
  "gender": "F",
  "scholarship": false,
  "hypertension": false,
  "diabetes": false,
  "alcoholism": false,
  "handicap": 0,
  "sms_received": 1,
  "lead_days": 7,
  "weekday": "Monday",
  "hour": 14,
  "neighbourhood": "JARDIM CAMBURI",
  "prev_noshow_rate": 0.2,
  "avg_lead_time": 10.5
}
```

**Response:**
```json
{
  "probability": 0.34,
  "prediction": "SHOW",
  "risk_assessment": {
    "tier": "MEDIUM",
    "probability": 0.34,
    "color": "#f1c40f",
    "intervention": "Double SMS reminder",
    "emoji": "🟡"
  },
  "intervention": {
    "primary_action": "Send automated SMS 24h before appointment",
    "secondary_action": "Follow-up call if no confirmation",
    "timeline": "1-2 days before appointment"
  },
  "metadata": {
    "model_version": "1.0.0",
    "timestamp": "2025-12-03T13:55:00Z",
    "latency_ms": 45
  }
}
```

#### POST `/api/v1/llm/chat`

**Request:**
```json
{
  "message": "Why did this patient get a high risk score?",
  "session_id": "abc123",
  "context": {
    "probability": 0.65,
    "patient_data": { "age": 25, "lead_days": 2, "sms_received": 0 }
  }
}
```

**Response:**
```json
{
  "response": "This patient received a HIGH risk score (65% no-show probability) primarily due to:\n\n1. **Short lead time** (2 days) - patients scheduled closer to appointment date show more often\n2. **No SMS received** - reminder messages reduce no-show rates by ~30%\n3. **Younger age** (25) - younger demographics have higher no-show rates\n\nRecommended intervention: Send immediate SMS reminder + phone call confirmation.",
  "session_id": "abc123",
  "intent": "prediction_explanation",
  "tool_calls": [
    {"tool": "ExplainRisk", "input": {"probability": 0.65}}
  ],
  "metadata": {
    "model": "gpt-4o-mini",
    "tokens": 156,
    "latency_ms": 892
  }
}
```

---

## 6. Project Structure

```
healthcare-appointments/
│
├── frontend/                         # React TypeScript Application
│   ├── src/
│   │   ├── components/              # UI Components
│   │   │   ├── Layout.tsx           # Navigation shell
│   │   │   ├── PredictionForm.tsx   # Patient data entry
│   │   │   ├── PredictionResult.tsx # Risk display
│   │   │   ├── ChatAssistant.tsx    # LLM chat interface
│   │   │   ├── ModelDashboard.tsx   # Analytics charts
│   │   │   └── BatchUpload.tsx      # CSV upload
│   │   ├── services/
│   │   │   └── api.ts               # Axios client
│   │   ├── types/
│   │   │   └── index.ts             # TypeScript interfaces
│   │   └── App.tsx                  # Root component
│   ├── package.json                 # Dependencies (React 19, Vite, Recharts)
│   └── Dockerfile                   # Production build (Nginx)
│
├── src/                             # Python Backend
│   ├── api/                         # FastAPI Application
│   │   ├── routes/
│   │   │   ├── predictions.py       # ML prediction endpoints
│   │   │   ├── llm_routes.py        # LLM chat endpoints
│   │   │   ├── rag_routes.py        # RAG management
│   │   │   ├── auth.py              # JWT authentication
│   │   │   ├── health.py            # Healthchecks
│   │   │   └── model_info.py        # Model metadata
│   │   ├── main.py                  # App factory
│   │   ├── config.py                # Settings (Pydantic)
│   │   ├── predict.py               # ML inference logic
│   │   ├── schemas.py               # Request/response models
│   │   ├── auth.py                  # JWT utilities
│   │   ├── cache.py                 # Redis client
│   │   └── tasks.py                 # Celery tasks
│   │
│   ├── llm/                         # LLM Integration
│   │   ├── client.py                # Multi-provider LLM client
│   │   ├── config.py                # LLM settings
│   │   ├── rag/
│   │   │   ├── document_loader.py   # PDF/DOCX ingestion
│   │   │   ├── chunking.py          # Text splitting
│   │   │   ├── embeddings.py        # OpenAI embeddings
│   │   │   ├── vector_store.py      # FAISS/Pinecone wrapper
│   │   │   └── retriever.py         # Similarity search
│   │   ├── chains/
│   │   │   ├── explanation.py       # Risk explanation chain
│   │   │   ├── intervention.py      # Recommendation chain
│   │   │   └── policy.py            # Policy Q&A chain
│   │   ├── agents/
│   │   │   ├── healthcare_agent.py  # Tool-using agent
│   │   │   └── orchestrator.py      # Intent routing
│   │   ├── tools/
│   │   │   ├── prediction_tool.py   # API call wrapper
│   │   │   └── policy_tool.py       # Document search
│   │   └── evaluation/
│   │       ├── ragas_eval.py        # Ragas metrics
│   │       └── regression.py        # Test suite
│   │
│   ├── ml/                          # ML Pipeline (future refactor)
│   ├── data_cleaner.py              # Data preprocessing
│   ├── feature_engineer.py          # Feature transformations
│   └── utils.py                     # Utilities
│
├── models/                          # Saved ML Artifacts
│   ├── production/
│   │   ├── model.joblib             # Trained model (RandomForest)
│   │   ├── preprocessor.joblib      # ColumnTransformer
│   │   └── model_metadata.json      # Version, metrics, features
│   ├── baseline/                    # Initial models
│   └── tuned/                       # Hyperparameter tuned
│
├── notebooks/                       # Jupyter Notebooks
│   ├── healthcare_appointments_eda.ipynb       # Week 1: EDA
│   ├── week5_baseline_models.ipynb             # Week 5: ML baselines
│   ├── week6_tuning_interpretability.ipynb     # Week 6: Tuning + SHAP
│   ├── week7_deployment.ipynb                  # Week 7: FastAPI
│   ├── week9_prompt_engineering.ipynb          # Week 9: Prompts
│   ├── week10_langchain.ipynb                  # Week 10: LangChain
│   ├── week11_rag.ipynb                        # Week 11: RAG
│   └── week12_evaluation.ipynb                 # Week 12: Ragas
│
├── sql_analytics/                   # SQL Reporting
│   ├── queries.sql                  # 10 analytical queries
│   ├── run_queries.py               # Execute and export results
│   ├── SQL_ANALYTICS_REPORT.md      # Insights document
│   └── results/                     # CSV exports
│
├── data/                            # Data Assets
│   ├── raw/                         # Original CSV
│   ├── processed/                   # Cleaned data
│   └── documents/                   # RAG corpus (policies, reports)
│
├── tests/                           # Test Suite
│   ├── test_predict.py              # ML inference tests
│   ├── test_llm.py                  # LLM integration tests
│   └── test_api.py                  # API endpoint tests
│
├── scripts/                         # Utility Scripts
│   ├── train_model.py               # Model training pipeline
│   ├── generate_data.py             # Synthetic data generation
│   └── check_imports.py             # Dependency verification
│
├── config/                          # Configuration Files
│   ├── api_config.yaml              # API settings
│   └── prometheus.yml               # Metrics scraping
│
├── docker-compose.yaml              # Full stack orchestration
├── Dockerfile                       # API service image
├── start-system.ps1                 # Windows startup script
├── .env.example                     # Environment template
└── requirements.txt                 # Python dependencies
```

---

## 7. Configuration Guide

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Configuration
NOSHOW_DEBUG=false
NOSHOW_HOST=0.0.0.0
NOSHOW_PORT=8000
NOSHOW_LOG_LEVEL=INFO

# Model Paths
NOSHOW_MODEL_PATH=models/production/model.joblib
NOSHOW_PREPROCESSOR_PATH=models/production/preprocessor.joblib

# LLM Settings
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
NOSHOW_LLM_DEFAULT_PROVIDER=openai
NOSHOW_LLM_DEFAULT_MODEL=gpt-4o-mini
NOSHOW_LLM_CACHE_ENABLED=true

# Database
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin123
POSTGRES_DB=healthcare
NOSHOW_DATABASE_URL=postgresql://admin:admin123@postgres:5432/healthcare

# Redis
NOSHOW_REDIS_HOST=redis
NOSHOW_REDIS_PORT=6379

# Security
NOSHOW_SECRET_KEY=change_me_in_production
```

### Risk Tier Configuration

Edit `src/api/config.py` to customize risk thresholds:

```python
TIERS = {
    "CRITICAL": {"min_probability": 0.7, "intervention": "..."},
    "HIGH": {"min_probability": 0.5, "intervention": "..."},
    # ...
}
```

---

## 8. Deployment & Operations

### Local Development

```powershell
# 1. Start backend
cd healthcare-appointments
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.api.main:app --reload --port 8000

# 2. Start frontend
cd frontend
npm install
npm run dev
```

### Docker Production

```powershell
# Full stack with all services
.\start-system.ps1 -Mode prod

# Or manually
docker-compose up -d

# Check logs
docker-compose logs -f api

# Scale API replicas
docker-compose up -d --scale api=5
```

### Monitoring

- **API Docs**: http://localhost:8000/docs
- **Frontend**: http://localhost:3000
- **Grafana**: http://localhost:3001
- **Prometheus**: http://localhost:9090
- **MLflow**: http://localhost:5000

---

## 9. Development Guide

### Adding a New Endpoint

1. Define schema in `src/api/schemas.py`
2. Create route in `src/api/routes/`
3. Register router in `src/api/main.py`
4. Add tests in `tests/`

### Retraining the Model

```bash
python train_model.py --data data/processed/cleaned.csv --output models/production/
```

### Running Tests

```bash
# Unit tests
pytest tests/ -v

# Integration tests
pytest tests/test_integration.py -v

# Coverage report
pytest --cov=src tests/
```

---

## License
MIT License - See LICENSE file for details.
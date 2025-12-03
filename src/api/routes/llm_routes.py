# src/api/routes/llm_routes.py
"""
LLM API Routes
==============
FastAPI endpoints for LLM-powered features.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import traceback
import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..config import get_settings, Settings
from ..schemas import AppointmentFeatures, PredictionResponse
from ..auth import get_current_user, User
from ..cache import cache_response

# LLM imports
from src.llm.chains import RiskExplanationChain, InterventionChain
from src.llm.chains.orchestrator import HealthcareOrchestrator, IntentType
from src.llm.agents import HealthcareAgent, create_healthcare_agent
from src.llm.tracing import get_metrics
from src.llm.memory.conversation_memory import get_memory_manager
from src.llm.config import get_llm_config


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/llm", tags=["LLM"])


# ==================== Request/Response Schemas ====================

class ChatMessage(BaseModel):
    """Chat message."""
    role: str = Field(description="Message role: user or assistant")
    content: str = Field(description="Message content")


class ChatRequest(BaseModel):
    """Chat request."""
    message: str = Field(description="User message")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation continuity")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class ChatResponse(BaseModel):
    """Chat response."""
    response: str = Field(description="Assistant response")
    session_id: str = Field(description="Session ID")
    intent: Optional[str] = Field(default=None, description="Detected intent")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default=None, description="Tools used")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExplanationRequest(BaseModel):
    """Request for prediction explanation."""
    probability: float = Field(ge=0, le=1, description="No-show probability")
    risk_tier: str = Field(description="Risk tier: CRITICAL, HIGH, MEDIUM, LOW, MINIMAL")
    patient_data: Dict[str, Any] = Field(description="Patient and appointment data")
    model_factors: Optional[Dict[str, Any]] = Field(default=None, description="Model feature importance")
    patient_history: Optional[Dict[str, Any]] = Field(default=None, description="Patient history")


class ExplanationResponse(BaseModel):
    """Explanation response."""
    explanation: str = Field(description="Human-readable explanation")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InterventionRequest(BaseModel):
    """Request for intervention recommendation."""
    probability: float = Field(ge=0, le=1)
    risk_tier: str
    patient_data: Dict[str, Any]
    constraints: Optional[Dict[str, Any]] = Field(default=None)


class SessionInfo(BaseModel):
    """Session information."""
    session_id: str
    message_count: int
    created_at: str
    last_activity: str


# ==================== Dependency Injection ====================

def get_orchestrator() -> HealthcareOrchestrator:
    """Get healthcare orchestrator instance."""
    return HealthcareOrchestrator()


def get_agent() -> HealthcareAgent:
    """Get healthcare agent instance."""
    return create_healthcare_agent()


def get_explanation_chain() -> RiskExplanationChain:
    """Get explanation chain instance."""
    return RiskExplanationChain()


def get_intervention_chain() -> InterventionChain:
    """Get intervention chain instance."""
    return InterventionChain()


# ==================== Endpoints ====================

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with Healthcare Assistant",
    description="Send a message to the healthcare assistant and get a response"
)
async def chat(
    chat_request: ChatRequest,
    orchestrator: HealthcareOrchestrator = Depends(get_orchestrator),
    user: User = Depends(get_current_user)
) -> ChatResponse:
    """
    Chat with the healthcare assistant.
    
    The assistant can:
    - Explain no-show predictions
    - Provide intervention recommendations
    - Answer policy questions
    - Handle general conversation
    """
    # Generate session ID if not provided
    session_id = chat_request.session_id or f"session-{datetime.utcnow().timestamp()}"
    
    try:
        # Run synchronously to debug crash
        result = orchestrator.process(
            message=chat_request.message,
            session_id=session_id,
            context=chat_request.context
        )
        
        return ChatResponse(
            response=result["output"],
            session_id=session_id,
            intent=result.get("intent"),
            metadata=result.get("metadata", {})
        )
    
    except BaseException as e:
        logger.error(f"Chat error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/agent",
    response_model=ChatResponse,
    summary="Chat with Agent (Tool-enabled)",
    description="Chat with the agent that can call tools like the prediction API"
)
async def chat_with_agent(
    chat_request: ChatRequest,
    agent: HealthcareAgent = Depends(get_agent),
    user: User = Depends(get_current_user)
) -> ChatResponse:
    """
    Chat with the tool-enabled agent.
    
    The agent can autonomously:
    - Call the prediction API to get risk assessments
    - Make batch predictions
    - Combine tool results with explanations
    
    Use this endpoint when you want the AI to actively make predictions
    based on your queries.
    """
    session_id = chat_request.session_id or agent.create_session()
    
    try:
        result = agent.chat(session_id, chat_request.message)
        
        return ChatResponse(
            response=result["output"],
            session_id=session_id,
            tool_calls=result.get("tool_calls"),
            metadata=result.get("metadata", {})
        )
    
    except Exception as e:
        logger.error(f"Agent chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/explain",
    response_model=ExplanationResponse,
    summary="Explain Prediction",
    description="Get a human-readable explanation of a no-show prediction"
)
@cache_response(expire=3600)
async def explain_prediction(
    request: ExplanationRequest,
    chain: RiskExplanationChain = Depends(get_explanation_chain),
    user: User = Depends(get_current_user)
) -> ExplanationResponse:
    """
    Generate a human-readable explanation for a prediction.
    
    Provide the prediction results and patient data to get:
    - Summary of risk level
    - Key contributing factors
    - Recommended actions
    """
    try:
        explanation = chain.explain(
            probability=request.probability,
            risk_tier=request.risk_tier,
            patient_data=request.patient_data,
            model_factors=request.model_factors,
            patient_history=request.patient_history
        )
        
        return ExplanationResponse(
            explanation=explanation,
            metadata={
                "risk_tier": request.risk_tier,
                "probability": request.probability,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/intervention",
    summary="Get Intervention Recommendation",
    description="Get intervention recommendations based on risk assessment"
)
@cache_response(expire=3600)
async def get_intervention(
    request: InterventionRequest,
    chain: InterventionChain = Depends(get_intervention_chain),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate intervention recommendations.
    
    Based on the risk level and patient profile, get:
    - Primary recommended action
    - Timeline for intervention
    - Communication scripts
    - Backup plans
    """
    try:
        recommendation = chain.recommend(
            probability=request.probability,
            risk_tier=request.risk_tier,
            patient_data=request.patient_data,
            constraints=request.constraints
        )
        
        return {
            "recommendation": recommendation,
            "risk_tier": request.risk_tier,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Intervention error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/predict-and-explain",
    summary="Predict and Explain",
    description="Make a prediction and get an explanation in one call"
)
async def predict_and_explain(
    appointment: AppointmentFeatures,
    chain: RiskExplanationChain = Depends(get_explanation_chain),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Combined endpoint that:
    1. Makes a no-show prediction
    2. Generates a human-readable explanation
    
    Returns both the raw prediction and the explanation.
    """
    from ..predict import get_predictor
    
    # Get prediction
    predictor = get_predictor()
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not available")
    
    prediction_result = predictor.predict(appointment)
    
    # Generate explanation
    patient_data = appointment.model_dump()
    
    explanation = chain.explain(
        probability=prediction_result.probability,
        risk_tier=prediction_result.risk.tier,
        patient_data=patient_data
    )
    
    return {
        "prediction": prediction_result.model_dump(),
        "explanation": explanation,
        "timestamp": datetime.utcnow().isoformat()
    }


# ==================== Session Management ====================

@router.get(
    "/sessions",
    response_model=List[SessionInfo],
    summary="List Active Sessions",
    description="Get list of active conversation sessions"
)
async def list_sessions(
    user: User = Depends(get_current_user)
) -> List[SessionInfo]:
    """List all active conversation sessions."""
    manager = get_memory_manager()
    sessions = manager.list_sessions()
    
    return [
        SessionInfo(
            session_id=s["session_id"],
            message_count=s["message_count"],
            created_at=s["created_at"],
            last_activity=s["last_activity"]
        )
        for s in sessions
    ]


@router.get(
    "/sessions/{session_id}/history",
    summary="Get Session History",
    description="Get conversation history for a session"
)
async def get_session_history(
    session_id: str,
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get conversation history for a session."""
    manager = get_memory_manager()
    
    try:
        history = manager.get_history(session_id)
        messages = [
            {"role": "user" if hasattr(m, "type") and m.type == "human" else "assistant", 
             "content": m.content}
            for m in history.messages
        ]
        
        return {
            "session_id": session_id,
            "messages": messages,
            "summary": history.get_summary()
        }
    
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")


@router.delete(
    "/sessions/{session_id}",
    summary="Delete Session",
    description="Delete a conversation session"
)
async def delete_session(
    session_id: str,
    user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """Delete a conversation session."""
    manager = get_memory_manager()
    
    if manager.delete_session(session_id):
        return {"status": "deleted", "session_id": session_id}
    else:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")


# ==================== Metrics & Health ====================

@router.get(
    "/metrics",
    summary="Get LLM Metrics",
    description="Get usage metrics for LLM operations"
)
async def get_llm_metrics() -> Dict[str, Any]:
    """Get LLM usage metrics."""
    metrics = get_metrics()
    return metrics.get_summary()


@router.get(
    "/health",
    summary="Health Check",
    description="Check if LLM service is healthy"
)
async def health_check() -> Dict[str, str]:
    """Check if LLM service is healthy."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
# src/api/routes/feedback.py
"""
Feedback API Routes
===================
Endpoints for collecting and analyzing user feedback on LLM responses.
"""

import logging
from typing import Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ..database import get_db_session
from ..models import LLMFeedback
from ...llm.metrics import get_metrics_tracker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["feedback"])


# Schemas
class FeedbackSubmission(BaseModel):
    """Feedback submission request."""
    request_id: str = Field(..., description="LLM request ID")
    endpoint: str = Field(..., description="Endpoint that was called")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 (poor) to 5 (excellent)")
    comment: Optional[str] = Field(None, max_length=1000, description="Optional feedback comment")
    user_id: Optional[str] = Field(None, description="User identifier (optional)")


class FeedbackResponse(BaseModel):
    """Feedback response."""
    id: int
    request_id: str
    endpoint: str
    rating: int
    comment: Optional[str]
    created_at: datetime


class FeedbackStats(BaseModel):
    """Feedback statistics."""
    endpoint: str
    total_responses: int
    average_rating: float
    rating_distribution: dict
    recent_comments: list


@router.post("/", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    feedback: FeedbackSubmission,
    db: Session = Depends(get_db_session)
):
    """
    Submit feedback for an LLM response.
    
    Allows users to rate LLM responses and provide optional comments.
    """
    try:
        # Create feedback record
        db_feedback = LLMFeedback(
            request_id=feedback.request_id,
            endpoint=feedback.endpoint,
            rating=feedback.rating,
            comment=feedback.comment,
            user_id=feedback.user_id,
            created_at=datetime.utcnow()
        )
        
        db.add(db_feedback)
        db.commit()
        db.refresh(db_feedback)
        
        # Update metrics
        metrics_tracker = get_metrics_tracker()
        metrics_tracker.track_feedback(feedback.endpoint, feedback.rating)
        
        logger.info(
            f"Feedback submitted: endpoint={feedback.endpoint}, "
            f"rating={feedback.rating}, request_id={feedback.request_id}"
        )
        
        return FeedbackResponse(
            id=db_feedback.id,
            request_id=db_feedback.request_id,
            endpoint=db_feedback.endpoint,
            rating=db_feedback.rating,
            comment=db_feedback.comment,
            created_at=db_feedback.created_at
        )
        
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback"
        )


@router.get("/stats/{endpoint}", response_model=FeedbackStats)
async def get_feedback_stats(
    endpoint: str,
    days: int = 30,
    db: Session = Depends(get_db_session)
):
    """
    Get feedback statistics for an endpoint.
    
    Parameters
    ----------
    endpoint : str
        Endpoint to get stats for
    days : int
        Number of days to look back (default: 30)
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Query feedback
        feedbacks = db.query(LLMFeedback)\
            .filter(LLMFeedback.endpoint == endpoint)\
            .filter(LLMFeedback.created_at >= cutoff_date)\
            .all()
        
        if not feedbacks:
            return FeedbackStats(
                endpoint=endpoint,
                total_responses=0,
                average_rating=0.0,
                rating_distribution={},
                recent_comments=[]
            )
        
        # Calculate statistics
        total_responses = len(feedbacks)
        average_rating = sum(f.rating for f in feedbacks) / total_responses
        
        # Rating distribution
        rating_distribution = {i: 0 for i in range(1, 6)}
        for f in feedbacks:
            rating_distribution[f.rating] += 1
        
        # Recent comments (last 10)
        recent_comments = [
            {
                "rating": f.rating,
                "comment": f.comment,
                "created_at": f.created_at.isoformat()
            }
            for f in sorted(feedbacks, key=lambda x: x.created_at, reverse=True)[:10]
            if f.comment
        ]
        
        return FeedbackStats(
            endpoint=endpoint,
            total_responses=total_responses,
            average_rating=round(average_rating, 2),
            rating_distribution=rating_distribution,
            recent_comments=recent_comments
        )
        
    except Exception as e:
        logger.error(f"Failed to get feedback stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve feedback statistics"
        )


@router.get("/summary")
async def get_overall_feedback_summary(
    days: int = 30,
    db: Session = Depends(get_db_session)
):
    """
    Get overall feedback summary across all endpoints.
    
    Parameters
    ----------
    days : int
        Number of days to look back (default: 30)
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Query all feedback
        feedbacks = db.query(LLMFeedback)\
            .filter(LLMFeedback.created_at >= cutoff_date)\
            .all()
        
        if not feedbacks:
            return {
                "period_days": days,
                "total_responses": 0,
                "average_rating": 0.0,
                "by_endpoint": {}
            }
        
        # Group by endpoint
        by_endpoint = {}
        for f in feedbacks:
            if f.endpoint not in by_endpoint:
                by_endpoint[f.endpoint] = []
            by_endpoint[f.endpoint].append(f.rating)
        
        # Calculate stats per endpoint
        endpoint_stats = {}
        for endpoint, ratings in by_endpoint.items():
            endpoint_stats[endpoint] = {
                "total_responses": len(ratings),
                "average_rating": round(sum(ratings) / len(ratings), 2)
            }
        
        # Overall stats
        all_ratings = [f.rating for f in feedbacks]
        
        return {
            "period_days": days,
            "total_responses": len(all_ratings),
            "average_rating": round(sum(all_ratings) / len(all_ratings), 2),
            "by_endpoint": endpoint_stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get feedback summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve feedback summary"
        )

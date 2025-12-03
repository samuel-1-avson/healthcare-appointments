# src/api/cost_tracking.py
"""
LLM Cost Tracking
=================
Track token usage and costs for LLM API calls.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict
from sqlalchemy.orm import Session

from .database import get_db_session
from .models import LLMUsage, UserQuota

logger = logging.getLogger(__name__)

# Pricing per 1K tokens (as of Dec 2024)
MODEL_PRICING = {
    "gpt-4": {
        "input": 0.03,
        "output": 0.06
    },
    "gpt-4-turbo": {
        "input": 0.01,
        "output": 0.03
    },
    "gpt-3.5-turbo": {
        "input": 0.0005,
        "output": 0.0015
    },
    "gpt-3.5-turbo-16k": {
        "input": 0.003,
        "output": 0.004
    },
}


class CostTracker:
    """Track and monitor LLM usage costs."""
    
    def __init__(self, session: Optional[Session] = None):
        """
        Initialize cost tracker.
        
        Parameters
        ----------
        session : Session, optional
            Database session
        """
        self.session = session or next(get_db_session())
    
    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost for LLM API call.
        
        Parameters
        ----------
        model : str
            Model name
        input_tokens : int
            Number of input tokens
        output_tokens : int
            Number of output tokens
            
        Returns
        -------
        float
            Cost in USD
        """
        if model not in MODEL_PRICING:
            logger.warning(f"Unknown model for pricing: {model}")
            return 0.0
        
        pricing = MODEL_PRICING[model]
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        total_cost = input_cost + output_cost
        
        return round(total_cost, 6)
    
    def log_usage(
        self,
        user_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        endpoint: str,
        request_id: Optional[str] = None
    ) -> LLMUsage:
        """
        Log LLM usage to database.
        
        Parameters
        ----------
        user_id : str
            User identifier
        model : str
            Model name
        input_tokens : int
            Number of input tokens
        output_tokens : int
            Number of output tokens
        endpoint : str
            API endpoint called
        request_id : str, optional
            Request ID for tracking
            
        Returns
        -------
        LLMUsage
            Created usage record
        """
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        
        usage = LLMUsage(
            user_id=user_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost=cost,
            endpoint=endpoint,
            request_id=request_id,
            created_at=datetime.utcnow()
        )
        
        self.session.add(usage)
        self.session.commit()
        
        logger.info(
            f"Logged LLM usage: user={user_id}, model={model}, "
            f"tokens={input_tokens + output_tokens}, cost=${cost:.4f}"
        )
        
        return usage
    
    def get_user_usage(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict:
        """
        Get user's LLM usage summary.
        
        Parameters
        ----------
        user_id : str
            User identifier
        days : int
            Number of days to look back
            
        Returns
        -------
        dict
            Usage summary
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        usages = self.session.query(LLMUsage)\
            .filter(LLMUsage.user_id == user_id)\
            .filter(LLMUsage.created_at >= cutoff_date)\
            .all()
        
        total_requests = len(usages)
        total_tokens = sum(u.total_tokens for u in usages)
        total_cost = sum(u.cost for u in usages)
        
        # Group by model
        by_model = {}
        for usage in usages:
            if usage.model not in by_model:
                by_model[usage.model] = {
                    "requests": 0,
                    "tokens": 0,
                    "cost": 0.0
                }
            by_model[usage.model]["requests"] += 1
            by_model[usage.model]["tokens"] += usage.total_tokens
            by_model[usage.model]["cost"] += usage.cost
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 2),
            "by_model": by_model
        }
    
    def check_quota(
        self,
        user_id: str
    ) -> tuple[bool, Dict]:
        """
        Check if user has exceeded their quota.
        
        Parameters
        ----------
        user_id : str
            User identifier
            
        Returns
        -------
        tuple[bool, dict]
            (quota_exceeded, quota_info)
        """
        # Get user quota
        quota = self.session.query(UserQuota)\
            .filter(UserQuota.user_id == user_id)\
            .first()
        
        if not quota:
            # No quota set, allow unlimited
            return False, {
                "has_quota": False,
                "unlimited": True
            }
        
        # Get current month usage
        start_of_month = datetime.utcnow().replace(
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0
        )
        
        monthly_usage = self.session.query(LLMUsage)\
            .filter(LLMUsage.user_id == user_id)\
            .filter(LLMUsage.created_at >= start_of_month)\
            .all()
        
        total_cost = sum(u.cost for u in monthly_usage)
        total_requests = len(monthly_usage)
        
        # Check limits
        cost_exceeded = total_cost >= quota.monthly_budget_usd
        requests_exceeded = total_requests >= quota.monthly_request_limit
        
        quota_exceeded = cost_exceeded or requests_exceeded
        
        return quota_exceeded, {
            "has_quota": True,
            "monthly_budget_usd": quota.monthly_budget_usd,
            "current_spend_usd": round(total_cost, 2),
            "budget_remaining_usd": round(quota.monthly_budget_usd - total_cost, 2),
            "monthly_request_limit": quota.monthly_request_limit,
            "current_requests": total_requests,
            "requests_remaining": quota.monthly_request_limit - total_requests,
            "quota_exceeded": quota_exceeded,
            "cost_exceeded": cost_exceeded,
            "requests_exceeded": requests_exceeded
        }


# Global instance
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get global cost tracker instance."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker

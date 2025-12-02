# src/llm/chains/prediction_chain.py
"""
Prediction Explanation Chain
============================
Chain that combines ML prediction with natural language explanation.
"""

import logging
import time
from typing import Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from .base import BaseHealthcareChain, ChainResult, get_llm
from ..tools import get_prediction_tool, get_explanation_tool
from ..config import LLMConfig

logger = logging.getLogger(__name__)


# Prompt template for combining prediction with explanation
PREDICTION_EXPLANATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Healthcare Appointment Assistant helping staff understand no-show predictions.

Your task is to:
1. Analyze the ML prediction results
2. Explain the risk factors in plain language
3. Provide actionable recommendations

Be empathetic, practical, and focus on patient support.
"""),
    ("human", """Please analyze this prediction and provide a comprehensive explanation.

PREDICTION RESULTS:
{prediction_result}

PATIENT INFORMATION:
- Age: {age}
- Gender: {gender}
- Days until appointment: {lead_days}
- SMS reminder: {sms_status}
- Previous no-shows: {previous_noshows}

Please provide:
1. **Risk Summary** (2-3 sentences)
2. **Key Factors** (bullet points)
3. **Recommended Actions** (specific steps)
4. **Communication Tips** (how to approach the patient)
""")
])


class PredictionExplanationChain(BaseHealthcareChain):
    """
    Chain that gets a prediction and generates an explanation.
    
    This chain:
    1. Calls the ML prediction API via tool
    2. Generates a human-readable explanation
    3. Returns structured results
    
    Example
    -------
    >>> chain = PredictionExplanationChain()
    >>> result = chain.run(
    ...     age=35,
    ...     gender="F",
    ...     lead_days=14,
    ...     sms_received=0
    ... )
    >>> print(result.content)
    """
    
    def __init__(self, **kwargs):
        self.prediction_tool = get_prediction_tool()
        super().__init__(**kwargs)
    
    def _build_chain(self):
        """Build the prediction explanation chain."""
        return (
            PREDICTION_EXPLANATION_PROMPT
            | self._llm
            | StrOutputParser()
        )
    
    def run(
        self,
        age: int,
        gender: str,
        lead_days: int,
        sms_received: int = 0,
        previous_noshows: int = 0,
        **kwargs
    ) -> ChainResult:
        """
        Execute the prediction and explanation chain.
        
        Parameters
        ----------
        age : int
            Patient age
        gender : str
            Patient gender
        lead_days : int
            Days until appointment
        sms_received : int
            Whether SMS reminder was sent
        previous_noshows : int
            Number of previous no-shows
        **kwargs
            Additional patient data
        
        Returns
        -------
        ChainResult
            Result containing explanation
        """
        start_time = time.time()
        
        try:
            # Step 1: Get prediction from ML API
            prediction_result = self.prediction_tool.invoke({
                "age": age,
                "gender": gender,
                "lead_days": lead_days,
                "sms_received": sms_received,
                **kwargs
            })
            
            # Step 2: Generate explanation
            explanation = self._chain.invoke({
                "prediction_result": prediction_result,
                "age": age,
                "gender": gender,
                "lead_days": lead_days,
                "sms_status": "Yes" if sms_received else "No",
                "previous_noshows": previous_noshows if previous_noshows else "Unknown"
            })
            
            latency = (time.time() - start_time) * 1000
            
            return ChainResult(
                success=True,
                content=explanation,
                chain_name="PredictionExplanationChain",
                metadata={
                    "prediction_result": prediction_result,
                    "patient_age": age,
                    "lead_days": lead_days
                },
                latency_ms=latency
            )
            
        except Exception as e:
            return self._handle_error(e, "PredictionExplanationChain")
    
    async def arun(self, **kwargs) -> ChainResult:
        """Async execution."""
        start_time = time.time()
        
        try:
            prediction_result = await self.prediction_tool.ainvoke({
                "age": kwargs["age"],
                "gender": kwargs["gender"],
                "lead_days": kwargs["lead_days"],
                "sms_received": kwargs.get("sms_received", 0)
            })
            
            explanation = await self._chain.ainvoke({
                "prediction_result": prediction_result,
                "age": kwargs["age"],
                "gender": kwargs["gender"],
                "lead_days": kwargs["lead_days"],
                "sms_status": "Yes" if kwargs.get("sms_received") else "No",
                "previous_noshows": kwargs.get("previous_noshows", "Unknown")
            })
            
            latency = (time.time() - start_time) * 1000
            
            return ChainResult(
                success=True,
                content=explanation,
                chain_name="PredictionExplanationChain",
                metadata={"prediction_result": prediction_result},
                latency_ms=latency
            )
            
        except Exception as e:
            return self._handle_error(e, "PredictionExplanationChain")


# Convenience function
def explain_prediction(
    age: int,
    gender: str,
    lead_days: int,
    **kwargs
) -> str:
    """
    Quick function to get a prediction explanation.
    
    Returns the explanation content directly.
    """
    chain = PredictionExplanationChain()
    result = chain.run(age=age, gender=gender, lead_days=lead_days, **kwargs)
    return result.content if result.success else f"Error: {result.error}"
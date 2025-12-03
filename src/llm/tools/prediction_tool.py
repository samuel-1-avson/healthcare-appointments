# src/llm/tools/prediction_tool.py
"""
Prediction API Tool
===================
LangChain tool for calling the no-show prediction API.
"""

import logging
from typing import Dict, Any, Optional, Type
import httpx
import json

from langchain_core.tools import BaseTool, StructuredTool, tool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic.v1 import BaseModel, Field, PrivateAttr

from ..langchain_config import get_langchain_settings


logger = logging.getLogger(__name__)


# ==================== Input Schemas ====================

class PredictionInput(BaseModel):
    """Input schema for prediction tool."""
    
    age: int = Field(description="Patient age in years (0-120)")
    gender: str = Field(description="Patient gender: M, F, or O")
    lead_days: int = Field(description="Days until appointment (0-365)")
    sms_received: int = Field(default=0, description="SMS reminder sent: 0 or 1")
    scholarship: int = Field(default=0, description="In welfare program: 0 or 1")
    hypertension: int = Field(default=0, description="Has hypertension: 0 or 1")
    diabetes: int = Field(default=0, description="Has diabetes: 0 or 1")
    alcoholism: int = Field(default=0, description="Has alcoholism: 0 or 1")
    handicap: int = Field(default=0, description="Disability level: 0-4")
    neighbourhood: Optional[str] = Field(default=None, description="Patient neighbourhood")
    appointment_weekday: Optional[str] = Field(default=None, description="Day of week: Monday-Sunday")


class BatchPredictionInput(BaseModel):
    """Input schema for batch prediction."""
    
    appointments: list[Dict[str, Any]] = Field(
        description="List of appointment dictionaries with patient data"
    )


# ==================== Tool Implementation ====================

class PredictionTool(BaseTool):
    """
    Tool for making no-show predictions via the API.
    
    This tool allows the LLM to call your existing prediction API
    and get real predictions for patient appointments.
    
    Example
    -------
    >>> tool = PredictionTool()
    >>> result = tool.invoke({
    ...     "age": 35,
    ...     "gender": "F",
    ...     "lead_days": 7
    ... })
    """
    
    # Fix for Pydantic v1 "unable to infer type for attribute artifact"
    _artifact: Any = PrivateAttr(default=None)
    
    name: str = "predict_noshow"
    description: str = """
    Predict whether a patient will miss their healthcare appointment.
    
    Use this tool when you need to:
    - Get a no-show risk prediction for a patient
    - Assess appointment risk levels
    - Determine if intervention is needed
    
    Required inputs: age, gender, lead_days
    Optional inputs: sms_received, scholarship, hypertension, diabetes, 
                     alcoholism, handicap, neighbourhood, appointment_weekday
    
    Returns: prediction (0/1), probability (0-1), risk tier, recommended intervention
    """
    args_schema: Type[BaseModel] = PredictionInput
    
    # Configuration
    api_base_url: str = "http://localhost:8000"
    timeout: float = 30.0
    
    def __init__(self, api_base_url: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        settings = get_langchain_settings()
        self.api_base_url = api_base_url or settings.ml_api_base_url
    
    def _run(
        self,
        age: int,
        gender: str,
        lead_days: int,
        sms_received: int = 0,
        scholarship: int = 0,
        hypertension: int = 0,
        diabetes: int = 0,
        alcoholism: int = 0,
        handicap: int = 0,
        neighbourhood: Optional[str] = None,
        appointment_weekday: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the prediction."""
        
        # Build request payload
        payload = {
            "age": age,
            "gender": gender,
            "lead_days": lead_days,
            "sms_received": sms_received,
            "scholarship": scholarship,
            "hypertension": hypertension,
            "diabetes": diabetes,
            "alcoholism": alcoholism,
            "handicap": handicap
        }
        
        if neighbourhood:
            payload["neighbourhood"] = neighbourhood
        if appointment_weekday:
            payload["appointment_weekday"] = appointment_weekday
        
        try:
            # Call the prediction API
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.api_base_url}/api/v1/predict",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
            
            # Format response for LLM
            return self._format_response(result, payload)
        
        except httpx.HTTPStatusError as e:
            logger.error(f"API error: {e}")
            return f"Error calling prediction API: {e.response.status_code} - {e.response.text}"
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return f"Error: {str(e)}"
    
    async def _arun(
        self,
        age: int,
        gender: str,
        lead_days: int,
        **kwargs
    ) -> str:
        """Async execution."""
        payload = {
            "age": age,
            "gender": gender,
            "lead_days": lead_days,
            **{k: v for k, v in kwargs.items() if v is not None}
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.api_base_url}/api/v1/predict",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
        
        return self._format_response(result, payload)
    
    def _format_response(self, result: Dict, inputs: Dict) -> str:
        """Format API response for LLM consumption."""
        prediction = result.get("prediction", 0)
        probability = result.get("probability", 0)
        risk = result.get("risk", {})
        intervention = result.get("intervention", {})
        
        return f"""
PREDICTION RESULT:
- Prediction: {"NO-SHOW" if prediction == 1 else "WILL ATTEND"}
- Probability: {probability:.1%}
- Risk Tier: {risk.get("tier", "UNKNOWN")}
- Confidence: {risk.get("confidence", "Unknown")}

RISK ASSESSMENT:
{risk.get("emoji", "•")} {risk.get("tier", "UNKNOWN")} risk - {probability:.0%} chance of no-show

RECOMMENDED INTERVENTION:
- Action: {intervention.get("action", "Standard process")}
- SMS Reminders: {intervention.get("sms_reminders", 1)}
- Phone Call: {"Yes" if intervention.get("phone_call") else "No"}
- Priority: {intervention.get("priority", "normal")}

INPUT DATA:
- Age: {inputs.get("age")}
- Gender: {inputs.get("gender")}
- Lead Time: {inputs.get("lead_days")} days
- SMS Enabled: {"Yes" if inputs.get("sms_received") else "No"}
"""


# ==================== Factory Functions ====================

def create_prediction_tool(api_url: Optional[str] = None) -> PredictionTool:
    """
    Create a prediction tool instance.
    
    Parameters
    ----------
    api_url : str, optional
        Base URL for the prediction API
    
    Returns
    -------
    PredictionTool
        Configured prediction tool
    """
    return PredictionTool(api_base_url=api_url)


@tool
def predict_noshow_simple(
    age: int,
    gender: str,
    lead_days: int,
    sms_received: int = 0
) -> str:
    """
    Make a quick no-show prediction with minimal inputs.
    
    Args:
        age: Patient age in years
        gender: M or F
        lead_days: Days until appointment
        sms_received: 1 if SMS reminder sent, 0 otherwise
    
    Returns:
        Prediction result with risk tier and recommendation
    """
    tool = PredictionTool()
    return tool._run(
        age=age,
        gender=gender,
        lead_days=lead_days,
        sms_received=sms_received
    )


def create_batch_prediction_tool(api_url: Optional[str] = None) -> StructuredTool:
    """Create tool for batch predictions."""
    
    settings = get_langchain_settings()
    api_base = api_url or settings.ml_api_base_url
    
    def batch_predict(appointments: list[Dict[str, Any]]) -> str:
        """Make predictions for multiple appointments."""
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{api_base}/api/v1/predict/batch",
                json={"appointments": appointments}
            )
            response.raise_for_status()
            result = response.json()
        
        summary = result.get("summary", {})
        total = summary.get("total", len(appointments))
        predicted_noshows = summary.get("predicted_noshows", 0)
        avg_prob = summary.get("avg_probability", 0)
        risk_dist = json.dumps(summary.get("risk_distribution", {}), indent=2)
        proc_time = int(result.get("processing_time_ms", 0))

        return f"""
BATCH PREDICTION RESULTS:
- Total Appointments: {total}
- Predicted No-Shows: {predicted_noshows}
- Average Probability: {avg_prob:.1%}

RISK DISTRIBUTION:
{risk_dist}

Processing Time: {proc_time}ms
"""
    
    return StructuredTool.from_function(
        func=batch_predict,
        name="batch_predict_noshows",
        description="Predict no-shows for multiple appointments at once. "
                   "Input: list of appointment dictionaries with patient data.",
        args_schema=BatchPredictionInput
    )
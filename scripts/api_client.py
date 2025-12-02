#!/usr/bin/env python
"""
No-Show Prediction API Client
==============================
Python client for interacting with the No-Show Prediction API.

Usage:
    from scripts.api_client import NoShowAPIClient
    
    client = NoShowAPIClient("http://localhost:8000")
    result = client.predict(age=35, gender="F", lead_days=7)
    print(result)
"""

import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json


@dataclass
class PredictionResult:
    """Container for prediction result."""
    prediction: int
    probability: float
    risk_tier: str
    intervention: str
    raw_response: Dict[str, Any]
    
    @property
    def will_show(self) -> bool:
        """Check if patient is predicted to show up."""
        return self.prediction == 0
    
    @property
    def will_noshow(self) -> bool:
        """Check if patient is predicted to no-show."""
        return self.prediction == 1
    
    def __str__(self) -> str:
        status = "NO-SHOW" if self.will_noshow else "WILL ATTEND"
        return (
            f"Prediction: {status}\n"
            f"Probability: {self.probability:.1%}\n"
            f"Risk Tier: {self.risk_tier}\n"
            f"Intervention: {self.intervention}"
        )


class NoShowAPIClient:
    """
    Client for the No-Show Prediction API.
    
    Example
    -------
    >>> client = NoShowAPIClient()
    >>> 
    >>> # Single prediction
    >>> result = client.predict(age=35, gender="F", lead_days=7)
    >>> print(f"Risk: {result.risk_tier}, Probability: {result.probability:.1%}")
    >>> 
    >>> # Batch prediction
    >>> appointments = [
    ...     {"age": 25, "gender": "M", "lead_days": 3},
    ...     {"age": 45, "gender": "F", "lead_days": 14}
    ... ]
    >>> results = client.predict_batch(appointments)
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_prefix: str = "/api/v1",
        timeout: int = 30
    ):
        """
        Initialize the API client.
        
        Parameters
        ----------
        base_url : str
            API base URL
        api_prefix : str
            API path prefix
        timeout : int
            Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_prefix = api_prefix
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def _url(self, path: str) -> str:
        """Build full URL for an endpoint."""
        return f"{self.base_url}{self.api_prefix}{path}"
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle API response."""
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 422:
            error = response.json()
            raise ValueError(f"Validation error: {error}")
        elif response.status_code == 503:
            raise RuntimeError("Model not available")
        else:
            response.raise_for_status()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health.
        
        Returns
        -------
        dict
            Health status information
        """
        response = self.session.get(
            f"{self.base_url}/health",
            timeout=self.timeout
        )
        return self._handle_response(response)
    
    def is_healthy(self) -> bool:
        """
        Check if API is healthy.
        
        Returns
        -------
        bool
            True if API is healthy
        """
        try:
            health = self.health_check()
            return health.get("status") == "healthy"
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns
        -------
        dict
            Model information
        """
        response = self.session.get(
            self._url("/model"),
            timeout=self.timeout
        )
        return self._handle_response(response)
    
    def predict(
        self,
        age: int,
        gender: str,
        lead_days: int,
        scholarship: int = 0,
        hypertension: int = 0,
        diabetes: int = 0,
        alcoholism: int = 0,
        handicap: int = 0,
        sms_received: int = 0,
        threshold: Optional[float] = None,
        include_explanation: bool = False,
        **kwargs
    ) -> PredictionResult:
        """
        Make a single prediction.
        
        Parameters
        ----------
        age : int
            Patient age (0-120)
        gender : str
            Patient gender ('M', 'F', or 'O')
        lead_days : int
            Days between scheduling and appointment
        scholarship : int
            Welfare program enrollment (0/1)
        hypertension : int
            Has hypertension (0/1)
        diabetes : int
            Has diabetes (0/1)
        alcoholism : int
            Has alcoholism (0/1)
        handicap : int
            Disability level (0-4)
        sms_received : int
            SMS reminder sent (0/1)
        threshold : float, optional
            Custom classification threshold
        include_explanation : bool
            Include feature explanations
        **kwargs
            Additional optional features
        
        Returns
        -------
        PredictionResult
            Prediction result
        """
        # Build request data
        data = {
            "age": age,
            "gender": gender,
            "lead_days": lead_days,
            "scholarship": scholarship,
            "hypertension": hypertension,
            "diabetes": diabetes,
            "alcoholism": alcoholism,
            "handicap": handicap,
            "sms_received": sms_received,
            **kwargs
        }
        
        # Build query params
        params = {}
        if threshold is not None:
            params["threshold"] = threshold
        if include_explanation:
            params["include_explanation"] = "true"
        
        # Make request
        response = self.session.post(
            self._url("/predict"),
            json=data,
            params=params,
            timeout=self.timeout
        )
        
        result = self._handle_response(response)
        
        return PredictionResult(
            prediction=result["prediction"],
            probability=result["probability"],
            risk_tier=result["risk"]["tier"],
            intervention=result["intervention"]["action"],
            raw_response=result
        )
    
    def predict_batch(
        self,
        appointments: List[Dict[str, Any]],
        threshold: Optional[float] = None,
        include_explanations: bool = False
    ) -> Dict[str, Any]:
        """
        Make batch predictions.
        
        Parameters
        ----------
        appointments : list
            List of appointment dictionaries
        threshold : float, optional
            Custom classification threshold
        include_explanations : bool
            Include feature explanations
        
        Returns
        -------
        dict
            Batch prediction results
        """
        data = {
            "appointments": appointments,
            "include_explanations": include_explanations
        }
        if threshold is not None:
            data["threshold"] = threshold
        
        response = self.session.post(
            self._url("/predict/batch"),
            json=data,
            timeout=self.timeout * 2  # Longer timeout for batch
        )
        
        return self._handle_response(response)
    
    def predict_quick(
        self,
        age: int,
        gender: str,
        lead_days: int,
        sms_received: int = 1
    ) -> Dict[str, Any]:
        """
        Quick prediction with minimal parameters.
        
        Parameters
        ----------
        age : int
            Patient age
        gender : str
            Patient gender
        lead_days : int
            Days until appointment
        sms_received : int
            SMS reminder sent
        
        Returns
        -------
        dict
            Quick prediction result
        """
        params = {
            "age": age,
            "gender": gender,
            "lead_days": lead_days,
            "sms_received": sms_received
        }
        
        response = self.session.post(
            self._url("/predict/quick"),
            params=params,
            timeout=self.timeout
        )
        
        return self._handle_response(response)


# ==================== CLI Interface ====================

def main():
    """Command-line interface for API client."""
    import argparse
    
    parser = argparse.ArgumentParser(description="No-Show Prediction API Client")
    parser.add_argument("--url", default="http://localhost:8000", help="API URL")
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Health command
    subparsers.add_parser("health", help="Check API health")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make prediction")
    predict_parser.add_argument("--age", type=int, required=True)
    predict_parser.add_argument("--gender", type=str, required=True)
    predict_parser.add_argument("--lead-days", type=int, required=True)
    predict_parser.add_argument("--sms", type=int, default=1)
    
    # Model info command
    subparsers.add_parser("model", help="Get model info")
    
    args = parser.parse_args()
    
    client = NoShowAPIClient(args.url)
    
    if args.command == "health":
        result = client.health_check()
        print(json.dumps(result, indent=2))
    
    elif args.command == "predict":
        result = client.predict(
            age=args.age,
            gender=args.gender,
            lead_days=args.lead_days,
            sms_received=args.sms
        )
        print(result)
    
    elif args.command == "model":
        result = client.get_model_info()
        print(json.dumps(result, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
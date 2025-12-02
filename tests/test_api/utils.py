from typing import Dict, Any

def assert_valid_prediction_response(response_data: Dict[str, Any]):
    """Assert that a prediction response has the correct structure."""
    assert "prediction" in response_data
    assert "probability" in response_data
    assert "risk" in response_data
    assert "intervention" in response_data
    assert "model_version" in response_data
    
    # Check prediction is binary
    assert response_data["prediction"] in [0, 1]
    
    # Check probability is valid
    assert 0 <= response_data["probability"] <= 1
    
    # Check risk structure
    risk = response_data["risk"]
    assert "tier" in risk
    assert "probability" in risk
    assert "color" in risk
    assert risk["tier"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL"]
    
    # Check intervention structure
    intervention = response_data["intervention"]
    assert "action" in intervention
    assert "sms_reminders" in intervention
    assert "phone_call" in intervention


def assert_valid_health_response(response_data: Dict[str, Any]):
    """Assert that a health response has the correct structure."""
    assert "status" in response_data
    assert "timestamp" in response_data
    assert "version" in response_data
    assert "model_loaded" in response_data

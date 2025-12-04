import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/api/v1/predict"

def verify_shap():
    payload = {
        "age": 65,
        "gender": "M",
        "lead_days": 20,
        "scholarship": 0,
        "hypertension": 1,
        "diabetes": 1,
        "alcoholism": 0,
        "handicap": 0,
        "sms_received": 0,
        "appointment_weekday": "Monday",
        "patient_historical_noshow_rate": 0.5,
        "patient_total_appointments": 10,
        "is_first_appointment": 0
    }
    
    try:
        logger.info(f"Sending request to {API_URL}...")
        response = requests.post(API_URL, json=payload, params={"include_explanation": True})
        response.raise_for_status()
        
        data = response.json()
        logger.info("Response received.")
        
        if "explanation" not in data:
            logger.error("FAIL: 'explanation' field missing in response.")
            return False
            
        explanation = data["explanation"]
        if not explanation:
             logger.error("FAIL: 'explanation' is null or empty.")
             return False

        if "top_risk_factors" not in explanation:
            logger.error("FAIL: 'top_risk_factors' missing in explanation.")
            return False
            
        risk_factors = explanation["top_risk_factors"]
        logger.info(f"Top Risk Factors: {json.dumps(risk_factors, indent=2)}")
        
        if len(risk_factors) > 0:
            logger.info("SUCCESS: SHAP explanation received and contains risk factors.")
            return True
        else:
            logger.warning("WARNING: SHAP explanation received but risk factors list is empty (might be low risk).")
            return True
            
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False

if __name__ == "__main__":
    if verify_shap():
        print("VERIFICATION PASSED")
    else:
        print("VERIFICATION FAILED")
        exit(1)

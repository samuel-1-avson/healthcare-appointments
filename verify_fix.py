import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.getcwd())

from src.api.predict import NoShowPredictor
from src.api.schemas import AppointmentFeatures

# Setup logging
logging.basicConfig(level=logging.INFO)

def verify():
    print("Initializing predictor...")
    predictor = NoShowPredictor()
    predictor.load_model()
    
    print("Creating test appointment...")
    # Create a dummy appointment with some data
    appointment = AppointmentFeatures(
        patientid="12345",
        appointmentid="67890",
        gender="F",
        scheduledday=datetime.now(),
        appointmentday=datetime.now(),
        age=30,
        neighbourhood="JARDIM DA PENHA",
        scholarship=0,
        hypertension=0,
        diabetes=0,
        alcoholism=0,
        handicap=0,
        sms_received=0,
        no_show=0
    )
    
    print("Running prediction...")
    try:
        response = predictor.predict(appointment)
        print("\n✅ Prediction successful!")
        print(f"Probability: {response.probability}")
        print(f"Risk Tier: {response.risk.tier}")
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        raise

if __name__ == "__main__":
    verify()

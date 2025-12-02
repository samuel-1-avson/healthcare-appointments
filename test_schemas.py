import sys
import os
sys.path.insert(0, os.getcwd())

try:
    from src.api.schemas import AppointmentFeatures
    print("✅ Import successful")
    
    # Try to instantiate
    obj = AppointmentFeatures(
        age=30,
        gender="F",
        lead_days=10
    )
    print("✅ Instantiation successful")
    print(obj.model_dump())

except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

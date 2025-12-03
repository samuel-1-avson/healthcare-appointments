# tests/load/locustfile.py
"""
Locust Load Testing Script
==========================
Simulate user traffic to benchmark API performance.

Usage:
    locust -f tests/load/locustfile.py --host=http://localhost:8000
"""

import json
import random
from locust import HttpUser, task, between, events

# Test data
SAMPLE_PATIENT = {
    "PatientId": "123456",
    "AppointmentID": "987654",
    "Gender": "F",
    "ScheduledDay": "2024-01-20T10:00:00Z",
    "AppointmentDay": "2024-01-25T14:30:00Z",
    "Age": 35,
    "Neighbourhood": "JARDIM DA PENHA",
    "Scholarship": 0,
    "Hipertension": 0,
    "Diabetes": 0,
    "Alcoholism": 0,
    "Handcap": 0,
    "SMS_received": 1
}

class HealthcareUser(HttpUser):
    """Simulated API user."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    token = None
    
    def on_start(self):
        """Login on start."""
        try:
            # Login to get token
            response = self.client.post(
                "/api/v1/auth/token",
                data={"username": "testuser", "password": "testpassword"},
                name="/auth/token"
            )
            if response.status_code == 200:
                self.token = response.json()["access_token"]
                self.client.headers.update({"Authorization": f"Bearer {self.token}"})
            else:
                # If login fails, maybe user doesn't exist, try creating/using default
                # For load testing, we might assume auth is disabled or use a known user
                pass
        except Exception as e:
            print(f"Login failed: {e}")

    @task(3)
    def predict_noshow(self):
        """Make a prediction request."""
        # Randomize data slightly to avoid caching everything
        data = SAMPLE_PATIENT.copy()
        data["Age"] = random.randint(18, 90)
        data["SMS_received"] = random.choice([0, 1])
        
        self.client.post(
            "/api/v1/predict/",
            json=data,
            name="/predict"
        )

    @task(1)
    def check_health(self):
        """Check API health."""
        self.client.get("/api/v1/health", name="/health")

    @task(1)
    def get_model_info(self):
        """Get model information."""
        self.client.get("/api/v1/model/info", name="/model/info")
        
    @task(1)
    def submit_feedback(self):
        """Submit random feedback."""
        feedback = {
            "request_id": "load_test_req",
            "endpoint": "/predict",
            "rating": random.randint(3, 5),
            "comment": "Load test feedback"
        }
        self.client.post(
            "/api/v1/feedback/",
            json=feedback,
            name="/feedback"
        )

# Hook to print stats at end
@events.quitting.add_listener
def _(environment, **kw):
    if environment.stats.total.fail_ratio > 0.01:
        print(f"Test failed due to failure ratio > 1%: {environment.stats.total.fail_ratio}")
        environment.process_exit_code = 1
    else:
        environment.process_exit_code = 0

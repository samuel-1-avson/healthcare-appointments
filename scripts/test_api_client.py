#!/usr/bin/env python
"""
API Testing Script
==================
Interactive script for testing the No-Show Prediction API.

Usage:
    python scripts/test_api_client.py
    python scripts/test_api_client.py --url http://production-api:8000
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.api_client import NoShowAPIClient


def print_header(text: str):
    """Print section header."""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def test_health(client: NoShowAPIClient):
    """Test health endpoints."""
    print_header("Testing Health Endpoints")
    
    print("\n1. Health Check:")
    health = client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Model Loaded: {health['model_loaded']}")
    print(f"   Version: {health['version']}")
    
    print("\n2. Is Healthy:", client.is_healthy())


def test_model_info(client: NoShowAPIClient):
    """Test model info endpoint."""
    print_header("Testing Model Information")
    
    info = client.get_model_info()
    print(json.dumps(info, indent=2))


def test_single_prediction(client: NoShowAPIClient):
    """Test single prediction."""
    print_header("Testing Single Predictions")
    
    # Test case 1: Low risk
    print("\n1. Low Risk Patient:")
    result = client.predict(
        age=55,
        gender="F",
        lead_days=3,
        sms_received=1,
        hypertension=1  # Chronic condition = lower risk
    )
    print(result)
    
    # Test case 2: High risk
    print("\n2. High Risk Patient:")
    result = client.predict(
        age=22,
        gender="M",
        lead_days=25,
        sms_received=0
    )
    print(result)
    
    # Test case 3: With threshold
    print("\n3. Custom Threshold (0.3):")
    result = client.predict(
        age=35,
        gender="F",
        lead_days=10,
        threshold=0.3
    )
    print(result)


def test_batch_prediction(client: NoShowAPIClient):
    """Test batch prediction."""
    print_header("Testing Batch Predictions")
    
    appointments = [
        {"age": 20, "gender": "M", "lead_days": 1, "sms_received": 1},
        {"age": 35, "gender": "F", "lead_days": 7, "sms_received": 1},
        {"age": 45, "gender": "M", "lead_days": 14, "sms_received": 0},
        {"age": 60, "gender": "F", "lead_days": 21, "sms_received": 1},
        {"age": 75, "gender": "M", "lead_days": 30, "sms_received": 0},
    ]
    
    result = client.predict_batch(appointments)
    
    print(f"\nTotal Appointments: {result['summary']['total']}")
    print(f"Predicted No-Shows: {result['summary']['predicted_noshows']}")
    print(f"Predicted Shows: {result['summary']['predicted_shows']}")
    print(f"Average Probability: {result['summary']['avg_probability']:.1%}")
    print(f"Processing Time: {result['processing_time_ms']:.1f}ms")
    
    print("\nRisk Distribution:")
    for tier, count in result['summary']['risk_distribution'].items():
        print(f"  {tier}: {count}")


def test_quick_prediction(client: NoShowAPIClient):
    """Test quick prediction."""
    print_header("Testing Quick Prediction")
    
    result = client.predict_quick(
        age=35,
        gender="F",
        lead_days=7,
        sms_received=1
    )
    
    print(json.dumps(result, indent=2))


def test_performance(client: NoShowAPIClient, n_requests: int = 10):
    """Test API performance."""
    print_header(f"Performance Test ({n_requests} requests)")
    
    times = []
    for i in range(n_requests):
        start = time.time()
        client.predict(age=35, gender="F", lead_days=7)
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\nResults:")
    print(f"  Average: {avg_time*1000:.1f}ms")
    print(f"  Min: {min_time*1000:.1f}ms")
    print(f"  Max: {max_time*1000:.1f}ms")
    print(f"  Throughput: {1/avg_time:.1f} requests/second")


def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the No-Show Prediction API")
    parser.add_argument("--url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--skip-perf", action="store_true", help="Skip performance test")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NO-SHOW PREDICTION API - TEST SUITE")
    print("=" * 60)
    print(f"API URL: {args.url}")
    
    client = NoShowAPIClient(args.url)
    
    try:
        # Run tests
        test_health(client)
        test_model_info(client)
        test_single_prediction(client)
        test_batch_prediction(client)
        test_quick_prediction(client)
        
        if not args.skip_perf:
            test_performance(client)
        
        print_header("ALL TESTS COMPLETED SUCCESSFULLY! ✅")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
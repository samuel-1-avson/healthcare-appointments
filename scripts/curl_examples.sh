#!/bin/bash
# ============================================================
# No-Show Prediction API - cURL Examples
# ============================================================
# 
# Usage:
#   ./scripts/curl_examples.sh
#   API_URL=http://production:8000 ./scripts/curl_examples.sh
#
# ============================================================

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
API_PREFIX="/api/v1"

echo "=============================================="
echo "No-Show Prediction API - cURL Examples"
echo "=============================================="
echo "API URL: ${API_URL}"
echo ""

# ==================== Health Check ====================
echo "1. Health Check"
echo "---------------"
curl -s "${API_URL}/health" | python -m json.tool
echo ""

# ==================== Model Info ====================
echo "2. Model Information"
echo "--------------------"
curl -s "${API_URL}${API_PREFIX}/model" | python -m json.tool
echo ""

# ==================== Single Prediction ====================
echo "3. Single Prediction"
echo "--------------------"
curl -s -X POST "${API_URL}${API_PREFIX}/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "age": 35,
        "gender": "F",
        "scholarship": 0,
        "hypertension": 0,
        "diabetes": 0,
        "alcoholism": 0,
        "handicap": 0,
        "sms_received": 1,
        "lead_days": 7
    }' | python -m json.tool
echo ""

# ==================== Prediction with Options ====================
echo "4. Prediction with Custom Threshold"
echo "------------------------------------"
curl -s -X POST "${API_URL}${API_PREFIX}/predict?threshold=0.3&include_explanation=true" \
    -H "Content-Type: application/json" \
    -d '{
        "age": 25,
        "gender": "M",
        "lead_days": 14,
        "sms_received": 0
    }' | python -m json.tool
echo ""

# ==================== Quick Prediction ====================
echo "5. Quick Prediction (Query Params)"
echo "-----------------------------------"
curl -s -X POST "${API_URL}${API_PREFIX}/predict/quick?age=40&gender=F&lead_days=5&sms_received=1" \
    | python -m json.tool
echo ""

# ==================== Batch Prediction ====================
echo "6. Batch Prediction"
echo "-------------------"
curl -s -X POST "${API_URL}${API_PREFIX}/predict/batch" \
    -H "Content-Type: application/json" \
    -d '{
        "appointments": [
            {"age": 20, "gender": "M", "lead_days": 1, "sms_received": 1},
            {"age": 35, "gender": "F", "lead_days": 7, "sms_received": 1},
            {"age": 50, "gender": "M", "lead_days": 14, "sms_received": 0},
            {"age": 65, "gender": "F", "lead_days": 21, "sms_received": 1}
        ],
        "include_explanations": false
    }' | python -m json.tool
echo ""

# ==================== Threshold Info ====================
echo "7. Threshold Information"
echo "------------------------"
curl -s "${API_URL}${API_PREFIX}/predict/thresholds" | python -m json.tool
echo ""

# ==================== Feature Info ====================
echo "8. Feature Information"
echo "----------------------"
curl -s "${API_URL}${API_PREFIX}/model/features" | python -m json.tool
echo ""

echo "=============================================="
echo "All examples completed!"
echo "=============================================="
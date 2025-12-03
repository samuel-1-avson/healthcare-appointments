# tests/e2e/test_flows.py
"""
End-to-End (E2E) Tests
======================
Verify critical user flows using Playwright.

Requires:
    pip install pytest-playwright
    playwright install
"""

import pytest
from playwright.sync_api import Page, expect

# Configuration
BASE_URL = "http://localhost:5173"  # Frontend URL
API_URL = "http://localhost:8000"   # Backend URL

@pytest.mark.e2e
def test_prediction_flow(page: Page):
    """
    Test the full prediction flow:
    1. Load page
    2. Fill form
    3. Submit
    4. Verify result
    """
    # 1. Load Page
    page.goto(BASE_URL)
    expect(page).to_have_title("Healthcare No-Show Predictor")
    
    # 2. Fill Form
    # Assuming we have IDs or accessible names for these inputs
    # If not, we might need to update the frontend to add data-testid attributes
    
    # Select Gender
    page.get_by_label("Gender").select_option("F")
    
    # Fill Age
    page.get_by_label("Age").fill("35")
    
    # Select Neighbourhood
    page.get_by_label("Neighbourhood").select_option("JARDIM DA PENHA")
    
    # Check SMS Received
    page.get_by_label("SMS Received").check()
    
    # 3. Submit
    page.get_by_role("button", name="Predict Probability").click()
    
    # 4. Verify Result
    # Wait for result card to appear
    result_card = page.locator(".prediction-result")
    expect(result_card).to_be_visible(timeout=5000)
    
    # Check for probability text
    expect(page.get_by_text("Probability of No-Show")).to_be_visible()


@pytest.mark.e2e
def test_feedback_flow(page: Page):
    """
    Test the feedback submission flow.
    """
    # Pre-requisite: Make a prediction first to get the result view
    test_prediction_flow(page)
    
    # Click Feedback button (assuming it exists in the result card)
    page.get_by_role("button", name="Provide Feedback").click()
    
    # Fill Feedback Form
    page.get_by_label("Rating").fill("5")
    page.get_by_label("Comments").fill("Great prediction!")
    
    # Submit
    page.get_by_role("button", name="Submit Feedback").click()
    
    # Verify Success Message
    expect(page.get_by_text("Thank you for your feedback")).to_be_visible()


@pytest.mark.e2e
def test_api_health_direct(page: Page):
    """
    Direct API check from browser context.
    """
    response = page.request.get(f"{API_URL}/api/v1/health")
    expect(response).to_be_ok()
    data = response.json()
    assert data["status"] == "healthy"

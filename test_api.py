"""Automatic checks. Run them with:   pytest"""
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

GOOD_CUSTOMER = {
    "credit_limit": 200000,
    "education": "graduate_school",
    "status": [-1, -1, -1, -1, -1, -1],  # always paid in full
    "bills": [3500, 4200, 2800, 5100, 3900, 4400],
    "paid": [4200, 2800, 5100, 3900, 4400, 3000],
}
RISKY_CUSTOMER = {
    "credit_limit": 50000,
    "education": "university",
    "status": [2, 2, 2, 0, 0, 0],  # 2 months late, three months in a row
    "bills": [48900, 47800, 46200, 45100, 44000, 43500],
    "paid": [0, 0, 1500, 1600, 1700, 1800],
}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_demo_page_loads():
    assert client.get("/").status_code == 200


def test_prediction_has_the_right_shape():
    result = client.post("/predict", json=GOOD_CUSTOMER).json()
    assert 0 <= result["probability_of_default"] <= 1
    assert result["decision"] in ["approve", "decline"]
    assert len(result["top_reasons"]) == 3


def test_good_customer_is_approved():
    assert client.post("/predict", json=GOOD_CUSTOMER).json()["decision"] == "approve"


def test_risky_customer_is_declined_because_of_late_payments():
    result = client.post("/predict", json=RISKY_CUSTOMER).json()
    assert result["decision"] == "decline"
    assert result["top_reasons"][0]["reason"] == "Late payments"


def test_paying_late_increases_risk():
    late = {**GOOD_CUSTOMER, "status": [3, -1, -1, -1, -1, -1]}
    p_good = client.post("/predict", json=GOOD_CUSTOMER).json()["probability_of_default"]
    p_late = client.post("/predict", json=late).json()["probability_of_default"]
    assert p_late > p_good


def test_bad_input_is_rejected():
    assert client.post("/predict", json={**GOOD_CUSTOMER, "credit_limit": -5}).status_code == 422
    assert client.post("/predict", json={**GOOD_CUSTOMER, "status": [0, 0]}).status_code == 422
    assert client.post("/predict", json={**GOOD_CUSTOMER, "education": "phd"}).status_code == 422


def test_model_beats_baseline():
    results = client.get("/model-info").json()["test_results"]
    assert results["lightgbm"]["roc_auc"] > results["baseline_logistic_regression"]["roc_auc"]

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi.testclient import TestClient
import pytest

from backend.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "title" in data
    assert "units" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models_ready" in data


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "lstm_architecture" in data
    assert "autoencoder_architecture" in data
    assert "dataset_features" in data


def test_predict_validation_error():
    # Negative income should fail Pydantic validation (ge=0)
    invalid_payload = {
        "gender": "Male",
        "married": "Yes",
        "applicant_income": -500.0,
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_term_exceeds_5_years():
    # Term > 60 months (5 years) should fail Pydantic validation (le=60)
    invalid_payload = {
        "gender": "Male",
        "married": "Yes",
        "dependents": "0",
        "education": "Graduate",
        "self_employed": "No",
        "applicant_income": 5000,
        "coapplicant_income": 1500,
        "loan_amount": 128,
        "loan_amount_term": 72.0,  # 6 years > 5 years limit
        "credit_history": 1.0,
        "property_area": "Urban",
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_endpoint_structure():
    valid_payload = {
        "gender": "Male",
        "married": "Yes",
        "dependents": "0",
        "education": "Graduate",
        "self_employed": "No",
        "applicant_income": 5000,
        "coapplicant_income": 1500,
        "loan_amount": 128,
        "loan_amount_term": 60,
        "credit_history": 1.0,
        "property_area": "Urban",
    }
    response = client.post("/predict", json=valid_payload)
    # Either 200 (if models are trained) or 503 (if models not ready)
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "label" in data
        assert "approval_probability" in data
        assert "reconstruction_error" in data
        assert "anomaly_flag" in data

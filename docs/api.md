# REST API Specification

FastAPI backend endpoints for the Loan Approval Prediction system.

Base URL: `http://localhost:8000`

---

## Endpoints Summary

| Method | Endpoint | Description | Status Code |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | API Root Metadata | 200 OK |
| `GET` | `/health` | Server & Model Readiness Check | 200 OK |
| `GET` | `/model-info` | Architectures & Threshold Details | 200 OK |
| `POST` | `/predict` | Predict Loan Approval & Anomaly Flag | 200 OK / 503 / 422 |
| `POST` | `/reconstruct` | Autoencoder Reconstruction Check | 200 OK / 503 / 422 |

---

## 1. POST /predict

Evaluates incoming loan applicant details using Unit IV (LSTM) and Unit V (Autoencoder).

### Request Payload (`application/json`)
```json
{
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
  "property_area": "Urban"
}
```

### Response Payload (`200 OK`)
```json
{
  "prediction": 1,
  "label": "Approved",
  "approval_probability": 0.8245,
  "reconstruction_error": 0.1832,
  "anomaly_threshold": 0.4512,
  "anomaly_flag": false,
  "decision_threshold": 0.5,
  "disclaimer": "This is an academic prediction prototype and should not be used as a real-world lending decision."
}
```

### Example cURL Command
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "gender": "Male",
  "married": "Yes",
  "dependents": "0",
  "education": "Graduate",
  "self_employed": "No",
  "applicant_income": 5000,
  "coapplicant_income": 1500,
  "loan_amount": 128,
  "loan_amount_term": 60,
  "credit_history": 1,
  "property_area": "Urban"
}'
```

---

## 2. GET /health

Checks server health and verifies that model artifacts exist in `backend/models/`.

### Response Payload (`200 OK`)
```json
{
  "status": "healthy",
  "models_ready": true,
  "message": "All model files exist and engine is ready for prediction."
}
```

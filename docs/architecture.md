# System Architecture Documentation

This document describes the full-stack system architecture, data flow, and component interactions for the **Loan Approval Prediction Using Deep Learning** project.

---

## 1. High-Level Architecture Diagram

```mermaid
flowchart TD
    subgraph Frontend["Next.js Frontend (React + Tailwind CSS)"]
        UI["Application Form / Dashboard"]
        ClientAPI["API Client (fetch / NEXT_PUBLIC_API_URL)"]
        ResultUI["Result Card (Probability & Anomaly Badge)"]
        UI --> ClientAPI
        ClientAPI --> ResultUI
    end

    subgraph Backend["FastAPI Backend Server (Python 3.12)"]
        Router["FastAPI Router (/predict, /health, /model-info)"]
        Pydantic["Pydantic Input Validation (LoanApplicationRequest)"]
        InferenceEngine["Inference Engine (LoanInferenceEngine)"]
        Router --> Pydantic
        Pydantic --> InferenceEngine
    end

    subgraph MLEngine["Machine Learning & Pipeline Artifacts"]
        Preprocessor["Joblib Preprocessor (ColumnTransformer)"]
        Autoencoder["Unit V Autoencoder (autoencoder.keras)"]
        LSTM["Unit IV LSTM Classifier (lstm_model.keras)"]
        Threshold["Anomaly Threshold (threshold.json)"]

        InferenceEngine --> Preprocessor
        Preprocessor --> Autoencoder
        Preprocessor --> LSTM
        Autoencoder --> Threshold
    end

    ClientAPI <-->|JSON / REST HTTP| Router
```

---

## 2. End-to-End Prediction Sequence

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Frontend as Next.js UI
    participant FastAPI as FastAPI Server
    participant Preproc as Scikit-Learn Pipeline
    participant AE as Unit V Autoencoder
    participant LSTM as Unit IV LSTM

    User->>Frontend: Fills & Submits Loan Form
    Frontend->>FastAPI: POST /predict (JSON Payload)
    FastAPI->>FastAPI: Validate Schema (Pydantic)
    FastAPI->>Preproc: Transform Raw Dict to Scaled Array
    Preproc-->>FastAPI: Processed Vector X ∈ ℝ^D
    FastAPI->>AE: Predict Reconstruction X̂
    AE-->>FastAPI: Compute MSE Reconstruction Error & Anomaly Flag
    FastAPI->>LSTM: Reshape X to (1, D, 1) & Predict Probability
    LSTM-->>FastAPI: Return Approval Probability p ∈ [0, 1]
    FastAPI-->>Frontend: Return JSON (prediction, probability, MSE, anomaly_flag)
    Frontend-->>User: Display Approval Result & Anomaly Badge
```

---

## 3. Directory Structure Summary

```
loan-approval-project/
├── backend/
│   ├── main.py               # FastAPI application & CORS
│   ├── schemas.py            # Pydantic validation models
│   ├── inference.py          # Inference wrapper engine
│   ├── preprocessing.py      # Feature engineering & ColumnTransformer
│   ├── train.py              # Main training orchestration script
│   ├── train_autoencoder.py  # Unit V Autoencoder trainer
│   ├── train_lstm.py         # Unit IV LSTM trainer
│   ├── evaluate.py           # Model evaluation & plot generator
│   ├── models/               # Saved model artifacts (*.keras, *.joblib, *.json)
│   └── tests/                # Pytest test suite
├── data/
│   └── loan_data.csv         # Kaggle Loan Prediction dataset
├── docs/
│   ├── architecture.md       # Architecture specification (this file)
│   ├── model.md              # Machine learning model documentation
│   ├── api.md                # REST API documentation
│   └── results/              # Evaluation plots & metrics
├── frontend/
│   ├── app/                  # Next.js App Router (pages & layout)
│   ├── components/           # React UI components (Form, ResultCard, etc.)
│   └── package.json
└── README.md
```

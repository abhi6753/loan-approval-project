import os
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from backend.inference import LoanInferenceEngine, ModelNotReadyException
from backend.schemas import LoanApplicationRequest, ModelInfoResponse, PredictionResponse

app = FastAPI(
    title="Loan Approval Prediction API",
    description="Academic Deep Learning Project incorporating Unit IV (LSTM) and Unit V (Autoencoder).",
    version="1.0.0",
)

# CORS configuration allowing any local Next.js dev server port (3000, 3001, 3002, 3003, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = LoanInferenceEngine()


@app.get("/")
def read_root():
    return {
        "title": "Loan Approval Prediction Using Deep Learning",
        "units": {
            "Unit_IV": "LSTM Sequential Binary Classifier",
            "Unit_V": "Autoencoder Feature Reconstruction & Anomaly Detection",
        },
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health")
def health_check():
    models_ready = engine.check_artifacts_exist()
    return {
        "status": "healthy" if models_ready else "degraded",
        "models_ready": models_ready,
        "message": (
            "All model files exist and engine is ready for prediction."
            if models_ready
            else "Model files not trained yet. Run backend/train.py to generate models."
        ),
    }


@app.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    models_ready = engine.check_artifacts_exist()
    thresh = None
    if models_ready:
        try:
            if not engine.is_loaded:
                engine.load_models()
            thresh = engine.anomaly_threshold
        except Exception:
            pass

    return ModelInfoResponse(
        status="Ready" if models_ready else "Not Trained",
        lstm_architecture="Input(D, 1) -> LSTM(32) -> Dropout(0.2) -> Dense(16, relu) -> Dense(1, sigmoid)",
        autoencoder_architecture="Input(D) -> Dense(32, relu) -> Dense(16, relu) -> Latent(8, relu) -> Dense(16, relu) -> Dense(32, relu) -> Output(D)",
        anomaly_threshold=thresh,
        dataset_features=[
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
            "Property_Area",
        ],
        target="Loan_Status (1: Approved, 0: Rejected)",
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_loan(application: LoanApplicationRequest):
    try:
        raw_dict = application.model_dump()
        result = engine.predict(raw_dict)
        return PredictionResponse(**result)
    except ModelNotReadyException as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(err),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction error: {str(err)}",
        )


@app.post("/reconstruct")
def reconstruct_loan(application: LoanApplicationRequest):
    try:
        raw_dict = application.model_dump()
        result = engine.predict(raw_dict)
        return {
            "reconstruction_error": result["reconstruction_error"],
            "anomaly_threshold": result["anomaly_threshold"],
            "anomaly_flag": result["anomaly_flag"],
            "status": "Anomaly Flagged" if result["anomaly_flag"] else "Normal Distribution",
        }
    except ModelNotReadyException as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(err),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Reconstruction error: {str(err)}",
        )

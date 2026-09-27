from typing import Optional
from pydantic import BaseModel, Field


class LoanApplicationRequest(BaseModel):
    gender: str = Field(default="Male", description="Gender ('Male', 'Female')")
    married: str = Field(default="Yes", description="Marital status ('Yes', 'No')")
    dependents: str = Field(default="0", description="Number of dependents ('0', '1', '2', '3+')")
    education: str = Field(default="Graduate", description="Education level ('Graduate', 'Not Graduate')")
    self_employed: str = Field(default="No", description="Self employment status ('Yes', 'No')")
    applicant_income: float = Field(default=5000.0, ge=0, description="Applicant monthly income")
    coapplicant_income: float = Field(default=1500.0, ge=0, description="Co-applicant monthly income")
    loan_amount: float = Field(default=128.0, ge=0, description="Loan amount in thousands")
    loan_amount_term: float = Field(
        default=60.0,
        ge=0,
        le=60,
        description="Loan term in months (must not exceed 60 months / 5 years)",
    )
    credit_history: float = Field(default=1.0, ge=0, le=1, description="Credit history indicator (1.0 or 0.0)")
    property_area: str = Field(default="Urban", description="Property location ('Urban', 'Rural', 'Semiurban')")

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Binary prediction: 1 (Approved), 0 (Rejected)")
    label: str = Field(..., description="Human-readable decision ('Approved' or 'Rejected')")
    approval_probability: float = Field(..., description="Predicted probability of loan approval (0.0 to 1.0)")
    reconstruction_error: float = Field(..., description="Autoencoder MSE reconstruction error")
    anomaly_threshold: float = Field(..., description="Calculated training reconstruction error threshold")
    anomaly_flag: bool = Field(..., description="True if reconstruction error exceeds threshold")
    decision_threshold: float = Field(default=0.5, description="Probability cutoff threshold")
    disclaimer: str = Field(
        default="This is an academic prediction prototype and should not be used as a real-world lending decision."
    )


class ModelInfoResponse(BaseModel):
    status: str
    lstm_architecture: str
    autoencoder_architecture: str
    anomaly_threshold: Optional[float]
    dataset_features: list[str]
    target: str

import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from backend.preprocessing import preprocess_for_prediction
from backend.train_autoencoder import calculate_reconstruction_error
from backend.train_lstm import reshape_for_lstm

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


class ModelNotReadyException(Exception):
    """Raised when model weights or preprocessing pipeline files are missing."""
    pass


class LoanInferenceEngine:
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self.model_dir = model_dir
        self.preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
        self.lstm_path = os.path.join(model_dir, "lstm_model.keras")
        self.autoencoder_path = os.path.join(model_dir, "autoencoder.keras")
        self.threshold_path = os.path.join(model_dir, "threshold.json")

        self.preprocessor = None
        self.lstm_model = None
        self.autoencoder_model = None
        self.anomaly_threshold = None
        self.is_loaded = False

    def check_artifacts_exist(self) -> bool:
        return (
            os.path.exists(self.preprocessor_path)
            and os.path.exists(self.lstm_path)
            and os.path.exists(self.autoencoder_path)
            and os.path.exists(self.threshold_path)
        )

    def load_models(self):
        if not self.check_artifacts_exist():
            raise ModelNotReadyException(
                "Model files are missing. Please execute 'python backend/train.py' first."
            )

        self.lstm_model = tf.keras.models.load_model(self.lstm_path)
        self.autoencoder_model = tf.keras.models.load_model(self.autoencoder_path)

        with open(self.threshold_path, "r") as f:
            threshold_data = json.load(f)
        self.anomaly_threshold = float(threshold_data["threshold"])
        self.is_loaded = True

    def predict(self, raw_input_dict: dict) -> dict:
        """
        Executes end-to-end prediction:
        Raw Dict -> DataFrame -> Preprocessing -> Autoencoder MSE -> LSTM Probability -> Response
        """
        if not self.is_loaded:
            self.load_models()

        # Map API keys (snake_case) to exact Dataset column names
        column_mapping = {
            "gender": "Gender",
            "married": "Married",
            "dependents": "Dependents",
            "education": "Education",
            "self_employed": "Self_Employed",
            "applicant_income": "ApplicantIncome",
            "coapplicant_income": "CoapplicantIncome",
            "loan_amount": "LoanAmount",
            "loan_amount_term": "Loan_Amount_Term",
            "credit_history": "Credit_History",
            "property_area": "Property_Area",
        }

        row_data = {}
        for key, val in raw_input_dict.items():
            mapped_col = column_mapping.get(key, key)
            row_data[mapped_col] = [val]

        raw_df = pd.DataFrame(row_data)

        # 1. Transform raw inputs
        processed_array = preprocess_for_prediction(
            raw_df, preprocessor_path=self.preprocessor_path
        )

        # 2. Unit V Autoencoder reconstruction error
        ae_errors = calculate_reconstruction_error(self.autoencoder_model, processed_array)
        reconstruction_error = float(ae_errors[0])
        anomaly_flag = bool(reconstruction_error > self.anomaly_threshold)

        # 3. Unit IV LSTM probability
        processed_3d = reshape_for_lstm(processed_array)
        prob_array = self.lstm_model.predict(processed_3d, verbose=0)
        probability = float(prob_array[0][0])
        prediction = 1 if probability >= 0.5 else 0
        label = "Approved" if prediction == 1 else "Rejected"

        return {
            "prediction": prediction,
            "label": label,
            "approval_probability": round(probability, 4),
            "reconstruction_error": round(reconstruction_error, 4),
            "anomaly_threshold": round(self.anomaly_threshold, 4),
            "anomaly_flag": anomaly_flag,
            "decision_threshold": 0.5,
            "disclaimer": (
                "This is an academic prediction prototype and should not be used as a real-world lending decision."
            ),
        }

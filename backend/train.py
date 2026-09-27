import os
import sys

# Ensure parent directory is in path when executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.preprocessing import prepare_training_data
from backend.train_autoencoder import train_autoencoder_model
from backend.train_lstm import train_lstm_model


def run_full_training_pipeline(csv_path: str = "data/loan_data.csv"):
    print("==================================================")
    print("STARTING LOAN APPROVAL MODEL TRAINING PIPELINE")
    print("==================================================")

    # Step 1: Preprocessing & Data Preparation
    print("\n[Step 1/3] Loading dataset and fitting preprocessing pipeline...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_training_data(
        csv_path=csv_path
    )
    print(f"Dataset split complete: Train shape = {X_train.shape}, Test shape = {X_test.shape}")
    print(f"Target distribution - Train 1s: {sum(y_train)}, 0s: {len(y_train)-sum(y_train)}")

    # Step 2: Train Unit V Autoencoder
    print("\n[Step 2/3] Training Unit V Autoencoder & Computing Anomaly Threshold...")
    autoencoder, encoder, threshold_data, ae_history = train_autoencoder_model(
        X_train=X_train, X_val=X_test, epochs=80, batch_size=16
    )
    print(f"Autoencoder training complete. Anomaly threshold = {threshold_data['threshold']:.4f}")

    # Step 3: Train Unit IV LSTM Classifier
    print("\n[Step 3/3] Training Unit IV LSTM Binary Classifier...")
    lstm_model, lstm_history = train_lstm_model(
        X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, epochs=100, batch_size=16
    )
    print("LSTM training complete.")

    print("\n==================================================")
    print("ALL MODELS AND PREPROCESSING PIPELINES TRAINED AND SAVED SUCCESSFULLY!")
    print("Saved files in backend/models/:")
    print(" - preprocessor.joblib")
    print(" - autoencoder.keras")
    print(" - encoder.keras")
    print(" - lstm_model.keras")
    print(" - threshold.json")
    print("==================================================")
    return preprocessor, autoencoder, lstm_model, threshold_data


if __name__ == "__main__":
    run_full_training_pipeline()

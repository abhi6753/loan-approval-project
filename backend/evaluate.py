import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.preprocessing import prepare_training_data
from backend.train_autoencoder import calculate_reconstruction_error
from backend.train_lstm import reshape_for_lstm

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "results"))


def run_evaluation(
    csv_path: str = "data/loan_data.csv",
    model_dir: str = DEFAULT_MODEL_DIR,
    results_dir: str = RESULTS_DIR,
):
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load test data and preprocessor
    X_train, X_test, y_train, y_test, preprocessor = prepare_training_data(
        csv_path=csv_path, save_path=os.path.join(model_dir, "preprocessor.joblib")
    )

    # 2. Load models and threshold
    lstm_path = os.path.join(model_dir, "lstm_model.keras")
    ae_path = os.path.join(model_dir, "autoencoder.keras")
    thresh_path = os.path.join(model_dir, "threshold.json")

    if not (os.path.exists(lstm_path) and os.path.exists(ae_path) and os.path.exists(thresh_path)):
        raise FileNotFoundError(
            "Trained model files missing. Run backend/train.py first before evaluation."
        )

    lstm_model = tf.keras.models.load_model(lstm_path)
    autoencoder = tf.keras.models.load_model(ae_path)

    with open(thresh_path, "r") as f:
        threshold_info = json.load(f)
    anomaly_threshold = threshold_info["threshold"]

    # 3. LSTM Predictions
    X_test_3d = reshape_for_lstm(X_test)
    y_probs = lstm_model.predict(X_test_3d, verbose=0).flatten()
    y_preds = (y_probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, y_preds))
    prec = float(precision_score(y_test, y_preds, zero_division=0))
    rec = float(recall_score(y_test, y_preds, zero_division=0))
    f1 = float(f1_score(y_test, y_preds, zero_division=0))
    cm = confusion_matrix(y_test, y_preds)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "total_test_samples": len(y_test),
    }

    # 4. Autoencoder Reconstruction Errors
    ae_errors = calculate_reconstruction_error(autoencoder, X_test)
    anomalies = (ae_errors > anomaly_threshold).astype(int)
    anomaly_count = int(np.sum(anomalies))

    metrics["autoencoder_anomaly_count"] = anomaly_count
    metrics["autoencoder_anomaly_percentage"] = float((anomaly_count / len(y_test)) * 100)
    metrics["anomaly_threshold"] = float(anomaly_threshold)

    print("\n================ EVALUATION METRICS ================")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall   : {rec * 100:.2f}%")
    print(f"F1-Score : {f1 * 100:.2f}%")
    print(f"Autoencoder Anomaly Flagged: {anomaly_count}/{len(y_test)} ({metrics['autoencoder_anomaly_percentage']:.1f}%)")
    print("Confusion Matrix:\n", cm)
    print("====================================================")

    # Save metrics JSON
    metrics_path = os.path.join(results_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot 1: Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Rejected (0)", "Approved (1)"],
        yticklabels=["Rejected (0)", "Approved (1)"],
    )
    plt.title("Unit IV LSTM Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # Plot 2: Prediction Probability Distribution
    plt.figure(figsize=(7, 4))
    plt.hist(y_probs[y_test == 1], bins=15, alpha=0.6, label="Actual Approved (1)", color="green")
    plt.hist(y_probs[y_test == 0], bins=15, alpha=0.6, label="Actual Rejected (0)", color="red")
    plt.axvline(0.5, color="black", linestyle="--", label="Decision Threshold (0.5)")
    plt.title("Unit IV LSTM Approval Probability Distribution")
    plt.xlabel("Predicted Approval Probability")
    plt.ylabel("Sample Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "lstm_accuracy.png"), dpi=300)
    plt.close()

    # Plot 3: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Unit IV LSTM Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "lstm_loss.png"), dpi=300)
    plt.close()

    # Plot 4: Autoencoder Reconstruction Error Distribution & Threshold
    plt.figure(figsize=(7, 4))
    plt.hist(ae_errors, bins=20, color="purple", alpha=0.7, edgecolor="black")
    plt.axvline(
        anomaly_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Anomaly Threshold ({anomaly_threshold:.3f})",
    )
    plt.title("Unit V Autoencoder Reconstruction Error Distribution")
    plt.xlabel("Mean Squared Error (MSE)")
    plt.ylabel("Applicant Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "reconstruction_error.png"), dpi=300)
    plt.close()

    # Plot 5: Summary Bar Chart of Key Metrics
    plt.figure(figsize=(6, 4))
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    metric_vals = [acc, prec, rec, f1]
    bars = plt.bar(metric_names, metric_vals, color=["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"])
    plt.ylim(0, 1.1)
    plt.title("Model Performance Metrics Summary")
    plt.ylabel("Score")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "autoencoder_loss.png"), dpi=300)
    plt.close()

    print(f"Evaluation plots saved successfully in: {results_dir}")
    return metrics


if __name__ == "__main__":
    run_evaluation()

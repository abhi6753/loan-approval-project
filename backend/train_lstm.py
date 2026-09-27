import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
LSTM_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "lstm_model.keras")


def reshape_for_lstm(X: np.ndarray) -> np.ndarray:
    """
    Reshapes tabular 2D array (samples, features) into 3D sequence array (samples, timesteps=features, 1)
    for sequence-based academic LSTM processing.
    """
    if X.ndim == 2:
        return np.expand_dims(X, axis=-1)
    return X


def build_lstm_classifier(input_shape: tuple):
    """
    Builds Unit IV LSTM binary classification model.
    Input: (timesteps, 1)
    LSTM(32) -> Dropout(0.2) -> Dense(16, relu) -> Dense(1, sigmoid)
    """
    model = Sequential(
        [
            layers.Input(shape=input_shape, name="lstm_input"),
            layers.LSTM(32, return_sequences=False, name="lstm_layer"),
            layers.Dropout(0.2, name="dropout_layer"),
            layers.Dense(16, activation="relu", name="dense_hidden"),
            layers.Dense(1, activation="sigmoid", name="prediction_output"),
        ],
        name="lstm_classifier",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    epochs: int = 100,
    batch_size: int = 16,
    model_dir: str = DEFAULT_MODEL_DIR,
):
    """
    Reshapes X_train into 3D sequence format, builds & trains LSTM classifier with EarlyStopping,
    and saves model to lstm_model.keras.
    """
    os.makedirs(model_dir, exist_ok=True)

    X_train_3d = reshape_for_lstm(X_train)
    input_shape = (X_train_3d.shape[1], X_train_3d.shape[2])

    model = build_lstm_classifier(input_shape)

    if X_val is not None and y_val is not None:
        validation_data = (reshape_for_lstm(X_val), y_val)
        validation_split = 0.0
    else:
        validation_data = None
        validation_split = 0.15

    callbacks = [
        EarlyStopping(
            monitor="val_loss" if (validation_data or validation_split > 0) else "loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    history = model.fit(
        X_train_3d,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
    )

    save_path = os.path.join(model_dir, "lstm_model.keras")
    model.save(save_path)
    print(f"LSTM classifier saved to {save_path}")

    return model, history


def evaluate_lstm_predictions(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5):
    """
    Calculates empirical evaluation metrics for LSTM model on test set.
    """
    X_test_3d = reshape_for_lstm(X_test)
    probabilities = model.predict(X_test_3d, verbose=0).flatten()
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    return metrics, probabilities, predictions

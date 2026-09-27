import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
AUTOENCODER_PATH = os.path.join(DEFAULT_MODEL_DIR, "autoencoder.keras")
ENCODER_PATH = os.path.join(DEFAULT_MODEL_DIR, "encoder.keras")
THRESHOLD_PATH = os.path.join(DEFAULT_MODEL_DIR, "threshold.json")


def build_autoencoder(input_dim: int):
    """
    Builds Encoder, Decoder, and Autoencoder Keras models for tabular feature reconstruction.
    Encoder: input_dim -> 32 -> 16 -> 8 (latent)
    Decoder: 8 -> 16 -> 32 -> input_dim (reconstructed output)
    """
    inputs = layers.Input(shape=(input_dim,), name="autoencoder_input")

    # Encoder
    x = layers.Dense(32, activation="relu", name="enc_dense1")(inputs)
    x = layers.Dense(16, activation="relu", name="enc_dense2")(x)
    latent = layers.Dense(8, activation="relu", name="latent_representation")(x)

    encoder = Model(inputs=inputs, outputs=latent, name="encoder_model")

    # Decoder
    latent_inputs = layers.Input(shape=(8,), name="latent_input")
    d = layers.Dense(16, activation="relu", name="dec_dense1")(latent_inputs)
    d = layers.Dense(32, activation="relu", name="dec_dense2")(d)
    reconstruction_out = layers.Dense(
        input_dim, activation="linear", name="reconstruction_output"
    )(d)

    decoder = Model(inputs=latent_inputs, outputs=reconstruction_out, name="decoder_model")

    # Full Autoencoder
    reconstructed = decoder(encoder(inputs))
    autoencoder = Model(inputs=inputs, outputs=reconstructed, name="autoencoder_model")

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse"
    )
    return autoencoder, encoder, decoder


def calculate_reconstruction_error(
    model: tf.keras.Model, X: np.ndarray
) -> np.ndarray:
    """
    Calculates Mean Squared Error (MSE) per sample between original inputs
    and reconstructed outputs.
    """
    reconstructions = model.predict(X, verbose=0)
    errors = np.mean(np.square(X - reconstructions), axis=1)
    return errors


def compute_anomaly_threshold(
    train_errors: np.ndarray, std_multiplier: float = 2.0
) -> float:
    """
    Computes anomaly threshold on training reconstruction errors only:
    threshold = mean(train_errors) + std_multiplier * std(train_errors)
    """
    mean_err = float(np.mean(train_errors))
    std_err = float(np.std(train_errors))
    threshold = mean_err + (std_multiplier * std_err)
    return float(threshold)


def train_autoencoder_model(
    X_train: np.ndarray,
    X_val: np.ndarray = None,
    epochs: int = 100,
    batch_size: int = 16,
    model_dir: str = DEFAULT_MODEL_DIR,
):
    """
    Trains autoencoder on X_train (unsupervised), computes anomaly threshold on X_train errors,
    and saves autoencoder.keras, encoder.keras, and threshold.json.
    """
    os.makedirs(model_dir, exist_ok=True)
    input_dim = X_train.shape[1]

    autoencoder, encoder, decoder = build_autoencoder(input_dim)

    validation_data = (X_val, X_val) if X_val is not None else None
    validation_split = 0.15 if X_val is None else 0.0

    callbacks = [
        EarlyStopping(
            monitor="val_loss" if (validation_data or validation_split > 0) else "loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    history = autoencoder.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
    )

    # Calculate threshold strictly on training set reconstruction errors
    train_errors = calculate_reconstruction_error(autoencoder, X_train)
    threshold = compute_anomaly_threshold(train_errors)

    # Save artifacts
    autoencoder_save_path = os.path.join(model_dir, "autoencoder.keras")
    encoder_save_path = os.path.join(model_dir, "encoder.keras")
    threshold_save_path = os.path.join(model_dir, "threshold.json")

    autoencoder.save(autoencoder_save_path)
    encoder.save(encoder_save_path)

    threshold_data = {
        "threshold": threshold,
        "train_mean_error": float(np.mean(train_errors)),
        "train_std_error": float(np.std(train_errors)),
        "train_max_error": float(np.max(train_errors)),
    }
    with open(threshold_save_path, "w") as f:
        json.dump(threshold_data, f, indent=2)

    print(f"Autoencoder saved to {autoencoder_save_path}")
    print(f"Encoder saved to {encoder_save_path}")
    print(f"Threshold saved ({threshold:.4f}) to {threshold_save_path}")

    return autoencoder, encoder, threshold_data, history

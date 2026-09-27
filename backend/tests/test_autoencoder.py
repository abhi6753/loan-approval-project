import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pytest
import tensorflow as tf

from backend.train_autoencoder import (
    build_autoencoder,
    calculate_reconstruction_error,
    compute_anomaly_threshold,
    train_autoencoder_model,
)


def test_build_autoencoder():
    input_dim = 16
    autoencoder, encoder, decoder = build_autoencoder(input_dim)

    dummy_input = np.random.randn(5, input_dim).astype(np.float32)
    latent = encoder.predict(dummy_input, verbose=0)
    reconstructed = autoencoder.predict(dummy_input, verbose=0)

    assert latent.shape == (5, 8)
    assert reconstructed.shape == (5, input_dim)


def test_calculate_reconstruction_error():
    input_dim = 10
    autoencoder, _, _ = build_autoencoder(input_dim)
    dummy_input = np.ones((4, input_dim), dtype=np.float32)

    errors = calculate_reconstruction_error(autoencoder, dummy_input)
    assert len(errors) == 4
    assert np.all(errors >= 0)


def test_compute_anomaly_threshold():
    errors = np.array([0.1, 0.2, 0.15, 0.25, 0.3])
    thresh = compute_anomaly_threshold(errors, std_multiplier=2.0)
    expected = float(np.mean(errors) + 2.0 * np.std(errors))
    assert pytest.approx(thresh) == expected


def test_train_autoencoder_model(tmp_path):
    model_dir = str(tmp_path / "models")
    X_dummy = np.random.randn(30, 12).astype(np.float32)

    autoencoder, encoder, threshold_data, history = train_autoencoder_model(
        X_train=X_dummy, epochs=2, batch_size=8, model_dir=model_dir
    )

    assert os.path.exists(os.path.join(model_dir, "autoencoder.keras"))
    assert os.path.exists(os.path.join(model_dir, "encoder.keras"))
    assert os.path.exists(os.path.join(model_dir, "threshold.json"))

    assert "threshold" in threshold_data
    assert threshold_data["threshold"] > 0

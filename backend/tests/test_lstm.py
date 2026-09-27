import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pytest
import tensorflow as tf

from backend.train_lstm import (
    reshape_for_lstm,
    build_lstm_classifier,
    train_lstm_model,
    evaluate_lstm_predictions,
)


def test_reshape_for_lstm():
    X_2d = np.random.randn(10, 16)
    X_3d = reshape_for_lstm(X_2d)
    assert X_3d.shape == (10, 16, 1)

    # Idempotent if already 3D
    X_already_3d = reshape_for_lstm(X_3d)
    assert X_already_3d.shape == (10, 16, 1)


def test_build_lstm_classifier():
    input_shape = (16, 1)
    model = build_lstm_classifier(input_shape)

    dummy_input = np.random.randn(5, 16, 1).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)

    assert output.shape == (5, 1)
    assert np.all(output >= 0.0) and np.all(output <= 1.0)


def test_train_lstm_model(tmp_path):
    model_dir = str(tmp_path / "models")
    X_dummy = np.random.randn(25, 12).astype(np.float32)
    y_dummy = np.random.choice([0, 1], size=25)

    model, history = train_lstm_model(
        X_train=X_dummy,
        y_train=y_dummy,
        epochs=2,
        batch_size=8,
        model_dir=model_dir,
    )

    save_file = os.path.join(model_dir, "lstm_model.keras")
    assert os.path.exists(save_file)

    metrics, probs, preds = evaluate_lstm_predictions(model, X_dummy, y_dummy)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(probs) == 25
    assert len(preds) == 25

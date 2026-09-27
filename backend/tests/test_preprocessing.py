import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib
import numpy as np
import pandas as pd
import pytest

from backend.preprocessing import (
    load_dataset,
    split_features_target,
    prepare_training_data,
    preprocess_for_prediction,
    build_preprocessor_pipeline,
    DEFAULT_PREPROCESSOR_PATH,
)


def test_load_dataset_exists():
    df = load_dataset("data/loan_data.csv")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "Loan_Status" in df.columns


def test_load_dataset_nonexistent():
    with pytest.raises(FileNotFoundError):
        load_dataset("data/non_existent_file.csv")


def test_split_features_target():
    df = load_dataset("data/loan_data.csv")
    X, y = split_features_target(df)

    assert "Loan_ID" not in X.columns
    assert "Loan_Status" not in X.columns
    assert set(np.unique(y)).issubset({0, 1})
    assert len(X) == len(y)


def test_prepare_training_data(tmp_path):
    save_file = str(tmp_path / "test_preprocessor.joblib")
    X_train, X_test, y_train, y_test, preprocessor = prepare_training_data(
        csv_path="data/loan_data.csv", test_size=0.2, save_path=save_file
    )

    assert os.path.exists(save_file)
    assert X_train.ndim == 2
    assert X_test.ndim == 2
    assert X_train.shape[1] == X_test.shape[1]
    assert len(X_train) + len(X_test) == 381  # dataset row count
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()


def test_preprocess_for_prediction(tmp_path):
    save_file = str(tmp_path / "preprocessor.joblib")
    prepare_training_data(csv_path="data/loan_data.csv", save_path=save_file)

    sample_raw = pd.DataFrame(
        [
            {
                "Gender": "Male",
                "Married": "Yes",
                "Dependents": "0",
                "Education": "Graduate",
                "Self_Employed": "No",
                "ApplicantIncome": 5000,
                "CoapplicantIncome": 1500.0,
                "LoanAmount": 128.0,
                "Loan_Amount_Term": 360.0,
                "Credit_History": 1.0,
                "Property_Area": "Urban",
            }
        ]
    )

    processed = preprocess_for_prediction(sample_raw, preprocessor_path=save_file)
    assert processed.shape[0] == 1
    assert processed.ndim == 2
    assert not np.isnan(processed).any()

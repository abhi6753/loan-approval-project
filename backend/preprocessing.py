import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Explicit feature definitions matching Kaggle Loan Prediction Dataset
CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]
NUMERIC_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
]
BINARY_NUMERIC_FEATURES = ["Credit_History"]

ALL_INPUT_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BINARY_NUMERIC_FEATURES

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_PREPROCESSOR_PATH = os.path.join(DEFAULT_MODEL_DIR, "preprocessor.joblib")


def load_dataset(csv_path: str = "data/loan_data.csv") -> pd.DataFrame:
    """Loads the loan prediction dataset from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset CSV not found at path: {csv_path}. Please place loan_data.csv in the data directory."
        )
    df = pd.read_csv(csv_path)
    return df


def split_features_target(df: pd.DataFrame):
    """
    Strips identifier column Loan_ID and maps target column Loan_Status:
    Y -> 1, N -> 0.
    """
    df_clean = df.copy()
    if "Loan_ID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["Loan_ID"])

    if "Loan_Status" not in df_clean.columns:
        raise KeyError("Target column 'Loan_Status' missing from dataset.")

    y = df_clean["Loan_Status"].map({"Y": 1, "N": 0}).astype(int)
    X = df_clean.drop(columns=["Loan_Status"])
    return X, y


def build_preprocessor_pipeline() -> ColumnTransformer:
    """
    Builds a ColumnTransformer that handles categorical encoding,
    numeric scaling, and missing value imputation without data leakage.
    """
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    bin_num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
            ("num", num_pipeline, NUMERIC_FEATURES),
            ("bin_num", bin_num_pipeline, BINARY_NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def preprocess_for_training(
    X_train: pd.DataFrame, save_path: str = DEFAULT_PREPROCESSOR_PATH
):
    """
    Fits preprocessing pipeline ONLY on training data and saves fitted preprocessor to joblib file.
    Returns fitted preprocessor and transformed X_train array.
    """
    preprocessor = build_preprocessor_pipeline()
    X_train_processed = preprocessor.fit_transform(X_train)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(preprocessor, save_path)
    return preprocessor, X_train_processed


def preprocess_for_prediction(
    input_data: pd.DataFrame, preprocessor_path: str = DEFAULT_PREPROCESSOR_PATH
) -> np.ndarray:
    """
    Transforms incoming inference raw DataFrame using saved preprocessor.
    """
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(
            f"Fitted preprocessor not found at {preprocessor_path}. Train the models first."
        )

    preprocessor = joblib.load(preprocessor_path)
    # Ensure correct columns and order
    for col in ALL_INPUT_FEATURES:
        if col not in input_data.columns:
            input_data[col] = np.nan

    input_data = input_data[ALL_INPUT_FEATURES]
    processed_array = preprocessor.transform(input_data)
    return processed_array


def prepare_training_data(
    csv_path: str = "data/loan_data.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    save_path: str = DEFAULT_PREPROCESSOR_PATH,
):
    """
    Full pipeline helper to load data, split train/test, fit preprocessor on train,
    transform test, and return split arrays with target vectors.
    """
    df = load_dataset(csv_path)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor, X_train_processed = preprocess_for_training(X_train, save_path=save_path)
    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_processed,
        X_test_processed,
        y_train.values,
        y_test.values,
        preprocessor,
    )


if __name__ == "__main__":
    X_train_proc, X_test_proc, y_tr, y_te, _ = prepare_training_data()
    print(f"Data prepared successfully!")
    print(f"Train shape: {X_train_proc.shape}, Test shape: {X_test_proc.shape}")

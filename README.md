# Loan Approval Prediction Using Deep Learning

An academic full-stack deep learning application combining **Unit IV (LSTM Sequential Classification)** and **Unit V (Autoencoder Feature Reconstruction & Anomaly Detection)** trained on the Kaggle Loan Prediction Dataset.

---

## Technical Stack

- **Frontend**: Next.js 14, React 18, TypeScript, Tailwind CSS, Lucide Icons.
- **Backend**: FastAPI, Uvicorn, Pydantic, Scikit-Learn, Joblib, Pytest.
- **Machine Learning**: TensorFlow 2.15+, Keras, NumPy, Pandas, Matplotlib, Seaborn.

---

## Key Features

1. **Unit IV — LSTM Sequential Classifier**: Reshapes tabular features into 3D sequence tensors `(samples, timesteps, 1)` for step-by-step feature memory modeling using Recurrent LSTM cells.
2. **Unit V — Autoencoder Anomaly Detection**: Unsupervised feature reconstruction network ($X_{\text{train}} \to X_{\text{train}}$) that derives a statistical anomaly threshold ($\mu + 2\sigma$) from training reconstruction errors.
3. **Leakage-Free Preprocessing Pipeline**: Scikit-Learn `ColumnTransformer` handles missing values (mode for categorical/binary, median for numeric), One-Hot Encoding, and Standard Scaling. Pipeline is fitted strictly on $X_{\text{train}}$ and serialized to `preprocessor.joblib`.
4. **FastAPI REST Service**: Validated request schema (`LoanApplicationRequest`), model status health check, CORS middleware, and dual-model prediction execution.
5. **Modern Glassmorphic Next.js UI**: Responsive dashboard featuring loan application input form with quick presets, live probability gauge, anomaly status badge, and methodology guide.

---

## Dataset Overview (`data/loan_data.csv`)

- **Total Records**: 381 rows
- **Target Column**: `Loan_Status` (`Y` -> 1 [71.1%], `N` -> 0 [28.9%])
- **Input Features**:
  - Categorical: `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`
  - Continuous Numeric: `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`
  - Binary Indicator: `Credit_History`

---

## Project Structure

```
loan-approval-project/
├── backend/
│   ├── main.py               # FastAPI application entrypoint
│   ├── schemas.py            # Pydantic validation schemas
│   ├── inference.py          # Dual-model inference engine
│   ├── preprocessing.py      # ColumnTransformer pipeline
│   ├── train.py              # Full training orchestration script
│   ├── train_autoencoder.py  # Unit V Autoencoder trainer
│   ├── train_lstm.py         # Unit IV LSTM classifier trainer
│   ├── evaluate.py           # Empirical model evaluation & plot generator
│   ├── models/               # Serialized models (.keras, .joblib, .json)
│   └── tests/                # Pytest test suite
├── data/
│   └── loan_data.csv         # Kaggle Loan Dataset
├── docs/
│   ├── architecture.md       # Architecture & Mermaid diagrams
│   ├── api.md                # REST API documentation
│   ├── model.md              # Deep learning model specification
│   └── results/              # Evaluation plots (.png) & metrics (.json)
├── frontend/                 # Next.js 14 App Router UI
├── run_project.md            # Detailed execution guide
└── README.md
```

---

## Quickstart Instructions

### 1. Model Training
Fit preprocessor and train both Autoencoder and LSTM models:
```bash
python backend/train.py
```

### 2. Model Evaluation
Calculate empirical metrics on test split and generate evaluation plots in `docs/results/`:
```bash
python backend/evaluate.py
```

### 3. Run Backend API Server
Launch FastAPI backend server on `http://localhost:8000`:
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Run Frontend UI
In a separate terminal:
```bash
cd frontend
npm run dev
```
Open `http://localhost:3000` in your web browser.

---

## Testing
Run complete Pytest test suite covering preprocessing, Autoencoder, LSTM, and FastAPI endpoints:
```bash
pytest backend/tests/ -v
```

---

## Academic Disclaimer
This project is an academic research prototype developed for coursework demonstration. Predictions and anomaly indicators must NOT be used for real-world commercial loan underwriting or credit decisions.

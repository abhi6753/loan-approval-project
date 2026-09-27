# Quickstart & Project Execution Guide

This guide provides step-by-step terminal commands to set up, train, evaluate, test, and run the **Loan Approval Prediction Using Deep Learning** project on Windows (PowerShell / Command Prompt).

---

## Prerequisite Verification
Ensure Node.js and Python 3.12 are installed:
```powershell
python --version
node -v
cmd /c npm -v
```

---

## 1. Setup Environment & Install Dependencies

### Backend Python Virtual Environment
```powershell
# Create virtual environment (if not already created)
python -m venv .venv

# Install backend dependencies via pip
cmd /c ".venv\Scripts\pip install -r backend/requirements.txt"
```

### Frontend Dependencies
```powershell
cd frontend
cmd /c npm install
cd ..
```

---

## 2. Train Models & Preprocessing Pipeline
To fit the scikit-learn preprocessor and train both **Unit IV (LSTM)** and **Unit V (Autoencoder)** models:

```powershell
cmd /c ".venv\Scripts\python backend/train.py"
```

This generates and saves artifacts to `backend/models/`:
- `preprocessor.joblib`
- `autoencoder.keras`
- `encoder.keras`
- `lstm_model.keras`
- `threshold.json`

---

## 3. Run Empirical Evaluation & Plot Generation
To compute evaluation metrics on the test set and generate performance plots in `docs/results/`:

```powershell
cmd /c ".venv\Scripts\python backend/evaluate.py"
```

Generates:
- `docs/results/confusion_matrix.png`
- `docs/results/lstm_accuracy.png`
- `docs/results/lstm_loss.png`
- `docs/results/reconstruction_error.png`
- `docs/results/autoencoder_loss.png`
- `docs/results/evaluation_metrics.json`

---

## 4. Run Pytest Automated Unit Tests
To verify all preprocessing, model architecture, and FastAPI endpoint tests:

```powershell
cmd /c ".venv\Scripts\pytest backend/tests/ -v"
```

---

## 5. Run Backend FastAPI Server
To start the FastAPI REST backend server on `http://localhost:8000`:

```powershell
cmd /c ".venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"
```

Interactive API documentation will be available at:
`http://localhost:8000/docs`

---

## 6. Run Next.js Frontend
In a separate terminal window:

```powershell
cd frontend
cmd /c npm run dev
```

Open your browser at:
`http://localhost:3000`

---

## 7. Full Automated Verification Script
To execute end-to-end model training, evaluation, and test suite in one step:

```powershell
cmd /c ".venv\Scripts\python backend/train.py && .venv\Scripts\python backend/evaluate.py && .venv\Scripts\pytest backend/tests/ -v"
```

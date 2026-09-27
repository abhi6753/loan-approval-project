import os
import sys

# Ensure project root directory is added to sys.path regardless of execution CWD
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from backend.inference import LoanInferenceEngine
except (ModuleNotFoundError, ImportError) as err:
    sys.stderr.write(
        "\n==========================================================\n"
        "ERROR: Failed to import project dependencies.\n"
        f"Details: {err}\n\n"
        "Please execute the script using the project virtual environment:\n"
        "  .\\.venv\\Scripts\\python.exe run_test_cases.py\n"
        "==========================================================\n\n"
    )
    sys.exit(1)

engine = LoanInferenceEngine()
engine.load_models()

cases = [
    {
        "id": "TC-01",
        "name": "Prime Eligible Applicant",
        "expected": "Approved",
        "input": {
            "gender": "Male",
            "married": "Yes",
            "dependents": "1",
            "education": "Graduate",
            "self_employed": "No",
            "applicant_income": 7500,
            "coapplicant_income": 2500,
            "loan_amount": 100,
            "loan_amount_term": 60,
            "credit_history": 1.0,
            "property_area": "Semiurban"
        }
    },
    {
        "id": "TC-02",
        "name": "Self-Employed Moderate Applicant",
        "expected": "Approved",
        "input": {
            "gender": "Male",
            "married": "Yes",
            "dependents": "2",
            "education": "Graduate",
            "self_employed": "Yes",
            "applicant_income": 6000,
            "coapplicant_income": 1500,
            "loan_amount": 140,
            "loan_amount_term": 48,
            "credit_history": 1.0,
            "property_area": "Urban"
        }
    },
    {
        "id": "TC-03",
        "name": "Rejected (Unfavorable Credit History)",
        "expected": "Not Approved",
        "input": {
            "gender": "Female",
            "married": "No",
            "dependents": "0",
            "education": "Graduate",
            "self_employed": "No",
            "applicant_income": 5000,
            "coapplicant_income": 0,
            "loan_amount": 120,
            "loan_amount_term": 60,
            "credit_history": 0.0,
            "property_area": "Urban"
        }
    },
    {
        "id": "TC-04",
        "name": "High-Risk Outlier / Anomaly",
        "expected": "Not Approved & Anomaly Flagged",
        "input": {
            "gender": "Female",
            "married": "No",
            "dependents": "3+",
            "education": "Not Graduate",
            "self_employed": "Yes",
            "applicant_income": 1800,
            "coapplicant_income": 0,
            "loan_amount": 350,
            "loan_amount_term": 36,
            "credit_history": 0.0,
            "property_area": "Rural"
        }
    },
    {
        "id": "TC-05",
        "name": "Low Income & Default History",
        "expected": "Not Approved",
        "input": {
            "gender": "Male",
            "married": "No",
            "dependents": "0",
            "education": "Not Graduate",
            "self_employed": "No",
            "applicant_income": 1200,
            "coapplicant_income": 0,
            "loan_amount": 280,
            "loan_amount_term": 24,
            "credit_history": 0.0,
            "property_area": "Rural"
        }
    }
]

print("==========================================================")
print("     LOAN APPROVAL SYSTEM - MANUAL TEST CASE RUNNER      ")
print("==========================================================")
for c in cases:
    res = engine.predict(c["input"])
    print(f"[{c['id']}] {c['name']}")
    print(f"  Inputs: Income=Rs.{c['input']['applicant_income']}, CoIncome=Rs.{c['input']['coapplicant_income']}, Loan=Rs.{c['input']['loan_amount']}k, Term={c['input']['loan_amount_term']}m, Credit={c['input']['credit_history']}, Education={c['input']['education']}")
    print(f"  Result Label:          {res['label']}")
    print(f"  Approval Probability:  {res['approval_probability'] * 100:.2f}%")
    print(f"  Reconstruction Error:  {res['reconstruction_error']} (Threshold: {res['anomaly_threshold']})")
    print(f"  Anomaly Flag:          {res['anomaly_flag']} -> {'Requires Review' if res['anomaly_flag'] else 'Verified Normal'}")
    print("----------------------------------------------------------")

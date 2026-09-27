"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Calculator, RotateCcw, Sparkles, AlertCircle } from "lucide-react";

export default function LoanForm() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const initialForm = {
    gender: "Male",
    married: "Yes",
    dependents: "0",
    education: "Graduate",
    self_employed: "No",
    applicant_income: "5000",
    coapplicant_income: "1500",
    loan_amount: "100",
    loan_amount_term: "48",
    credit_history: "1.0",
    property_area: "Urban",
  };

  const [formData, setFormData] = useState(initialForm);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleReset = () => {
    setFormData(initialForm);
    setErrorMessage(null);
  };

  const fillHighCreditPreset = () => {
    setFormData({
      gender: "Male",
      married: "Yes",
      dependents: "1",
      education: "Graduate",
      self_employed: "No",
      applicant_income: "7500",
      coapplicant_income: "2500",
      loan_amount: "100",
      loan_amount_term: "48",
      credit_history: "1.0",
      property_area: "Semiurban",
    });
  };

  const fillRiskPreset = () => {
    setFormData({
      gender: "Female",
      married: "No",
      dependents: "3+",
      education: "Not Graduate",
      self_employed: "Yes",
      applicant_income: "1800",
      coapplicant_income: "0",
      loan_amount: "350",
      loan_amount_term: "36",
      credit_history: "0.0",
      property_area: "Rural",
    });
  };

  const fillTC05Preset = () => {
    setFormData({
      gender: "Male",
      married: "No",
      dependents: "0",
      education: "Not Graduate",
      self_employed: "No",
      applicant_income: "1200",
      coapplicant_income: "0",
      loan_amount: "280",
      loan_amount_term: "24",
      credit_history: "0.0",
      property_area: "Rural",
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setErrorMessage(null);

    const appIncome = isNaN(parseFloat(formData.applicant_income)) ? 0 : parseFloat(formData.applicant_income);
    const coappIncome = isNaN(parseFloat(formData.coapplicant_income)) ? 0 : parseFloat(formData.coapplicant_income);
    const loanAmt = isNaN(parseFloat(formData.loan_amount)) ? 0 : parseFloat(formData.loan_amount);
    const termVal = isNaN(parseFloat(formData.loan_amount_term)) ? 48 : parseFloat(formData.loan_amount_term);
    const creditVal = isNaN(parseFloat(formData.credit_history)) ? 1.0 : parseFloat(formData.credit_history);

    // Client-side numeric bounds check
    if (appIncome < 0 || coappIncome < 0 || loanAmt < 0) {
      setErrorMessage("Income and Loan amounts cannot be negative.");
      setLoading(false);
      return;
    }

    if (termVal > 60) {
      setErrorMessage("Loan amount term cannot exceed 60 months (5 years).");
      setLoading(false);
      return;
    }

    const payload = {
      gender: formData.gender,
      married: formData.married,
      dependents: formData.dependents,
      education: formData.education,
      self_employed: formData.self_employed,
      applicant_income: appIncome,
      coapplicant_income: coappIncome,
      loan_amount: loanAmt,
      loan_amount_term: termVal,
      credit_history: creditVal,
      property_area: formData.property_area,
    };

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

    try {
      const response = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        if (response.status === 503) {
          throw new Error("Machine learning models are not trained yet on backend server.");
        }
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned status ${response.status}`);
      }

      const result = await response.json();
      sessionStorage.setItem("latestLoanResult", JSON.stringify({ result, input: payload }));
      router.push("/result");
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to communicate with FastAPI backend server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-panel rounded-2xl p-6 sm:p-8 max-w-4xl mx-auto shadow-2xl">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6 pb-6 border-b border-slate-800">
        <div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            <Calculator className="w-5 h-5 text-cyan-400" />
            Loan Applicant Credentials
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Fill in the applicant details below to evaluate real-time loan approval eligibility.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={fillHighCreditPreset}
            className="px-3 py-1.5 rounded-lg text-xs bg-emerald-500/10 text-emerald-300 border border-emerald-500/20 hover:bg-emerald-500/20 transition-colors"
          >
            Sample: High Eligibility (TC-01)
          </button>
          <button
            type="button"
            onClick={fillRiskPreset}
            className="px-3 py-1.5 rounded-lg text-xs bg-rose-500/10 text-rose-300 border border-rose-500/20 hover:bg-rose-500/20 transition-colors"
          >
            Sample: High Risk (TC-04)
          </button>
        </div>
      </div>

      {errorMessage && (
        <div className="mb-6 p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm flex items-start gap-3">
          <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold block">Submission Error</span>
            <span>{errorMessage}</span>
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Personal & Categorical Info */}
        <div>
          <h3 className="text-xs uppercase font-mono tracking-wider text-cyan-400 mb-3">
            Applicant Information
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">Gender</label>
              <select
                name="gender"
                value={formData.gender}
                onChange={handleChange}
                className="w-full bg-slate-900/90 border border-slate-700/80 rounded-xl px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-cyan-500"
              >
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">Marital Status</label>
              <select
                name="married"
                value={formData.married}
                onChange={handleChange}
                className="w-full bg-slate-900/90 border border-slate-700/80 rounded-xl px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-cyan-500"
              >
                <option value="Yes">Married (Yes)</option>
                <option value="No">Unmarried (No)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">Dependents</label>
              <select
                name="dependents"
                value={formData.dependents}
                onChange={handleChange}
                className="w-full bg-slate-900/90 border border-slate-700/80 rounded-xl px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-cyan-500"
              >
                <option value="0">0</option>
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3+">3+</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">Education</label>
              <select
                name="education"
                value={formData.education}
                onChange={handleChange}
                className="w-full bg-slate-900/90 border border-slate-700/80 rounded-xl px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-cyan-500"
              >
                <option value="Graduate">Graduate</option>
                <option value="Not Graduate">Not Graduate</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">Self Employed</label>
              <select
                name="self_employed"
                value={formData.self_employed}
                onChange={handleChange}
                className="w-full bg-slate-900/90 border border-slate-700/80 rounded-xl px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-cyan-500"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">Property Area</label>
              <select
                name="property_area"
                value={formData.property_area}
                onChange={handleChange}
                className="w-full bg-slate-900/90 border border-slate-700/80 rounded-xl px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-cyan-500"
              >
                <option value="Urban">Urban</option>
                <option value="Semiurban">Semiurban</option>
                <option value="Rural">Rural</option>
              </select>
            </div>
          </div>
        </div>

        {/* Financial Information */}
        <div>
          <h3 className="text-xs uppercase font-mono tracking-wider text-cyan-400 mb-3">
            Financial & Loan Specifications
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">
                Applicant Monthly Income (₹)
              </label>
              <input
                type="number"
                name="applicant_income"
                value={formData.applicant_income}
                onChange={handleChange}
                min="0"
                step="1"
                placeholder="e.g. 5000"
                required
                className="w-full bg-slate-900/90 border border-slate-700/80 rounded-xl px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-cyan-500"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">
                Coapplicant Monthly Income (₹)
              </label>
              <input
                type="number"
                name="coapplicant_income"
                value={formData.coapplicant_income}
                onChange={handleChange}
                min="0"
                step="1"
                placeholder="e.g. 1500"
                required
                className="w-full bg-slate-900/90 border border-slate-700/80 rounded-xl px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-cyan-500"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">
                Loan Amount (₹ in Thousands)
              </label>
              <input
                type="number"
                name="loan_amount"
                value={formData.loan_amount}
                onChange={handleChange}
                min="1"
                step="1"
                placeholder="e.g. 100"
                required
                className="w-full bg-slate-900/90 border border-slate-700/80 rounded-xl px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-cyan-500"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">
                Loan Amount Term (Tenure)
              </label>
              <select
                name="loan_amount_term"
                value={formData.loan_amount_term}
                onChange={handleChange}
                className="w-full bg-slate-900/90 border border-slate-700/80 rounded-xl px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-cyan-500"
              >
                <option value="60">60 Months (5 Years)</option>
                <option value="48">48 Months (4 Years)</option>
                <option value="36">36 Months (3 Years)</option>
                <option value="24">24 Months (2 Years)</option>
                <option value="12">12 Months (1 Year)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">
                Credit History Status
              </label>
              <select
                name="credit_history"
                value={formData.credit_history}
                onChange={handleChange}
                className="w-full bg-slate-900/90 border border-slate-700/80 rounded-xl px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-cyan-500"
              >
                <option value="1.0">1.0 (Meets Credit History Requirements)</option>
                <option value="0.0">0.0 (Unfavorable / Default History)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Buttons */}
        <div className="flex items-center justify-end gap-3 pt-4 border-t border-slate-800">
          <button
            type="button"
            onClick={handleReset}
            disabled={loading}
            className="px-4 py-2.5 rounded-xl border border-slate-700 text-slate-300 hover:bg-slate-800 text-sm font-medium flex items-center space-x-2 transition-colors disabled:opacity-50"
          >
            <RotateCcw className="w-4 h-4" />
            <span>Reset</span>
          </button>

          <button
            type="submit"
            disabled={loading}
            className="gradient-btn px-6 py-2.5 rounded-xl text-sm font-semibold flex items-center space-x-2 disabled:opacity-50"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Evaluating Loan Application...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                <span>Check Loan Approval Eligibility</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}

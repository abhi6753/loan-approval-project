"use client";

import Link from "next/link";
import { CheckCircle2, XCircle, AlertTriangle, Activity, Cpu, ArrowLeft, ShieldCheck } from "lucide-react";

interface ResultProps {
  data: {
    result: {
      prediction: number;
      label: string;
      approval_probability: number;
      reconstruction_error: number;
      anomaly_threshold: number;
      anomaly_flag: boolean;
      decision_threshold: number;
      disclaimer: string;
    };
    input?: any;
  };
}

export default function ResultCard({ data }: ResultProps) {
  const { result, input } = data;
  const isApproved = result.prediction === 1;
  const probPercentage = Math.round(result.approval_probability * 100);

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div className="glass-panel rounded-2xl p-6 sm:p-8 shadow-2xl border border-slate-800 relative overflow-hidden">
        {/* Decorative background glow */}
        <div
          className={`absolute -top-24 -right-24 w-64 h-64 rounded-full blur-3xl opacity-20 pointer-events-none ${isApproved ? "bg-emerald-500" : "bg-rose-500"
            }`}
        />

        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-800">
          <div>
            <span className="text-xs font-mono uppercase tracking-wider text-slate-400">
              Loan Application Evaluation Output
            </span>
            <h2 className="text-2xl font-bold text-slate-100 flex items-center gap-2 mt-1">
              Decision:{" "}
              <span className={isApproved ? "text-emerald-400" : "text-rose-400"}>
                {result.label}
              </span>
            </h2>
          </div>

          <div className="flex items-center gap-2">
            {isApproved ? (
              <div className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
                <CheckCircle2 className="w-5 h-5" />
                <span className="font-semibold text-sm">Approved</span>
              </div>
            ) : (
              <div className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400">
                <XCircle className="w-5 h-5" />
                <span className="font-semibold text-sm">Not Approved</span>
              </div>
            )}
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-6">
          {/* Approval Probability */}
          <div className="glass-card rounded-xl p-5 border border-slate-700/50">
            <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
              <span className="flex items-center gap-1.5 font-medium text-slate-300">
                <Cpu className="w-4 h-4 text-cyan-400" /> Approval Probability Score
              </span>
              <span className="font-mono text-cyan-300 font-bold">{probPercentage}%</span>
            </div>

            <div className="w-full bg-slate-800 h-3 rounded-full overflow-hidden p-0.5 mb-2">
              <div
                className={`h-full rounded-full transition-all duration-1000 ${isApproved
                  ? "bg-gradient-to-r from-emerald-500 to-cyan-400"
                  : "bg-gradient-to-r from-rose-500 to-amber-500"
                  }`}
                style={{ width: `${probPercentage}%` }}
              />
            </div>

            <p className="text-[11px] text-slate-400">
              Predictive approval probability score calculated from financial credentials. Cutoff threshold: 50%.
            </p>
          </div>

          {/* Profile Risk Verification */}
          <div className="glass-card rounded-xl p-5 border border-slate-700/50">
            <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
              <span className="flex items-center gap-1.5 font-medium text-slate-300">
                <Activity className="w-4 h-4 text-indigo-400" /> Profile Verification Index
              </span>
              <span className="font-mono text-indigo-300">
                Index: {result.reconstruction_error}
              </span>
            </div>

            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-slate-400">Tolerance Limit: {result.anomaly_threshold}</span>
              {result.anomaly_flag ? (
                <span className="px-2.5 py-1 rounded-md bg-amber-500/15 border border-amber-500/30 text-amber-300 text-xs font-semibold flex items-center gap-1">
                  <AlertTriangle className="w-3.5 h-3.5" /> Requires Review
                </span>
              ) : (
                <span className="px-2.5 py-1 rounded-md bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 text-xs font-semibold flex items-center gap-1">
                  <ShieldCheck className="w-3.5 h-3.5" /> Verified Normal Profile
                </span>
              )}
            </div>

            <p className="text-[11px] text-slate-400">
              Evaluates profile risk variance against historical approval baselines.
            </p>
          </div>
        </div>

        {/* Submitted Input Summary */}
        {input && (
          <div className="mb-6 p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-xs">
            <h4 className="font-semibold text-slate-300 mb-2">Applicant Financial Summary</h4>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-slate-400 font-mono">
              <div>Income: ₹{input.applicant_income.toLocaleString('en-IN')}/mo</div>
              <div>Co-Income: ₹{input.coapplicant_income.toLocaleString('en-IN')}/mo</div>
              <div>Loan Amount: ₹{input.loan_amount}k</div>
              <div>Tenure: {input.loan_amount_term} Months</div>
              <div>Credit History: {Number(input.credit_history) === 1 ? 'Clear (1.0)' : 'Unfavorable / Default (0.0)'}</div>
              <div>Education: {input.education}</div>
              <div>Married: {input.married}</div>
              <div>Property: {input.property_area}</div>
            </div>
          </div>
        )}

        {/* System Disclaimer */}
        <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 text-slate-300 text-xs flex items-start gap-3">
          <ShieldCheck className="w-4 h-4 shrink-0 mt-0.5 text-cyan-400" />
          <div>
            <span className="font-semibold block text-slate-200">Automated Decision Disclaimer</span>
            <span>This is an automated loan prediction assessment powered by artificial intelligence predictive modeling.</span>
          </div>
        </div>

        {/* Action Link */}
        <div className="mt-6 pt-6 border-t border-slate-800 flex justify-between items-center">
          <Link
            href="/apply"
            className="px-4 py-2 rounded-xl border border-slate-700 hover:bg-slate-800 text-slate-300 text-sm font-medium flex items-center space-x-2 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Evaluate Another Application</span>
          </Link>
        </div>
      </div>
    </div>
  );
}

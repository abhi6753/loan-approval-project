import Link from "next/link";
import { Landmark, ArrowRight, ShieldCheck, Zap, Activity, CheckCircle2 } from "lucide-react";

export default function HomePage() {
  return (
    <div className="space-y-16 py-4">
      {/* Hero Section */}
      <section className="text-center max-w-4xl mx-auto space-y-6">
        <div className="inline-flex items-center space-x-2 px-3 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-medium">
          <Zap className="w-4 h-4 text-cyan-400" />
          <span>AI-Powered Credit Risk & Eligibility System</span>
        </div>

        <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight text-slate-100 leading-tight">
          Smart Loan Approval <span className="gradient-text">Prediction Portal</span>
        </h1>

        <p className="text-base sm:text-lg text-slate-300 max-w-2xl mx-auto leading-relaxed">
          Instantly evaluate applicant credit eligibility, financial ratios, loan tenure, and risk scores using advanced predictive intelligence.
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
          <Link
            href="/apply"
            className="gradient-btn px-8 py-3.5 rounded-xl font-semibold flex items-center space-x-2 text-sm w-full sm:w-auto justify-center shadow-lg shadow-cyan-500/20 hover:scale-105 transition-transform"
          >
            <span>Start Loan Assessment</span>
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </section>

      {/* Core Feature Cards */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="glass-panel rounded-2xl p-6 space-y-3 border border-slate-800 hover:border-cyan-500/40 transition-colors">
          <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
            <CheckCircle2 className="w-5 h-5" />
          </div>
          <h3 className="text-lg font-bold text-slate-100">Real-Time Probability</h3>
          <p className="text-xs text-slate-300 leading-relaxed">
            Evaluates financial income, co-applicant contribution, loan amount, and credit history to output approval probability.
          </p>
        </div>

        <div className="glass-panel rounded-2xl p-6 space-y-3 border border-slate-800 hover:border-indigo-500/40 transition-colors">
          <div className="w-10 h-10 rounded-xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
            <Activity className="w-5 h-5" />
          </div>
          <h3 className="text-lg font-bold text-slate-100">Risk & Anomaly Flagging</h3>
          <p className="text-xs text-slate-300 leading-relaxed">
            Automated verification checks profile variance against historical baselines to flag high-risk or irregular applications.
          </p>
        </div>

        <div className="glass-panel rounded-2xl p-6 space-y-3 border border-slate-800 hover:border-blue-500/40 transition-colors">
          <div className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-500/30 flex items-center justify-center text-blue-400">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <h3 className="text-lg font-bold text-slate-100">Tenure & Income Ratios</h3>
          <p className="text-xs text-slate-300 leading-relaxed">
            Flexible loan amount term inputs (12 to 360 months) with INR currency formatting and instant eligibility feedback.
          </p>
        </div>
      </section>
    </div>
  );
}

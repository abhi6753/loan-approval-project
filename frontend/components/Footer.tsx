import { ShieldCheck } from "lucide-react";

export default function Footer() {
  return (
    <footer className="border-t border-slate-800/80 bg-slate-950 py-8 px-4 mt-auto">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-slate-400">
        <div>
          <p className="font-semibold text-slate-300">
            Loan Approval Prediction & Eligibility System
          </p>
          <p className="mt-0.5">Powered by Advanced Deep Learning Predictive Modeling</p>
        </div>

        <div className="flex items-center space-x-2 bg-slate-900/80 text-slate-300 px-3 py-1.5 rounded-md border border-slate-800 max-w-lg">
          <ShieldCheck className="w-4 h-4 text-cyan-400 shrink-0" />
          <span>
            Automated Credit Risk & Loan Eligibility Decision Engine
          </span>
        </div>
      </div>
    </footer>
  );
}

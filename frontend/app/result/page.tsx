"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import ResultCard from "@/components/ResultCard";
import { FileText, ArrowLeft, RefreshCw } from "lucide-react";

export default function ResultPage() {
  const [resultData, setResultData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const saved = sessionStorage.getItem("latestLoanResult");
    if (saved) {
      try {
        setResultData(JSON.parse(saved));
      } catch (err) {
        console.error("Failed to parse saved loan result", err);
      }
    }
    setLoading(false);
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-slate-400">
        <RefreshCw className="w-8 h-8 animate-spin text-cyan-400 mb-3" />
        <p className="text-sm font-medium">Loading evaluation result...</p>
      </div>
    );
  }

  if (!resultData) {
    return (
      <div className="max-w-md mx-auto text-center py-16 glass-panel rounded-2xl p-8 border border-slate-800 space-y-4">
        <div className="w-12 h-12 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center mx-auto text-slate-400">
          <FileText className="w-6 h-6" />
        </div>
        <h2 className="text-xl font-bold text-slate-200">No Assessment Found</h2>
        <p className="text-xs text-slate-400">
          Please submit a loan application form to generate a live prediction from the Unit IV LSTM and Unit V Autoencoder models.
        </p>
        <Link
          href="/apply"
          className="gradient-btn px-5 py-2.5 rounded-xl font-semibold text-xs inline-flex items-center space-x-2"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Go to Loan Application Form</span>
        </Link>
      </div>
    );
  }

  return (
    <div className="py-4">
      <ResultCard data={resultData} />
    </div>
  );
}

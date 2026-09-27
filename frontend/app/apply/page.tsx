import LoanForm from "@/components/LoanForm";

export default function ApplyPage() {
  return (
    <div className="py-4">
      <div className="text-center max-w-2xl mx-auto mb-8 space-y-2">
        <h1 className="text-3xl font-bold text-slate-100">
          Loan Application Assessment
        </h1>
        <p className="text-sm text-slate-400">
          Enter the applicant's financial details, income, loan amount, tenure, and credit history to check instant approval eligibility.
        </p>
      </div>

      <LoanForm />
    </div>
  );
}

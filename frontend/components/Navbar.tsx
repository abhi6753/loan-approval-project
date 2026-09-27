"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Landmark, FileText, Home, Sparkles } from "lucide-react";

export default function Navbar() {
  const pathname = usePathname();

  const navLinks = [
    { name: "Home", href: "/", icon: Home },
    { name: "Apply Loan", href: "/apply", icon: FileText },
    { name: "Prediction Result", href: "/result", icon: Sparkles },
  ];

  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-slate-800/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center space-x-3 group">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20 group-hover:scale-105 transition-transform">
            <Landmark className="w-5 h-5 text-white" />
          </div>
          <div>
            <span className="font-bold text-lg text-slate-100 tracking-tight block leading-none">
              Loan<span className="text-cyan-400">Approve.AI</span>
            </span>
            <span className="text-[10px] text-slate-400 font-medium">Smart Loan Eligibility Portal</span>
          </div>
        </Link>

        <nav className="flex items-center space-x-1 sm:space-x-2">
          {navLinks.map((link) => {
            const Icon = link.icon;
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`px-3.5 py-2 rounded-lg text-xs sm:text-sm font-medium flex items-center space-x-1.5 transition-colors ${
                  isActive
                    ? "bg-cyan-500/15 text-cyan-300 border border-cyan-500/30"
                    : "text-slate-300 hover:text-white hover:bg-slate-800/60"
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{link.name}</span>
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}

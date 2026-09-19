"use client";

import React, { useState, useRef, useEffect } from "react";
import { LogOut, User as UserIcon, ShieldCheck } from "lucide-react";
import { useAuth } from "@/context/AuthContext";

export default function UserMenu() {
  const { user, logout } = useAuth();
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  if (!user) return null;

  const initial = (user.full_name || user.email)[0].toUpperCase();

  return (
    <div className="relative" ref={menuRef}>
      <button
        type="button"
        onClick={() => setIsOpen((prev) => !prev)}
        className="flex items-center gap-2 p-1.5 rounded-xl hover:bg-slate-800/80 border border-slate-800 transition-colors"
        title="Account Menu"
      >
        <div className="h-6 w-6 rounded-lg bg-indigo-600/30 border border-indigo-500/40 text-indigo-300 font-semibold text-xs flex items-center justify-center">
          {initial}
        </div>
        <span className="text-xs font-medium text-slate-200 max-w-[120px] sm:max-w-[160px] truncate hidden sm:inline-block">
          {user.full_name || user.email}
        </span>
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-64 rounded-xl bg-slate-900 border border-slate-800 shadow-2xl p-2 z-50 animate-in fade-in-0 zoom-in-95 duration-150">
          <div className="p-2 border-b border-slate-800 mb-1">
            <div className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-lg bg-indigo-600/30 border border-indigo-500/40 text-indigo-300 font-bold text-sm flex items-center justify-center shrink-0">
                {initial}
              </div>
              <div className="min-w-0 flex-1">
                <div className="text-xs font-semibold text-slate-100 truncate">
                  {user.full_name || "Researcher"}
                </div>
                <div className="text-[11px] text-slate-400 truncate mt-0.5">
                  {user.email}
                </div>
              </div>
            </div>
            <div className="mt-2 flex items-center gap-1 text-[10px] text-emerald-400 bg-emerald-950/40 px-2 py-0.5 rounded-md border border-emerald-800/40 w-fit">
              <ShieldCheck className="h-3 w-3" />
              <span>Isolated Workspace</span>
            </div>
          </div>

          <button
            type="button"
            onClick={() => {
              setIsOpen(false);
              logout();
            }}
            className="w-full flex items-center gap-2 px-2.5 py-2 rounded-lg text-xs text-red-400 hover:text-red-300 hover:bg-red-950/40 transition-colors cursor-pointer"
          >
            <LogOut className="h-3.5 w-3.5" />
            <span>Sign Out</span>
          </button>
        </div>
      )}
    </div>
  );
}

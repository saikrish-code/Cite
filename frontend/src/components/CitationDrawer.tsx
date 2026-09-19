"use client";

import React, { useEffect } from "react";
import { X, FileText, Bookmark, Sparkles, BookOpen } from "lucide-react";
import { Citation } from "@/lib/api";

interface CitationDrawerProps {
  citation: Citation | null;
  onClose: () => void;
}

export default function CitationDrawer({
  citation,
  onClose,
}: CitationDrawerProps) {
  // Listen for Escape key to close
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  if (!citation) return null;

  const sourceId =
    citation.source_id ??
    (citation as unknown as { citation_id?: number }).citation_id ??
    1;
  const filename =
    citation.filename ||
    (citation as unknown as { document_name?: string }).document_name ||
    "Document";
  const page =
    citation.page ??
    (citation as unknown as { page_number?: number }).page_number ??
    1;
  const rawScore = citation.score ?? 0.9;
  const scorePercent = Math.min(100, Math.max(0, Math.round(rawScore * 100)));

  return (
    <div className="fixed inset-0 z-50 flex justify-end pointer-events-none">
      {/* Dimmed backdrop */}
      <div
        onClick={onClose}
        className="fixed inset-0 bg-black/60 backdrop-blur-xs transition-opacity pointer-events-auto"
        aria-hidden="true"
      />

      {/* Slide-in panel */}
      <div className="relative w-full max-w-md h-full bg-slate-900 border-l border-slate-800 shadow-2xl flex flex-col pointer-events-auto z-10 animate-in slide-in-from-right duration-250">
        {/* Header */}
        <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/90 backdrop-blur-md">
          <div className="flex items-center gap-2">
            <span className="flex items-center justify-center h-6 w-6 rounded-md bg-indigo-600/30 text-indigo-300 font-mono text-xs font-bold border border-indigo-500/40">
              [{sourceId}]
            </span>
            <h3 className="font-semibold text-sm text-slate-100">
              Cited Source Evidence
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-100 transition-colors"
            title="Close panel"
            aria-label="Close panel"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Panel Body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          {/* Document metadata card */}
          <div className="bg-slate-800/50 border border-slate-700/60 rounded-xl p-4 space-y-3">
            <div className="flex items-start gap-3">
              <div className="p-2 rounded-lg bg-indigo-950/60 text-indigo-400 border border-indigo-800/40 shrink-0">
                <FileText className="h-5 w-5" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
                  Source Document
                </div>
                <div className="font-medium text-sm text-slate-100 truncate mt-0.5" title={filename}>
                  {filename}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 pt-2 border-t border-slate-700/50 text-xs">
              <div className="flex items-center gap-2 text-slate-300">
                <BookOpen className="h-4 w-4 text-indigo-400" />
                <span>
                  Page <strong className="text-slate-100 font-mono">{page}</strong>
                </span>
              </div>
              <div className="flex items-center gap-2 text-slate-300">
                <Sparkles className="h-4 w-4 text-emerald-400" />
                <span>
                  Similarity: <strong className="text-slate-100 font-mono">{scorePercent}%</strong>
                </span>
              </div>
            </div>

            {/* Relevance meter */}
            <div className="space-y-1 pt-1">
              <div className="w-full bg-slate-700/60 h-1.5 rounded-full overflow-hidden">
                <div
                  className="bg-gradient-to-r from-indigo-500 to-emerald-400 h-full rounded-full transition-all duration-300"
                  style={{ width: `${scorePercent}%` }}
                />
              </div>
            </div>
          </div>

          {/* Snippet passage */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span className="font-medium uppercase tracking-wider text-[11px] flex items-center gap-1.5">
                <Bookmark className="h-3.5 w-3.5 text-indigo-400" />
                Retrieved Context Snippet
              </span>
            </div>

            <div className="relative p-4 rounded-xl bg-slate-950/80 border border-slate-800 text-slate-200 text-xs leading-relaxed font-sans shadow-inner">
              <div className="absolute top-2 left-2 text-slate-700 select-none text-2xl font-serif leading-none">
                “
              </div>
              <p className="pl-4 pt-1 whitespace-pre-wrap selection:bg-indigo-500/40">
                {citation.snippet}
              </p>
            </div>
          </div>

          {/* Provenance note */}
          <div className="p-3 bg-indigo-950/20 border border-indigo-900/30 rounded-lg text-[11px] text-indigo-300/80 flex items-start gap-2 leading-relaxed">
            <Sparkles className="h-4 w-4 text-indigo-400 shrink-0 mt-0.5" />
            <p>
              This passage was retrieved directly from the indexed document and supplied to the model as grounding context for citation <strong>[{citation.source_id}]</strong>.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-slate-800 bg-slate-900/60 flex items-center justify-between text-xs text-slate-400">
          <span className="font-mono text-[11px]">ID: src-{citation.source_id}</span>
          <button
            onClick={onClose}
            className="px-3 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium transition-colors"
          >
            Dismiss
          </button>
        </div>
      </div>
    </div>
  );
}

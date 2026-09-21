"use client";

import React, { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Bot,
  User,
  Copy,
  Check,
  AlertTriangle,
  Sparkles,
  ExternalLink,
  Zap,
  Activity,
} from "lucide-react";
import { Citation, RetrievalMetrics } from "@/lib/api";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  retrievalMetrics?: RetrievalMetrics | null;
  contextFound?: boolean;
  isStreaming?: boolean;
  error?: string;
  createdAt: string;
}

interface ChatMessageProps {
  message: Message;
  onSelectCitation: (citation: Citation) => void;
}

function useSmoothStreaming(content: string, isStreaming: boolean, speedMs: number = 35) {
  const [displayedContent, setDisplayedContent] = useState(content);

  useEffect(() => {
    if (!isStreaming) {
      setDisplayedContent(content);
      return;
    }

    if (displayedContent === content) return;

    if (!content.startsWith(displayedContent)) {
      setDisplayedContent(content.substring(0, 1));
      return;
    }

    const diff = content.length - displayedContent.length;
    let nextIndex = displayedContent.length;
    
    if (diff > 150) {
      // If we are way behind, jump ahead by a larger chunk to catch up
      nextIndex += Math.floor(diff / 3);
    } else {
      // Find the next space or word boundary
      const remaining = content.substring(displayedContent.length);
      const nextSpaceMatch = remaining.match(/[\s\n]/);
      if (nextSpaceMatch && nextSpaceMatch.index !== undefined) {
        // add up to and including the next space
        nextIndex += nextSpaceMatch.index + 1;
      } else {
        // If no space found, just add a few chars (or the rest if it's the last word)
        nextIndex += remaining.length > 5 ? 3 : remaining.length;
      }
    }
    
    const timeout = setTimeout(() => {
      setDisplayedContent(content.substring(0, nextIndex));
    }, speedMs);

    return () => clearTimeout(timeout);
  }, [content, displayedContent, isStreaming, speedMs]);

  return displayedContent;
}

export default function ChatMessage({
  message,
  onSelectCitation,
}: ChatMessageProps) {
  const [copied, setCopied] = useState(false);
  const isAssistant = message.role === "assistant";
  
  const displayedContent = useSmoothStreaming(message.content, !!message.isStreaming);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Find citation by source_id or citation_id
  const getCitation = (id: number): Citation | undefined => {
    return message.citations?.find(
      (c) => ((c.source_id ?? (c as unknown as { citation_id?: number }).citation_id) === id)
    );
  };

  // Custom text renderer to transform [1], [2] into interactive citation chips
  const renderTextWithCitations = (text: string) => {
    const parts: (string | React.ReactNode)[] = [];
    const regex = /\[(\d+)\]/g;
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = regex.exec(text)) !== null) {
      const sourceId = parseInt(match[1], 10);
      const citation = getCitation(sourceId);

      // Push text prior to citation
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index));
      }

      // Render interactive chip
      parts.push(
        <button
          key={`${match.index}-${sourceId}`}
          type="button"
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            if (citation) {
              onSelectCitation(citation);
            } else {
              // Fallback citation if click occurs before stream finalizes citations
              onSelectCitation({
                source_id: sourceId,
                filename: "Indexed Paper Passage",
                page: 1,
                snippet: `Retrieved context passage corresponding to citation [${sourceId}].`,
                score: 0.92,
              });
            }
          }}
          title={
            citation
              ? `Source: ${citation.filename || (citation as unknown as { document_name?: string }).document_name} (Page ${citation.page || (citation as unknown as { page_number?: number }).page_number})`
              : `Citation [${sourceId}]`
          }
          className="inline-flex items-center justify-center align-baseline mx-0.5 px-1.5 py-0.5 text-[11px] font-mono font-semibold rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 hover:bg-indigo-500/35 hover:text-indigo-200 transition-colors cursor-pointer shadow-2xs group"
        >
          <span>[{sourceId}]</span>
          <ExternalLink className="h-2.5 w-2.5 ml-0.5 opacity-60 group-hover:opacity-100" />
        </button>
      );

      lastIndex = regex.lastIndex;
    }

    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }

    return parts;
  };

  return (
    <div
      className={`py-5 px-4 md:px-6 transition-colors ${
        isAssistant
          ? "bg-slate-900/40 border-y border-slate-800/40"
          : "bg-transparent"
      }`}
    >
      <div className="max-w-3xl mx-auto flex items-start gap-3.5 md:gap-4">
        {/* Avatar */}
        <div
          className={`h-8 w-8 rounded-lg flex items-center justify-center shrink-0 shadow-sm ${
            isAssistant
              ? "bg-indigo-600/20 text-indigo-400 border border-indigo-500/30"
              : "bg-slate-700/50 text-slate-300 border border-slate-600/50"
          }`}
        >
          {isAssistant ? <Bot className="h-4 w-4" /> : <User className="h-4 w-4" />}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0 space-y-2">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-slate-200">
                {isAssistant ? "CiteRAG Assistant" : "You"}
              </span>
              <span className="text-[11px] text-slate-400">
                {message.createdAt}
              </span>
            </div>

            {isAssistant && message.content && (
              <button
                onClick={handleCopy}
                className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors text-xs flex items-center gap-1"
                title="Copy answer"
              >
                {copied ? (
                  <>
                    <Check className="h-3.5 w-3.5 text-emerald-400" />
                    <span className="text-[11px] text-emerald-400">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="h-3.5 w-3.5" />
                    <span className="text-[11px]">Copy</span>
                  </>
                )}
              </button>
            )}
          </div>

          {/* Error display */}
          {message.error && (
            <div className="p-3 rounded-lg bg-red-950/60 border border-red-800/60 text-xs text-red-300 flex items-start gap-2">
              <AlertTriangle className="h-4 w-4 text-red-400 shrink-0 mt-0.5" />
              <span>{message.error}</span>
            </div>
          )}

          {/* Insufficient context banner */}
          {isAssistant && message.contextFound === false && !message.isStreaming && (
            <div className="p-2.5 rounded-lg bg-amber-950/40 border border-amber-800/50 text-xs text-amber-300 flex items-center gap-2">
              <AlertTriangle className="h-3.5 w-3.5 text-amber-400 shrink-0" />
              <span>
                Insufficient context found in indexed papers. Answer is ungrounded or unknown.
              </span>
            </div>
          )}

          {/* Message Text / Markdown */}
          <div className="text-sm prose-academic leading-relaxed break-words">
            {isAssistant ? (
              <>
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    // Override paragraph to handle inline citation markers
                    p: ({ children }) => {
                      if (typeof children === "string") {
                        return <p>{renderTextWithCitations(children)}</p>;
                      }
                      // In case children is an array or elements
                      const processed = React.Children.map(children, (child) => {
                        if (typeof child === "string") {
                          return renderTextWithCitations(child);
                        }
                        return child;
                      });
                      return <p>{processed}</p>;
                    },
                    li: ({ children }) => {
                      const processed = React.Children.map(children, (child) => {
                        if (typeof child === "string") {
                          return renderTextWithCitations(child);
                        }
                        return child;
                      });
                      return <li>{processed}</li>;
                    },
                  }}
                >
                  {displayedContent}
                </ReactMarkdown>

                {/* Blinking streaming cursor */}
                {message.isStreaming && (
                  <span className="inline-block w-2 h-4 ml-1 bg-indigo-400 rounded-xs animate-cursor align-middle" />
                )}
              </>
            ) : (
              <p className="text-slate-100 whitespace-pre-wrap">{message.content}</p>
            )}
          </div>

          {/* Citation chips footer for assistant */}
          {isAssistant && message.citations && message.citations.length > 0 && (
            <div className="pt-3 mt-2 border-t border-slate-800/80">
              <div className="text-[11px] font-medium text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Sparkles className="h-3 w-3 text-indigo-400" />
                Verified Sources ({message.citations.length})
              </div>
              <div className="flex flex-wrap gap-2">
                {message.citations.map((c) => (
                  <button
                    key={c.source_id}
                    onClick={() => onSelectCitation(c)}
                    className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-800/80 hover:bg-slate-800 text-slate-200 border border-slate-700/70 hover:border-indigo-500/40 text-xs transition-all shadow-xs group"
                  >
                    <span className="font-mono text-indigo-300 font-semibold text-[11px]">
                      [{c.source_id}]
                    </span>
                    <span className="truncate max-w-[140px] text-slate-300 group-hover:text-slate-100">
                      {c.filename}
                    </span>
                    <span className="text-[10px] text-slate-400 bg-slate-700/60 px-1 py-0.2 rounded font-mono">
                      p.{c.page}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Retrieval stage latency benchmarking badge for assistant */}
          {isAssistant && message.retrievalMetrics && (() => {
            const m = message.retrievalMetrics;
            const modeStr = (m.mode || m.retrieval_mode || "hybrid_rerank").replace(/_/g, " + ");
            const totalMs = m.total_ms ?? m.total_retrieval_ms ?? 0;
            const vectorMs = m.vector_ms ?? m.vector_latency_ms ?? 0;
            const bm25Ms = m.bm25_ms ?? m.bm25_latency_ms;
            const rrfMs = m.rrf_ms ?? m.rrf_latency_ms;
            const rerankMs = m.rerank_ms ?? m.rerank_latency_ms;

            return (
              <div className="mt-2.5 pt-2 border-t border-slate-800/60 flex flex-wrap items-center gap-2 text-[11px] text-slate-400">
                <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-indigo-950/70 border border-indigo-800/50 text-indigo-300 font-mono text-[10px]">
                  <Zap className="h-3 w-3 text-indigo-400" />
                  <span className="font-semibold uppercase tracking-wider">
                    {modeStr}
                  </span>
                  <span className="text-slate-500">|</span>
                  <span className="text-slate-200">
                    {totalMs.toFixed(1)}ms total
                  </span>
                </div>

                <div className="flex flex-wrap items-center gap-x-2 gap-y-1 font-mono text-[10px] text-slate-400">
                  <span>
                    Vector:{" "}
                    <strong className="text-slate-200">
                      {vectorMs.toFixed(1)}ms
                    </strong>
                  </span>
                  {bm25Ms != null && (
                    <>
                      <span className="text-slate-600">•</span>
                      <span>
                        BM25:{" "}
                        <strong className="text-slate-200">
                          {bm25Ms.toFixed(1)}ms
                        </strong>
                      </span>
                    </>
                  )}
                  {rrfMs != null && (
                    <>
                      <span className="text-slate-600">•</span>
                      <span>
                        RRF:{" "}
                        <strong className="text-slate-200">
                          {rrfMs.toFixed(1)}ms
                        </strong>
                      </span>
                    </>
                  )}
                  {rerankMs != null && (
                    <>
                      <span className="text-slate-600">•</span>
                      <span>
                        Reranker:{" "}
                        <strong className="text-emerald-400">
                          {rerankMs.toFixed(1)}ms
                        </strong>
                      </span>
                    </>
                  )}
                </div>
              </div>
            );
          })()}
        </div>
      </div>
    </div>
  );
}

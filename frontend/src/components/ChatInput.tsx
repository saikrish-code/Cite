"use client";

import React, { useState, useRef, useEffect } from "react";
import { Send, Square, Sparkles, CornerDownLeft } from "lucide-react";

interface ChatInputProps {
  onSendMessage: (query: string) => void;
  isStreaming: boolean;
  onStopStreaming: () => void;
  disabled?: boolean;
  activeDocName?: string | null;
}

export default function ChatInput({
  onSendMessage,
  isStreaming,
  onStopStreaming,
  disabled = false,
  activeDocName = null,
}: ChatInputProps) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea as user types
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        200
      )}px`;
    }
  }, [input]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (isStreaming) {
      onStopStreaming();
      return;
    }
    const trimmed = input.trim();
    if (!trimmed || disabled) return;

    onSendMessage(trimmed);
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto px-4 md:px-6 pb-4 pt-2">
      <form
        onSubmit={handleSubmit}
        className="relative flex flex-col bg-slate-900 border border-slate-800 focus-within:border-indigo-500/60 focus-within:ring-1 focus-within:ring-indigo-500/30 rounded-2xl shadow-xl transition-all"
      >
        {/* Scope indicator banner */}
        {activeDocName && (
          <div className="px-3.5 pt-2 pb-0 flex items-center gap-1.5 text-[11px] text-indigo-300">
            <span className="h-1.5 w-1.5 rounded-full bg-indigo-400 animate-pulse" />
            <span className="text-slate-400">Scoped to:</span>
            <span className="font-medium truncate max-w-[280px] text-indigo-300">
              {activeDocName}
            </span>
          </div>
        )}

        {/* Input Area */}
        <div className="flex items-end px-3.5 py-2.5 gap-2">
          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            placeholder={
              activeDocName
                ? `Ask anything about ${activeDocName}...`
                : "Ask questions across your indexed research papers..."
            }
            className="w-full resize-none bg-transparent text-slate-100 text-sm placeholder:text-slate-500 focus:outline-hidden py-1.5 max-h-48 min-h-[28px] leading-relaxed"
          />

          <div className="flex items-center gap-1.5 shrink-0 mb-0.5">
            {isStreaming ? (
              <button
                type="button"
                onClick={onStopStreaming}
                className="h-8 w-8 rounded-lg bg-red-600/20 text-red-400 border border-red-500/40 hover:bg-red-600/30 flex items-center justify-center transition-all"
                title="Stop generation"
              >
                <Square className="h-3.5 w-3.5 fill-current" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim() || disabled}
                className="h-8 w-8 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:hover:bg-indigo-600 text-white flex items-center justify-center shadow-md transition-all cursor-pointer disabled:cursor-not-allowed"
                title="Send question (Enter)"
              >
                <Send className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        </div>

        {/* Hints footer */}
        <div className="px-3.5 pb-2 pt-0 flex items-center justify-between text-[11px] text-slate-400 select-none">
          <span className="hidden sm:inline-flex items-center gap-1">
            <CornerDownLeft className="h-3 w-3 text-slate-400" />
            <kbd className="font-sans">Enter</kbd> to send, <kbd className="font-sans">Shift+Enter</kbd> for newline
          </span>
          <span className="flex items-center gap-1 ml-auto text-indigo-400/80">
            <Sparkles className="h-3 w-3" />
            Grounded Citations Enforced
          </span>
        </div>
      </form>
    </div>
  );
}

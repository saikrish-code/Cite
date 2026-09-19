"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Menu,
  Sparkles,
  BookOpen,
  Trash2,
  AlertCircle,
  RefreshCw,
  FolderOpen,
} from "lucide-react";
import DocumentSidebar from "@/components/DocumentSidebar";
import ChatMessage, { Message } from "@/components/ChatMessage";
import ChatInput from "@/components/ChatInput";
import CitationDrawer from "@/components/CitationDrawer";
import {
  DocumentItem,
  Citation,
  listDocuments,
  streamChat,
} from "@/lib/api";

const STARTER_PROMPTS = [
  "Summarize the primary contributions and findings.",
  "What methodology and experimental setup were evaluated?",
  "What are the key limitations or future research directions?",
  "Extract the core quantitative results and metrics.",
];

let messageCounter = 0;
function createMessageId(prefix: string): string {
  messageCounter += 1;
  return `${prefix}-${Date.now()}-${messageCounter}`;
}

function getCurrentTimestamp(): string {
  return new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function Home() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [backendConnected, setBackendConnected] = useState<boolean | null>(null);
  const [isLoadingDocs, setIsLoadingDocs] = useState(true);

  const abortControllerRef = useRef<AbortController | null>(null);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll chat to bottom
  const scrollToBottom = useCallback(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  // Fetch documents list and test backend health
  const refreshDocuments = useCallback(async () => {
    try {
      const docs = await listDocuments();
      setDocuments(docs);
      setBackendConnected(true);
    } catch (err) {
      console.error("Failed to load documents:", err);
      setBackendConnected(false);
    } finally {
      setIsLoadingDocs(false);
    }
  }, []);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const docs = await listDocuments();
        if (active) {
          setDocuments(docs);
          setBackendConnected(true);
        }
      } catch (err) {
        console.error("Failed to load documents:", err);
        if (active) {
          setBackendConnected(false);
        }
      } finally {
        if (active) {
          setIsLoadingDocs(false);
        }
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  // Scroll on message updates
  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Selected document helper
  const selectedDoc = documents.find((d) => d.document_id === selectedDocId);

  // Handle sending a question
  const handleSendMessage = async (queryText: string) => {
    if (isStreaming) return;

    const userMessageId = createMessageId("user");
    const assistantMessageId = createMessageId("assistant");
    const timestamp = getCurrentTimestamp();

    const userMessage: Message = {
      id: userMessageId,
      role: "user",
      content: queryText,
      createdAt: timestamp,
    };

    const assistantPlaceholder: Message = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      citations: [],
      contextFound: true,
      isStreaming: true,
      createdAt: timestamp,
    };

    setMessages((prev) => [...prev, userMessage, assistantPlaceholder]);
    setIsStreaming(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    let accumulated = "";

    await streamChat(
      {
        query: queryText,
        document_id: selectedDocId,
        k: 4,
      },
      {
        onToken: (token) => {
          accumulated += token;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMessageId
                ? { ...m, content: accumulated }
                : m
            )
          );
        },
        onCitations: (citations, contextFound) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMessageId
                ? { ...m, citations, contextFound }
                : m
            )
          );
        },
        onDone: (_query, fullAnswer) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMessageId
                ? { ...m, content: fullAnswer || accumulated, isStreaming: false }
                : m
            )
          );
          setIsStreaming(false);
          abortControllerRef.current = null;
        },
        onError: (errorMessage) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMessageId
                ? {
                    ...m,
                    isStreaming: false,
                    error: errorMessage,
                    content: m.content || "Failed to generate answer.",
                  }
                : m
            )
          );
          setIsStreaming(false);
          abortControllerRef.current = null;
        },
      },
      controller.signal
    );
  };

  const handleStopStreaming = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
    setMessages((prev) =>
      prev.map((m) => (m.isStreaming ? { ...m, isStreaming: false } : m))
    );
  };

  const handleClearChat = () => {
    handleStopStreaming();
    setMessages([]);
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-950 text-slate-100 antialiased">
      {/* Mobile Sidebar Overlay */}
      {mobileSidebarOpen && (
        <div className="fixed inset-0 z-40 md:hidden flex">
          <div
            className="fixed inset-0 bg-black/60 backdrop-blur-xs"
            onClick={() => setMobileSidebarOpen(false)}
          />
          <div className="relative w-80 max-w-[85vw] h-full z-50">
            <DocumentSidebar
              documents={documents}
              selectedDocId={selectedDocId}
              onSelectDoc={(id) => {
                setSelectedDocId(id);
                setMobileSidebarOpen(false);
              }}
              onDocumentsChange={refreshDocuments}
              onCloseMobile={() => setMobileSidebarOpen(false)}
            />
          </div>
        </div>
      )}

      {/* Desktop Sidebar */}
      <div className="hidden md:block w-80 lg:w-88 shrink-0 h-full">
        <DocumentSidebar
          documents={documents}
          selectedDocId={selectedDocId}
          onSelectDoc={setSelectedDocId}
          onDocumentsChange={refreshDocuments}
        />
      </div>

      {/* Main Chat View */}
      <div className="flex-1 flex flex-col h-full min-w-0 bg-slate-950 relative">
        {/* Top Navigation Bar */}
        <header className="h-14 border-b border-slate-800/80 px-4 md:px-6 flex items-center justify-between bg-slate-900/40 backdrop-blur-md shrink-0 z-10">
          <div className="flex items-center gap-3 min-w-0">
            {/* Mobile menu trigger */}
            <button
              onClick={() => setMobileSidebarOpen(true)}
              className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-100 md:hidden"
              title="Open document sidebar"
            >
              <Menu className="h-5 w-5" />
            </button>

            {/* Branding */}
            <div className="flex items-center gap-2">
              <div className="h-7 w-7 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-bold text-xs shadow-md shadow-indigo-600/30">
                CR
              </div>
              <div className="flex items-baseline gap-1.5">
                <span className="font-semibold text-sm tracking-tight text-slate-100">
                  CiteRAG
                </span>
                <span className="text-[11px] text-indigo-400 hidden sm:inline-block font-mono">
                  research assistant
                </span>
              </div>
            </div>

            {/* Scope pill */}
            {selectedDoc && (
              <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-indigo-950/60 border border-indigo-800/50 text-[11px] text-indigo-300 truncate max-w-xs">
                <span className="h-1.5 w-1.5 rounded-full bg-indigo-400" />
                <span className="text-slate-400">Scoped:</span>
                <span className="truncate font-medium">{selectedDoc.filename}</span>
              </div>
            )}
          </div>

          {/* Right Header Actions */}
          <div className="flex items-center gap-2.5">
            {/* Backend status indicator */}
            <div
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-850 border border-slate-800 text-[11px]"
              title={
                backendConnected === true
                  ? "Backend connected (FastAPI)"
                  : backendConnected === false
                  ? "Backend offline"
                  : "Checking backend..."
              }
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  backendConnected === true
                    ? "bg-emerald-400 shadow-xs shadow-emerald-400/50 animate-pulse"
                    : backendConnected === false
                    ? "bg-red-400"
                    : "bg-amber-400"
                }`}
              />
              <span className="text-slate-400 hidden sm:inline">
                {backendConnected === true
                  ? "FastAPI API"
                  : backendConnected === false
                  ? "Offline"
                  : "Connecting"}
              </span>
            </div>

            {/* Clear chat button */}
            {messages.length > 0 && (
              <button
                onClick={handleClearChat}
                className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors text-xs flex items-center gap-1"
                title="Clear conversation"
              >
                <Trash2 className="h-4 w-4" />
                <span className="hidden sm:inline text-xs">Clear</span>
              </button>
            )}
          </div>
        </header>

        {/* Backend Offline Warning Alert */}
        {backendConnected === false && (
          <div className="bg-red-950/80 border-b border-red-800/80 px-4 py-2 text-xs text-red-200 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertCircle className="h-4 w-4 text-red-400 shrink-0" />
              <span>
                Cannot reach FastAPI backend at <code>http://localhost:8000</code>. Make sure the backend server is running.
              </span>
            </div>
            <button
              onClick={refreshDocuments}
              className="flex items-center gap-1 underline hover:text-white font-medium"
            >
              <RefreshCw className="h-3 w-3" /> Retry
            </button>
          </div>
        )}

        {/* Chat Messages or Empty State */}
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            /* Empty State Hero */
            <div className="h-full flex flex-col items-center justify-center px-4 py-8 text-center max-w-2xl mx-auto">
              <div className="h-14 w-14 rounded-2xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center text-white shadow-xl shadow-indigo-600/25 mb-5 ring-4 ring-indigo-500/10">
                <BookOpen className="h-7 w-7" />
              </div>

              <h1 className="text-xl md:text-2xl font-bold tracking-tight text-slate-100 mb-2">
                CiteRAG Academic Assistant
              </h1>

              <p className="text-sm text-slate-400 max-w-md mb-8 leading-relaxed">
                Upload research papers to ask questions grounded directly in literature. Answers stream live with interactive, verifiable citation chips.
              </p>

              {/* Starter prompts */}
              <div className="w-full space-y-2">
                <div className="text-[11px] font-medium uppercase tracking-wider text-slate-400 mb-3 flex items-center justify-center gap-1.5">
                  <Sparkles className="h-3.5 w-3.5 text-indigo-400" />
                  Suggested Questions
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-left">
                  {STARTER_PROMPTS.map((prompt, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleSendMessage(prompt)}
                      className="p-3 rounded-xl bg-slate-900/70 border border-slate-800/80 hover:border-indigo-500/40 hover:bg-slate-900 text-xs text-slate-300 hover:text-slate-100 transition-all text-left group flex items-start justify-between gap-2 shadow-xs"
                    >
                      <span>{prompt}</span>
                      <Sparkles className="h-3.5 w-3.5 text-slate-400 group-hover:text-indigo-400 shrink-0 mt-0.5 transition-colors" />
                    </button>
                  ))}
                </div>
              </div>

              {/* Prompt to upload if no docs */}
              {documents.length === 0 && !isLoadingDocs && (
                <div className="mt-8 p-3 rounded-xl bg-indigo-950/20 border border-indigo-900/40 text-xs text-indigo-300/80 flex items-center gap-2">
                  <FolderOpen className="h-4 w-4 text-indigo-400 shrink-0" />
                  <span>
                    Upload a PDF, DOCX, or TXT file using the sidebar to begin searching!
                  </span>
                </div>
              )}
            </div>
          ) : (
            /* Message List */
            <div className="divide-y divide-slate-800/30">
              {messages.map((msg) => (
                <ChatMessage
                  key={msg.id}
                  message={msg}
                  onSelectCitation={(citation) => setActiveCitation(citation)}
                />
              ))}
              <div ref={chatBottomRef} className="h-4" />
            </div>
          )}
        </div>

        {/* Chat Input Bar */}
        <ChatInput
          onSendMessage={handleSendMessage}
          isStreaming={isStreaming}
          onStopStreaming={handleStopStreaming}
          disabled={backendConnected === false}
          activeDocName={selectedDoc?.filename}
        />
      </div>

      {/* Slide-out Citation Side Panel Drawer */}
      <CitationDrawer
        citation={activeCitation}
        onClose={() => setActiveCitation(null)}
      />
    </div>
  );
}

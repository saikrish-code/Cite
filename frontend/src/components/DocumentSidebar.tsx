"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  UploadCloud,
  FileText,
  Trash2,
  CheckCircle2,
  Clock,
  AlertCircle,
  RefreshCw,
  Layers,
  Filter,
  X,
} from "lucide-react";
import { DocumentItem, uploadDocument, deleteDocument } from "@/lib/api";

interface UploadingFile {
  id: string;
  name: string;
  size: number;
  progress: number;
  status: "uploading" | "processing" | "ready" | "error";
  errorMessage?: string;
}

interface DocumentSidebarProps {
  documents: DocumentItem[];
  selectedDocId: string | null;
  onSelectDoc: (id: string | null) => void;
  onDocumentsChange: () => void;
  onCloseMobile?: () => void;
}

export default function DocumentSidebar({
  documents,
  selectedDocId,
  onSelectDoc,
  onDocumentsChange,
  onCloseMobile,
}: DocumentSidebarProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadQueue, setUploadQueue] = useState<UploadingFile[]>([]);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [errorBanner, setErrorBanner] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Poll for document status changes if any are "processing"
  useEffect(() => {
    const hasProcessing = documents.some((d) => d.status === "processing");
    if (!hasProcessing) return;

    const interval = setInterval(() => {
      onDocumentsChange();
    }, 2500);

    return () => clearInterval(interval);
  }, [documents, onDocumentsChange]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(Array.from(e.target.files));
      e.target.value = "";
    }
  };

  const handleFiles = async (files: File[]) => {
    setErrorBanner(null);
    for (const file of files) {
      const ext = file.name.split(".").pop()?.toLowerCase();
      if (!["pdf", "docx", "txt"].includes(ext || "")) {
        setErrorBanner(`"${file.name}" is not supported. Please upload PDF, DOCX, or TXT.`);
        continue;
      }

      const tempId = `${Date.now()}-${file.name}`;
      const newUpload: UploadingFile = {
        id: tempId,
        name: file.name,
        size: file.size,
        progress: 0,
        status: "uploading",
      };

      setUploadQueue((prev) => [newUpload, ...prev]);

      try {
        await uploadDocument(file, "anonymous", (progress) => {
          setUploadQueue((prev) =>
            prev.map((item) =>
              item.id === tempId ? { ...item, progress } : item
            )
          );
        });

        // Set to processing in uploadQueue
        setUploadQueue((prev) =>
          prev.map((item) =>
            item.id === tempId
              ? { ...item, progress: 100, status: "processing" }
              : item
          )
        );

        // Refresh documents list
        onDocumentsChange();

        // Clear finished upload after brief delay
        setTimeout(() => {
          setUploadQueue((prev) => prev.filter((item) => item.id !== tempId));
        }, 3000);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : "Upload failed";
        setUploadQueue((prev) =>
          prev.map((item) =>
            item.id === tempId
              ? { ...item, status: "error", errorMessage: message }
              : item
          )
        );
      }
    }
  };

  const handleDelete = async (e: React.MouseEvent, docId: string) => {
    e.stopPropagation();
    try {
      await deleteDocument(docId);
      if (selectedDocId === docId) {
        onSelectDoc(null);
      }
      onDocumentsChange();
    } catch (err: unknown) {
      setErrorBanner(err instanceof Error ? err.message : "Failed to delete document");
    }
  };

  const handleManualRefresh = async () => {
    setIsRefreshing(true);
    await onDocumentsChange();
    setTimeout(() => setIsRefreshing(false), 500);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <aside className="w-full h-full flex flex-col bg-slate-900/95 border-r border-slate-800 text-slate-200">
      {/* Sidebar Header */}
      <div className="p-4 border-b border-slate-800/80 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="h-8 w-8 rounded-lg bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
            <Layers className="h-4 w-4" />
          </div>
          <div>
            <h2 className="font-semibold text-sm text-slate-100 leading-none">
              Knowledge Base
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              {documents.length} {documents.length === 1 ? "document" : "documents"} indexed
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={handleManualRefresh}
            title="Refresh document list"
            aria-label="Refresh document list"
            className="p-1.5 rounded-md hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
          >
            <RefreshCw className={`h-4 w-4 ${isRefreshing ? "animate-spin text-indigo-400" : ""}`} />
          </button>
          {onCloseMobile && (
            <button
              onClick={onCloseMobile}
              className="p-1.5 rounded-md hover:bg-slate-800 text-slate-400 hover:text-slate-200 md:hidden"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {/* Error alert if any */}
      {errorBanner && (
        <div className="mx-3 mt-3 p-2.5 bg-red-950/60 border border-red-800/60 rounded-lg text-xs text-red-300 flex items-start justify-between gap-2">
          <span>{errorBanner}</span>
          <button onClick={() => setErrorBanner(null)} className="hover:text-red-100">
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      {/* Drag & Drop Upload Zone */}
      <div className="p-3">
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`relative border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-all duration-200 flex flex-col items-center justify-center group ${
            isDragging
              ? "border-indigo-400 bg-indigo-950/30 scale-[0.99]"
              : "border-slate-700/70 hover:border-indigo-500/50 hover:bg-slate-800/40"
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.docx,.txt"
            onChange={handleFileInputChange}
            className="hidden"
          />
          <div className="h-10 w-10 rounded-full bg-slate-800/80 border border-slate-700 flex items-center justify-center text-slate-300 group-hover:text-indigo-400 group-hover:border-indigo-500/30 group-hover:bg-indigo-950/20 transition-all mb-2">
            <UploadCloud className="h-5 w-5" />
          </div>
          <p className="text-xs font-medium text-slate-200 group-hover:text-indigo-300">
            Click to upload or drag & drop
          </p>
          <p className="text-[11px] text-slate-400 mt-1">PDF, DOCX, or TXT (up to 50MB)</p>
        </div>
      </div>

      {/* Uploading Queue Progress */}
      {uploadQueue.length > 0 && (
        <div className="px-3 pb-2 space-y-2">
          <div className="text-[11px] font-medium text-slate-400 uppercase tracking-wider px-1">
            Uploading ({uploadQueue.length})
          </div>
          {uploadQueue.map((item) => (
            <div
              key={item.id}
              className="p-2.5 rounded-lg bg-slate-800/80 border border-slate-700/60 text-xs"
            >
              <div className="flex items-center justify-between gap-2 mb-1.5">
                <div className="flex items-center gap-1.5 truncate max-w-[160px]">
                  <span className="font-medium text-slate-200 truncate" title={item.name}>
                    {item.name}
                  </span>
                  <span className="text-[10px] text-slate-400">({formatFileSize(item.size)})</span>
                </div>
                <span className="text-[11px] text-slate-400 shrink-0">
                  {item.status === "uploading" && `${item.progress}%`}
                  {item.status === "processing" && "Indexing..."}
                  {item.status === "ready" && "Done"}
                  {item.status === "error" && "Error"}
                </span>
              </div>
              {/* Progress bar */}
              <div className="w-full h-1.5 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-200 ${
                    item.status === "error"
                      ? "bg-red-500"
                      : item.status === "ready"
                      ? "bg-emerald-500"
                      : "bg-indigo-500"
                  }`}
                  style={{ width: `${item.progress}%` }}
                />
              </div>
              {item.errorMessage && (
                <div className="text-[11px] text-red-400 mt-1 truncate">
                  {item.errorMessage}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Filter / Scope Selector */}
      <div className="px-3 pt-2 pb-1 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
        <span className="font-medium uppercase tracking-wider text-[11px]">
          Target Documents
        </span>
        {selectedDocId && (
          <button
            onClick={() => onSelectDoc(null)}
            className="text-indigo-400 hover:text-indigo-300 text-[11px] flex items-center gap-1"
          >
            Reset filter
          </button>
        )}
      </div>

      {/* Document List */}
      <div className="flex-1 overflow-y-auto px-3 py-1 space-y-1.5">
        {/* All documents option */}
        <button
          onClick={() => onSelectDoc(null)}
          className={`w-full text-left p-2.5 rounded-lg border transition-all flex items-center justify-between text-xs ${
            selectedDocId === null
              ? "bg-indigo-600/15 border-indigo-500/40 text-indigo-200 shadow-sm"
              : "bg-slate-800/30 border-transparent text-slate-300 hover:bg-slate-800/70 hover:text-slate-100"
          }`}
        >
          <div className="flex items-center gap-2 min-w-0">
            <Filter className="h-3.5 w-3.5 text-indigo-400 shrink-0" />
            <span className="font-medium truncate">Search All Documents</span>
          </div>
          {selectedDocId === null && (
            <div className="h-2 w-2 rounded-full bg-indigo-400 shrink-0" />
          )}
        </button>

        {documents.length === 0 ? (
          <div className="py-8 text-center text-slate-500 text-xs">
            <FileText className="h-8 w-8 mx-auto mb-2 opacity-40 text-slate-400" />
            <p>No documents uploaded yet.</p>
            <p className="text-[11px] text-slate-600 mt-1">
              Upload papers to ask grounded questions.
            </p>
          </div>
        ) : (
          documents.map((doc) => {
            const isSelected = selectedDocId === doc.document_id;
            return (
              <div
                key={doc.document_id}
                onClick={() => onSelectDoc(doc.document_id)}
                className={`group relative p-2.5 rounded-lg border transition-all cursor-pointer flex flex-col gap-1.5 ${
                  isSelected
                    ? "bg-indigo-600/20 border-indigo-500/50 shadow-sm ring-1 ring-indigo-500/30"
                    : "bg-slate-800/40 border-slate-800/60 hover:border-slate-700 hover:bg-slate-800/80"
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0 flex-1">
                    <FileText className="h-4 w-4 text-slate-400 group-hover:text-indigo-400 shrink-0 transition-colors" />
                    <span
                      className="font-medium text-xs text-slate-200 truncate leading-snug"
                      title={doc.filename}
                    >
                      {doc.filename}
                    </span>
                  </div>
                  <button
                    onClick={(e) => handleDelete(e, doc.document_id)}
                    title="Delete document"
                    aria-label="Delete document"
                    className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-950/60 hover:text-red-400 text-slate-400 transition-all shrink-0"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>

                <div className="flex items-center justify-between text-[11px] text-slate-400 mt-0.5">
                  {/* Status Indicator */}
                  <div className="flex items-center gap-1.5">
                    {doc.status === "ready" && (
                      <span className="flex items-center gap-1 text-emerald-400">
                        <CheckCircle2 className="h-3 w-3" />
                        <span>Ready</span>
                      </span>
                    )}
                    {doc.status === "processing" && (
                      <span className="flex items-center gap-1 text-amber-400">
                        <Clock className="h-3 w-3 animate-pulse" />
                        <span>Indexing</span>
                      </span>
                    )}
                    {doc.status === "failed" && (
                      <span className="flex items-center gap-1 text-red-400">
                        <AlertCircle className="h-3 w-3" />
                        <span>Failed</span>
                      </span>
                    )}
                  </div>

                  {/* Chunk Count */}
                  {doc.status === "ready" && (
                    <span className="bg-slate-700/60 px-1.5 py-0.5 rounded text-[10px] text-slate-300">
                      {doc.chunk_count} {doc.chunk_count === 1 ? "chunk" : "chunks"}
                    </span>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Sidebar Footer */}
      <div className="p-3 border-t border-slate-800/80 text-[11px] text-slate-400 flex items-center justify-between">
        <span className="truncate">RAG Model: BGE + MiniLM</span>
        <span className="font-mono text-xs text-slate-400">v0.1.0</span>
      </div>
    </aside>
  );
}

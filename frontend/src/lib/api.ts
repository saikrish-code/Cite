/**
 * API client module for interacting with the CiteRAG FastAPI backend.
 * Handles authentication, document upload with progress tracking, document management,
 * and Server-Sent Events (SSE) streaming for grounded RAG chat.
 */

export interface User {
  id: string;
  email: string;
  full_name?: string | null;
  is_active: boolean;
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface Citation {
  source_id: number;
  filename: string;
  page: number;
  snippet: string;
  score: number;
}

export interface DocumentItem {
  document_id: string;
  filename: string;
  status: "processing" | "ready" | "failed";
  chunk_count: number;
  user_id: string;
  created_at: string;
}

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  status: "processing" | "ready" | "failed";
  chunk_count?: number;
  message?: string;
}

export interface RetrievalMetrics {
  mode?: string;
  retrieval_mode?: string;
  vector_ms?: number;
  bm25_ms?: number;
  rrf_ms?: number;
  rerank_ms?: number;
  total_ms?: number;
  vector_latency_ms?: number;
  bm25_latency_ms?: number | null;
  rrf_latency_ms?: number | null;
  rerank_latency_ms?: number | null;
  total_retrieval_ms?: number;
  vector_candidates?: number | null;
  bm25_candidates?: number | null;
  fused_candidates?: number | null;
  final_returned?: number | null;
}

export interface ChatRequest {
  query: string;
  k?: number;
  document_id?: string | null;
  user_id?: string | null;
  retrieval_mode?: "vector_only" | "hybrid" | "hybrid_rerank";
}

export interface ChatHistoryEntry {
  id: string;
  query: string;
  answer: string;
  citations: Citation[];
  context_found: boolean;
  created_at: string;
}

export interface ChatStreamCallbacks {
  onToken: (token: string) => void;
  onCitations: (
    citations: Citation[],
    contextFound: boolean,
    retrievalMetrics?: RetrievalMetrics | null
  ) => void;
  onDone: (query: string, fullAnswer: string) => void;
  onError: (error: string) => void;
}

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

/**
 * Register a new user account with email and password.
 */
export async function registerUser(
  email: string,
  password: string,
  fullName?: string
): Promise<AuthResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ email, password, full_name: fullName }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Registration failed with status ${response.status}`);
  }

  return response.json();
}

/**
 * Authenticate with email and password to receive a JWT access token.
 */
export async function loginUser(
  email: string,
  password: string
): Promise<AuthResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ email, password }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Login failed with status ${response.status}`);
  }

  return response.json();
}

/**
 * Fetch profile for current authenticated user.
 */
export async function getCurrentUser(token: string): Promise<User> {
  const response = await fetch(`${API_BASE_URL}/auth/me`, {
    method: "GET",
    headers: {
      Accept: "application/json",
      Authorization: `Bearer ${token}`,
    },
  });

  if (!response.ok) {
    throw new Error(`Authentication token invalid or expired (${response.status})`);
  }

  return response.json();
}

/**
 * Upload a document with real-time upload progress tracking and JWT authorization.
 */
export function uploadDocument(
  file: File,
  token?: string | null,
  onProgress?: (percentage: number) => void
): Promise<DocumentUploadResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const url = `${API_BASE_URL}/documents/upload`;

    xhr.open("POST", url, true);

    if (token) {
      xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    }

    if (xhr.upload && onProgress) {
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const percent = Math.round((event.loaded / event.total) * 100);
          onProgress(percent);
        }
      };
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const response: DocumentUploadResponse = JSON.parse(xhr.responseText);
          resolve(response);
        } catch {
          reject(new Error("Failed to parse upload response JSON."));
        }
      } else {
        try {
          const errorData = JSON.parse(xhr.responseText);
          reject(new Error(errorData.detail || `Upload failed with status ${xhr.status}`));
        } catch {
          reject(new Error(`Upload failed with HTTP ${xhr.status}: ${xhr.statusText}`));
        }
      }
    };

    xhr.onerror = () => {
      reject(new Error("Network error during document upload."));
    };

    const formData = new FormData();
    formData.append("file", file);
    xhr.send(formData);
  });
}

/**
 * Fetch the list of tracked documents belonging to the authenticated user.
 */
export async function listDocuments(token?: string | null): Promise<DocumentItem[]> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE_URL}/documents`, {
    method: "GET",
    headers,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to list documents (${response.status}): ${errorText}`);
  }

  const data = await response.json();
  return data.documents || [];
}

/**
 * Delete a document and its indexed vector chunks.
 */
export async function deleteDocument(
  documentId: string,
  token?: string | null
): Promise<void> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE_URL}/documents/${documentId}`, {
    method: "DELETE",
    headers,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to delete document (${response.status}): ${errorText}`);
  }
}

/**
 * Fetch chat conversation history for the authenticated user.
 */
export async function getChatHistory(
  token?: string | null,
  limit: number = 20,
  offset: number = 0
): Promise<ChatHistoryEntry[]> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const url = `${API_BASE_URL}/chat/history?limit=${limit}&offset=${offset}`;
  const response = await fetch(url, {
    method: "GET",
    headers,
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch chat history (${response.status})`);
  }

  return response.json();
}

/**
 * Stream a chat response using Server-Sent Events (SSE) with user isolation.
 */
export async function streamChat(
  request: ChatRequest,
  callbacks: ChatStreamCallbacks,
  token?: string | null,
  signal?: AbortSignal
): Promise<void> {
  const url = `${API_BASE_URL}/chat`;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "text/event-stream",
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(request),
      signal,
    });
  } catch (err: unknown) {
    if (signal?.aborted) {
      return;
    }
    callbacks.onError(err instanceof Error ? err.message : "Failed to connect to chat API.");
    return;
  }

  if (!response.ok) {
    const errBody = await response.text();
    callbacks.onError(`Chat error (${response.status}): ${errBody || response.statusText}`);
    return;
  }

  if (!response.body) {
    callbacks.onError("No response stream received from backend.");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let currentEvent = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) {
          currentEvent = "";
          continue;
        }

        if (trimmed.startsWith("event:")) {
          currentEvent = trimmed.slice(6).trim();
        } else if (trimmed.startsWith("data:")) {
          const dataStr = trimmed.slice(5).trim();
          try {
            const data = JSON.parse(dataStr);

            if (currentEvent === "token") {
              if (data.token) {
                callbacks.onToken(data.token);
              }
            } else if (currentEvent === "citations") {
              callbacks.onCitations(
                data.citations || [],
                data.context_found ?? true,
                data.retrieval_metrics || null
              );
            } else if (currentEvent === "done") {
              callbacks.onDone(data.query || request.query, data.answer || "");
            } else if (currentEvent === "error") {
              callbacks.onError(data.detail || "Server error occurred during streaming.");
            }
          } catch {
            // Non-JSON or partial data line
          }
        }
      }
    }
  } catch (err: unknown) {
    if (signal?.aborted) {
      return;
    }
    callbacks.onError(err instanceof Error ? err.message : "Stream reading interrupted.");
  }
}

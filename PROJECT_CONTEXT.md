# PROJECT_CONTEXT.md: CiteRAG — Production-Ready Research Paper Assistant

## 1. Project Overview & Portfolio Goals

**CiteRAG** is a production-quality, end-to-end Retrieval-Augmented Generation (RAG) system engineered specifically for academic literature, research papers, and technical documents. The application allows researchers and engineers to upload complex academic papers (PDFs), interrogate them through conversational natural language, and receive accurate answers backed by verifiable, page-level citations with zero unanchored hallucinations.

### SDE Internship Application Objective
This project is architected from the ground up to showcase core Software Development Engineer (SDE) competencies sought by top engineering teams:
- **Clean Architecture & Separation of Concerns:** Domain-driven, modular design with strict layer boundaries (API, Domain/Services, Data Access, Infrastructure).
- **Production Systems Engineering:** Non-blocking async I/O, resilient streaming, schema validation, database migrations, connection pooling, and multi-tenant security.
- **Advanced Retrieval Science:** Beyond naive vector lookups—implementing hybrid dense/sparse retrieval (Vector + BM25), Reciprocal Rank Fusion (RRF), and Cross-Encoder neural reranking.
- **Reliability & Observability:** Systematic RAG evaluation (Faithfulness, Answer Relevance, Context Precision), structured JSON logging, and comprehensive test suites (`pytest`, `pytest-asyncio`).

---

## 2. Technology Stack

| Layer | Technology | Rationale & Selection Criteria |
| :--- | :--- | :--- |
| **Language & Runtime** | Python 3.11+ | Modern typing syntax, performance optimizations, native async support. |
| **Backend Framework** | FastAPI | High throughput async ASGI framework, automated OpenAPI/Swagger docs, native Pydantic v2 integration. |
| **Vector Database** | ChromaDB | Lightweight, open-source vector store supporting persistent storage, collection partitioning, and metadata filtering. |
| **Embeddings** | `sentence-transformers` (`BAAI/bge-small-en-v1.5` / `all-MiniLM-L6-v2`) | High retrieval performance on MTEB benchmarks, local execution without third-party API latency. |
| **Reranking** | Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2` / `bge-reranker-base`) | Deep cross-attention score for candidate reranking to maximize precision@k. |
| **Sparse / Lexical Search** | BM25 (`rank-bm25`) | Exact keyword, author, and equation matching complementary to semantic dense vectors. |
| **Relational Database** | PostgreSQL 16 (via SQLAlchemy 2.0 Async + Alembic) | ACID compliance for user accounts, document metadata, audit logs, and conversation histories. |
| **Frontend Framework** | Next.js 14+ (App Router, TypeScript) | Server-side rendering, type-safe API communication, streaming response consumption. |
| **Containerization** | Docker & Docker Compose | Reproducible multi-container orchestration (FastAPI app, PostgreSQL, ChromaDB, Next.js). |
| **Testing & Quality** | `pytest`, `pytest-asyncio`, `ruff`, `mypy` | Strict static typing, lightning-fast linting, and automated unit/integration test suites. |

---

## 3. System Architecture & Information Flow

```mermaid
flowchart TD
    subgraph Client ["Frontend (Next.js + TypeScript)"]
        UI[Web Interface / PDF Viewer]
        StreamReader[SSE Streaming Client]
    end

    subgraph IngestionPipeline ["Ingestion Pipeline"]
        Upload[PDF Upload] --> Parser[PyMuPDF / PDF Plumber]
        Parser --> Cleaner[Text Normalization & Header/Footer Stripping]
        Cleaner --> Chunker[Semantic / Recursive Window Chunker]
        Chunker --> Embedder[Sentence-Transformers Embeddings]
        Embedder --> ChromaWrite[(ChromaDB Vector Store)]
        Cleaner --> PGWrite[(PostgreSQL Metadata)]
    end

    subgraph QueryPipeline ["Query & Retrieval Pipeline"]
        Query[User Question] --> QueryEmbedding[Dense Query Embedding]
        Query --> BM25Tokenize[Sparse Query Tokenization]
        QueryEmbedding --> DenseSearch[(ChromaDB Dense Search)]
        BM25Tokenize --> SparseSearch[BM25 Inverted Index]
        DenseSearch & SparseSearch --> RRF[Reciprocal Rank Fusion (RRF)]
        RRF --> TopCandidates[Top-N Candidates (e.g., N=25)]
        TopCandidates --> Reranker[Cross-Encoder Neural Reranker]
        Reranker --> TopK[Top-K Context Chunks (e.g., K=5)]
    end

    subgraph GenerationPipeline ["Grounded Generation & Streaming"]
        TopK --> PromptEngine[Citation-Enforced Prompt Engine]
        PromptEngine --> LLM[LLM Provider / Local Model]
        LLM --> StreamParser[Citation Anchor & Token Streamer]
        StreamParser -->|Server-Sent Events| StreamReader
        StreamReader --> UI
    end
```

---

## 4. Repository & Directory Structure

A clean, modular monorepo cleanly separating frontend, backend services, tests, and deployment infrastructure:

```text
cite/
├── .github/
│   └── workflows/
│       ├── lint-and-test.yml          # CI workflow for backend & frontend
│       └── docker-build.yml           # Container build verification
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── v1/
│   │   │   │   ├── endpoints/
│   │   │   │   │   ├── auth.py        # Authentication & JWT tokens
│   │   │   │   │   ├── documents.py   # Document upload, listing, deletion
│   │   │   │   │   ├── chat.py        # Q&A conversation & SSE streaming
│   │   │   │   │   ├── evaluation.py  # RAG evaluation triggers & metrics
│   │   │   │   │   └── health.py      # Liveness and readiness probes
│   │   │   │   └── router.py          # Unified v1 API router
│   │   │   └── dependencies.py        # Database sessions, current user, rate limiters
│   │   ├── core/
│   │   │   ├── config.py              # Pydantic BaseSettings (env management)
│   │   │   ├── security.py            # Password hashing, JWT encode/decode
│   │   │   ├── database.py            # Async engine, sessionmaker, Base model
│   │   │   └── logging.py             # Structured JSON logger configuration
│   │   ├── models/                    # SQLAlchemy 2.0 ORM Models
│   │   │   ├── user.py                # User credentials, roles, quotas
│   │   │   ├── document.py            # Document metadata, chunk tracking
│   │   │   └── chat.py                # Chat sessions, messages, citations
│   │   ├── schemas/                   # Pydantic v2 validation models
│   │   │   ├── user.py                # UserCreate, UserResponse, Token
│   │   │   ├── document.py            # DocumentUpload, DocumentMeta, ChunkResponse
│   │   │   ├── chat.py                # QueryRequest, ChatResponse, StreamChunk
│   │   │   └── citation.py            # CitationSource, CitationVerification
│   │   ├── services/
│   │   │   ├── ingestion/
│   │   │   │   ├── parser.py          # PDF text/metadata extraction
│   │   │   │   ├── chunker.py         # Recursive token-aware chunking with overlap
│   │   │   │   └── pipeline.py        # Ingestion coordinator (parsing -> indexing)
│   │   │   ├── retrieval/
│   │   │   │   ├── vector_store.py    # ChromaDB client wrapper & operations
│   │   │   │   ├── bm25_index.py      # BM25 sparse index builder & searcher
│   │   │   │   ├── hybrid.py          # Reciprocal Rank Fusion combiner
│   │   │   │   └── reranker.py        # Cross-Encoder model inference
│   │   │   ├── generation/
│   │   │   │   ├── llm_client.py      # LLM provider interface (OpenAI/Anthropic/Local)
│   │   │   │   ├── prompts.py         # Versioned prompt templates with strict citation rules
│   │   │   │   ├── citation_engine.py # Regex/AST citation parser & source grounding
│   │   │   │   └── streaming.py       # Async generator for Server-Sent Events (SSE)
│   │   │   └── evaluation/
│   │   │       ├── metrics.py         # Faithfulness, Answer Relevance, Precision
│   │   │       └── evaluator.py       # Automated benchmark runner on synthetic test sets
│   │   └── main.py                    # FastAPI application factory & middleware
│   ├── migrations/                    # Alembic migration scripts
│   │   ├── env.py
│   │   └── versions/
│   ├── tests/
│   │   ├── conftest.py                # Shared fixtures (async db, test client, mock embeddings)
│   │   ├── unit/
│   │   │   ├── test_chunker.py        # Boundary tests for chunking logic
│   │   │   ├── test_hybrid_search.py  # RRF and reranker scoring logic
│   │   │   └── test_citation_parser.py# Citation extraction and validation
│   │   └── integration/
│   │       ├── test_auth_api.py       # Signup, login, JWT protection
│   │       ├── test_documents_api.py  # Upload, processing, metadata endpoints
│   │       └── test_chat_stream.py    # SSE stream consumption tests
│   ├── Dockerfile
│   ├── pyproject.toml                 # Dependencies, ruff, mypy, pytest configs
│   ├── requirements.txt               # Locked dependencies
│   └── alembic.ini
├── frontend/
│   ├── src/
│   │   ├── app/                       # Next.js App Router pages
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx               # Landing / Dashboard
│   │   │   ├── login/page.tsx
│   │   │   └── workspace/[docId]/     # Main Q&A workspace with split PDF viewer
│   │   ├── components/
│   │   │   ├── chat/                  # ChatBox, MessageList, CitationPill
│   │   │   ├── pdf/                   # PDFViewer with active citation highlight
│   │   │   ├── upload/                # Drag-and-drop document uploader with progress
│   │   │   └── ui/                    # Base UI components (Button, Dialog, etc.)
│   │   ├── hooks/
│   │   │   ├── use-chat-stream.ts     # SSE stream handler hook
│   │   │   └── use-auth.ts            # Authentication state hook
│   │   ├── lib/
│   │   │   ├── api.ts                 # Type-safe Fetch wrapper
│   │   │   └── utils.ts
│   │   └── types/
│   │       └── index.ts               # Shared TypeScript schemas matching backend Pydantic
│   ├── Dockerfile
│   ├── package.json
│   ├── tsconfig.json
│   └── next.config.mjs
├── docker-compose.yml                 # Local dev stack (FastAPI, Postgres, Chroma, Next.js)
├── docker-compose.prod.yml            # Production deployment stack
├── .env.example                       # Documented environment variables template
├── .gitignore
├── README.md
└── PROJECT_CONTEXT.md                 # This specification document
```

---

## 5. Coding Standards & Engineering Conventions

To maintain a professional codebase that impresses senior engineers and hiring managers, all code must adhere to these non-negotiable standards:

### 5.1. Strict Static Typing & Pydantic
- **100% Type Hinting:** Every function signature (arguments and return types) in Python must include type annotations.
- **Python 3.11 Standard Generics:** Use `list[str]`, `dict[str, Any]`, `tuple[int, ...]`, and `str | None` instead of `typing.List`, `typing.Optional`, etc.
- **Pydantic for Data Boundaries:** Any data entering (HTTP requests, files) or leaving (API responses, Chroma metadata) must pass through a strict Pydantic `BaseModel`.
- `mypy --strict` compliance across the backend.

### 5.2. Modular Design & File Sizing
- **Single Responsibility Principle (SRP):** Each module must do exactly one thing. Keep files under 250 lines of code wherever possible.
- **Service Layer Pattern:** API routers should remain slim—handling only request unpacking, dependency resolution, service dispatch, and response wrapping. All business logic belongs in `services/`.
- **No Circular Imports:** Dependency flows strictly inward: `API -> Services -> Core/Models/Schemas`.

### 5.3. Documentation & Docstrings
- **Google-Style Docstrings:** Every public class, method, and function must have a clear docstring documenting:
  - Purpose/Behavior
  - `Args:` with descriptions
  - `Returns:` with description and structure
  - `Raises:` documenting explicit domain errors
- Self-explanatory naming over comments; comments should only explain *why*, never *what*.

### 5.4. Asynchronous Best Practices & Concurrency
- **Non-blocking Event Loop:** All I/O operations (database calls via SQLAlchemy async, HTTP requests, file reads) must be `async`/`await`.
- **CPU Offloading:** Compute-intensive operations (PDF parsing, text chunking, local sentence-transformer embeddings, cross-encoder inference) must be dispatched to `asyncio.to_thread` or a background worker threadpool to avoid blocking incoming HTTP requests.

### 5.5. Error Handling & Standardized Responses
- Define custom domain exceptions in `core/exceptions.py` (e.g., `DocumentNotFoundError`, `IngestionError`, `UnanchoredCitationError`).
- Map custom domain exceptions to standard HTTP error envelopes via FastAPI global exception handlers.
- **No bare `except:` blocks.** Always catch explicit exceptions and log tracebacks with structured context (document ID, user ID).

### 5.6. Testing Rigor
- **Framework:** `pytest` and `pytest-asyncio`.
- **Pytest Fixtures:** Use dependency injection for mock databases, in-memory ChromaDB, and mock LLM clients.
- **Coverage Target:** Minimum 85% branch coverage on ingestion, retrieval, and citation verification services.
- **Test Categories:**
  - `unit`: Fast, no external service dependencies (chunking, RRF math, prompt generation).
  - `integration`: Testing API routes with an isolated SQLite/Postgres test database and ephemeral vector store.

---

## 6. Feature Roadmap & Implementation Milestones

### Milestone 1: Document Ingestion Pipeline
- [ ] **PDF Parser Service (`services/ingestion/parser.py`):**
  - Extract text while preserving page numbers, section headers, and reading order.
  - Filter headers, footers, page numbers, and artifact characters.
  - Retain structured metadata (Paper Title, Authors, Abstract, Page Count).
- [ ] **Recursive Semantic Chunker (`services/ingestion/chunker.py`):**
  - Implement token-aware hierarchical chunking (e.g., 512 tokens with 64-token overlap).
  - Respect natural paragraph and section boundaries to avoid splitting equations or theorem definitions.
  - Attach rich metadata to every chunk: `chunk_id`, `doc_id`, `page_number`, `section_title`, `char_start`, `char_end`.
- [ ] **Batch Embedding Generation:**
  - Leverage `sentence-transformers` with batched tensor inference.
  - Run embedding generation in background threads to maintain API responsiveness.

### Milestone 2: Vector Storage & Baseline Retrieval
- [ ] **ChromaDB Integration (`services/retrieval/vector_store.py`):**
  - Initialize persistent Chroma client with HNSW indexing and cosine similarity metric.
  - Multi-tenant partitioning: Filter queries by `user_id` and selected `document_ids`.
- [ ] **Vector Retrieval API:**
  - Query embedding generation and k-NN search.
  - Score normalization and threshold filtering to reject irrelevant chunks.

### Milestone 3: Grounded Citations & Hallucination Mitigation
- [ ] **Citation-Enforcing Prompt Engineering:**
  - Design system prompts that force the model to anchor every assertion to a specific chunk UUID or citation ID `[^chunk_id]`.
  - Enforce explicit rejection behavior: *"If the provided context does not contain sufficient evidence to answer, state that the information is unavailable in the document."*
- [ ] **Citation Parsing & Verification Engine (`services/generation/citation_engine.py`):**
  - Parse inline citation markers (`[^1]`, `[^2]`) from generated text.
  - Validate that each cited chunk actually contains textual entailment for the generated claim.
  - Return rich citation payloads: page number, verbatim excerpt, and confidence score for interactive frontend highlighting.

### Milestone 4: Real-time Streaming via Server-Sent Events (SSE)
- [ ] **Streaming Generator (`services/generation/streaming.py`):**
  - Stream tokens asynchronously using SSE (`text/event-stream`).
  - Structured event types:
    - `event: status` (e.g., "Retrieving relevant passages...", "Reranking context...")
    - `event: token` (raw answer text tokens as they generate)
    - `event: citation` (grounded citation metadata objects)
    - `event: done` (latency metrics, total tokens consumed)
  - Handle client disconnection gracefully to abort generation and release GPU/CPU memory.

### Milestone 5: Authentication, Authorization & User Metadata
- [ ] **User & Document Models (PostgreSQL + SQLAlchemy 2.0 Async):**
  - `User`: ID, email, hashed_password, created_at.
  - `Document`: ID, user_id, filename, file_size, page_count, status (`PROCESSING`, `READY`, `FAILED`).
  - `ChatMessage` & `Conversation`: History persistence with citation snapshots.
- [ ] **Security & Auth Services (`core/security.py`, `api/v1/endpoints/auth.py`):**
  - Password hashing with `bcrypt` / `argon2`.
  - JWT Access & Refresh token rotation with OAuth2 Password Bearer flow.
  - Enforce strict row-level authorization: users can only access their own uploaded documents.

### Milestone 6: Advanced Retrieval — Hybrid Search & Cross-Encoder Reranking
- [ ] **BM25 Lexical Index (`services/retrieval/bm25_index.py`):**
  - Maintain an in-memory or persisted BM25 index per document/corpus for keyword precision.
- [ ] **Reciprocal Rank Fusion (RRF) (`services/retrieval/hybrid.py`):**
  - Combine ranked candidate lists from dense vector search and BM25 sparse search using standard RRF formula:
    $$RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + rank_m(d)}$$ (with constant $k \approx 60$).
- [ ] **Cross-Encoder Neural Reranking (`services/retrieval/reranker.py`):**
  - Pass the top-N (e.g., 25) fused candidate passages through a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scoring the full `(query, passage)` pair.
  - Select the top-K (e.g., 5) highest scoring passages to construct the final prompt context, maximizing context precision.

### Milestone 7: Evaluation, Observability & Guardrails
- [ ] **RAG Evaluation Suite (`services/evaluation/`):**
  - Integrate an automated evaluation harness measuring core RAG triads:
    1. **Faithfulness / Groundedness:** Are all answer claims directly entailed by the retrieved context?
    2. **Answer Relevance:** Does the answer directly address the user's prompt without topic drift?
    3. **Context Precision & Recall:** Were the most pertinent document chunks retrieved and ranked at top positions?
- [ ] **Observability & Logging (`core/logging.py`):**
  - Structured JSON logs recording retrieval latency, reranking duration, LLM generation time, and token counts.

### Milestone 8: Docker Containerization & Local Orchestration
- [ ] **Production Multi-Stage Dockerfiles:**
  - `backend/Dockerfile`: Minimal Python slim base, multi-stage build separating dependencies and runtime, non-root user execution.
  - `frontend/Dockerfile`: Multi-stage Next.js standalone output for optimal image size.
- [ ] **Docker Compose Orchestration (`docker-compose.yml`):**
  - Services: `backend`, `frontend`, `postgres`, `chromadb`.
  - Health checks on all services with proper startup dependencies (`depends_on: condition: service_healthy`).
  - Persistent volume mounts for PostgreSQL data and ChromaDB vector indices.

### Milestone 9: Production Hardening, CI/CD & Deployment
- [ ] **Continuous Integration (GitHub Actions):**
  - Automated workflow on PR: linting (`ruff`), type checking (`mypy`), and test execution (`pytest`).
- [ ] **Security & Production Hardening:**
  - CORS middleware configuration with strict origins.
  - SlowAPI rate limiting on sensitive endpoints (auth, document upload, query stream).
  - Maximum upload size limits and MIME-type validation for PDF security.
- [ ] **Deployment Blueprint:**
  - Production guide for deployment (e.g., AWS ECS/EC2, GCP Cloud Run, or Render/Railway) with environment variable separation.

---

## 7. Development Guidelines & Workflow Instructions

When implementing features in subsequent tasks:
1. **Always refer back to this document** to ensure alignment with architectural patterns, folder hierarchy, and typing requirements.
2. **Never commit untyped or untested code.** For every new service or endpoint created, corresponding unit/integration tests must be added in `tests/`.
3. **Keep changes granular and verifiable.** Run linters and test suites incrementally at each milestone.

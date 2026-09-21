<div align="center">

# CiteRAG 📄🔍

**Ask your research papers anything — get answers with page-level citations.**

[![CI](https://github.com/saikrish-code/Cite/actions/workflows/ci.yml/badge.svg)](https://github.com/saikrish-code/Cite/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-black.svg?logo=next.js)](https://nextjs.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

---

CiteRAG is a production-quality Retrieval-Augmented Generation (RAG) system built for academic literature and technical research papers. Upload PDFs (or DOCX/TXT), ask natural-language questions, and receive accurate answers with verifiable, page-level inline citations — streamed in real time. The hybrid retrieval pipeline combines dense vector search, BM25 lexical matching, Reciprocal Rank Fusion, and Cross-Encoder neural reranking to maximize answer precision.

> <!-- TODO: Add a screenshot or GIF demonstrating the main Q&A workspace with citation highlights -->

---

## Table of Contents

- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Roadmap & Known Limitations](#roadmap--known-limitations)
- [Contributing](#contributing)
- [License](#license)

---

## Key Features

- **Hybrid Retrieval Pipeline** — Dense vector search (ChromaDB) + BM25 sparse search, fused via Reciprocal Rank Fusion (RRF, k=60), then re-scored with a Cross-Encoder neural reranker
- **Grounded Citations** — Every answer assertion is anchored to a numbered citation `[1]`, `[2]`, etc., with page numbers and verbatim excerpts; the engine refuses to answer if evidence is insufficient
- **Real-Time SSE Streaming** — Token-by-token Server-Sent Events with structured event types (`token`, `citations`, `done`, `error`)
- **Multi-Format Ingestion** — Upload PDF (PyMuPDF), DOCX (python-docx), or TXT files with semantic recursive chunking (800-char chunks, 100-char overlap)
- **Multi-Provider LLM Support** — OpenAI, Anthropic Claude, Ollama (local), and Gemini — switchable via a single env var
- **3-Way Retrieval Modes** — Toggle between `vector_only`, `hybrid`, and `hybrid_rerank` via environment variable or API
- **JWT Authentication** — OAuth2 password flow with bcrypt hashing and row-level document isolation per user
- **Chat History Persistence** — Conversations and citation snapshots stored in PostgreSQL
- **Docker Orchestration** — Full-stack `docker-compose.yml` with PostgreSQL, ChromaDB, backend, and frontend services
- **Evaluation Harness** — Benchmarking suite measuring Faithfulness, Answer Relevance, Context Precision, and Citation Entailment

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Backend** | Python 3.11+, FastAPI, Pydantic v2, SQLAlchemy 2.0 (async), Alembic |
| **Frontend** | Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS 4 |
| **Vector DB** | ChromaDB (HNSW + cosine similarity) |
| **Embeddings** | Sentence-Transformers (`BAAI/bge-small-en-v1.5`) |
| **Reranker** | Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| **Sparse Search** | BM25 via `rank-bm25` |
| **Relational DB** | PostgreSQL 15 (via `asyncpg`) |
| **LLM Clients** | `openai`, `anthropic` Python SDKs; Ollama HTTP |
| **Auth** | JWT (PyJWT) + bcrypt (passlib) |
| **Containerization** | Docker multi-stage builds, Docker Compose |
| **Quality** | Ruff, Black, mypy (strict), pytest, pytest-asyncio |

---

## Quick Start

Get a working local instance in three steps:

### 1. Start infrastructure services

```bash
docker compose up -d postgres chroma
```

### 2. Run the backend

```bash
# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env — at minimum, set a valid LLM API key (see Configuration below)

# Start the FastAPI server
cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to use the app. The backend API docs are at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Prerequisites

| Requirement | Version |
| :--- | :--- |
| Python | 3.11+ |
| Node.js | 20+ |
| PostgreSQL | 15+ |
| Docker & Docker Compose | Latest (optional, for containerized setup) |

PostgreSQL and ChromaDB can be run via Docker Compose (see [Quick Start](#quick-start)) or installed natively.

---

## Installation

### Option A: Local Development (recommended for development)

```bash
git clone https://github.com/saikrish-code/Cite.git
cd Cite

# Backend
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r backend/requirements.txt

# Frontend
cd frontend && npm install && cd ..

# Start Postgres + ChromaDB
docker compose up -d postgres chroma

# Configure environment
cp .env.example .env
# Edit .env with your API keys and database credentials

# Run database migrations
cd backend && alembic upgrade head && cd ..
```

### Option B: Full Docker Stack

```bash
git clone https://github.com/saikrish-code/Cite.git
cd Cite
cp .env.example .env
# Edit .env with your API keys

docker compose up --build
```

This starts all four services: `postgres` (:5432), `chroma` (:8001), `backend` (:8000), and `frontend` (:3000).

---

## Configuration

Copy `.env.example` to `.env` and configure the variables below. The backend loads settings via Pydantic `BaseSettings` from `backend/app/core/config.py`.

### General Settings

| Variable | Description | Default | Required |
| :--- | :--- | :--- | :--- |
| `PROJECT_NAME` | Application display name | `CiteRAG - Research Paper Assistant` | No |
| `ENVIRONMENT` | Runtime environment (`development` / `production`) | `development` | No |
| `DEBUG` | Enable debug mode | `true` | No |
| `API_V1_STR` | API version prefix | `/api/v1` | No |
| `BACKEND_HOST` | Server bind host | `0.0.0.0` | No |
| `BACKEND_PORT` | Server bind port | `8000` | No |
| `CORS_ORIGINS` | Comma-separated allowed origins | `http://localhost:3000,http://127.0.0.1:3000` | No |

### Security & Auth

| Variable | Description | Default | Required |
| :--- | :--- | :--- | :--- |
| `JWT_SECRET_KEY` | Secret for signing JWT tokens | Insecure dev default | **Yes** (production) |
| `JWT_ALGORITHM` | JWT signing algorithm | `HS256` | No |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token expiration in minutes | `60` | No |

### Database (PostgreSQL)

| Variable | Description | Default | Required |
| :--- | :--- | :--- | :--- |
| `DATABASE_URL` | Async PostgreSQL connection string | `postgresql+asyncpg://postgres:postgrespassword@localhost:5432/citerag` | **Yes** |
| `POSTGRES_USER` | PostgreSQL username | `postgres` | No |
| `POSTGRES_PASSWORD` | PostgreSQL password | `postgrespassword` | No |
| `POSTGRES_DB` | Database name | `citerag` | No |

### ChromaDB

| Variable | Description | Default | Required |
| :--- | :--- | :--- | :--- |
| `CHROMA_PERSIST_DIRECTORY` | Local fallback storage path | `./chroma_data` | No |
| `CHROMA_COLLECTION_NAME` | Vector collection name | `citerag_chunks` | No |
| `CHROMA_SERVER_HOST` | ChromaDB server hostname | `localhost` | No |
| `CHROMA_SERVER_PORT` | ChromaDB server port | `8001` | No |

### Embeddings & Reranking

| Variable | Description | Default | Required |
| :--- | :--- | :--- | :--- |
| `EMBEDDING_MODEL_NAME` | Sentence-Transformers model for embeddings | `BAAI/bge-small-en-v1.5` | No |
| `RERANKER_MODEL_NAME` | Cross-Encoder model for reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | No |
| `RETRIEVAL_MODE` | Retrieval strategy: `vector_only`, `hybrid`, or `hybrid_rerank` | `hybrid_rerank` | No |

### LLM Provider

| Variable | Description | Default | Required |
| :--- | :--- | :--- | :--- |
| `LLM_PROVIDER` | Active provider: `openai`, `anthropic`, `ollama`, or `mock` | `openai` | No |
| `LLM_TEMPERATURE` | Generation temperature | `0.0` | No |
| `LLM_MAX_TOKENS` | Max generation tokens | `1024` | No |
| `OPENAI_API_KEY` | OpenAI (or compatible) API key | — | **Yes** if provider is `openai` |
| `OPENAI_MODEL` | OpenAI model name | `gpt-4o-mini` | No |
| `OPENAI_BASE_URL` | Custom base URL (e.g., for Groq compatibility) | — | No |
| `ANTHROPIC_API_KEY` | Anthropic API key | — | **Yes** if provider is `anthropic` |
| `ANTHROPIC_MODEL` | Anthropic model name | `claude-3-5-sonnet-20241022` | No |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` | No |
| `OLLAMA_MODEL` | Ollama model name | `llama3.2` | No |
| `GEMINI_API_KEY` | Google Gemini API key | — | No |

### Frontend

| Variable | Description | Default | Required |
| :--- | :--- | :--- | :--- |
| `NEXT_PUBLIC_API_URL` | Backend API base URL exposed to the browser | `http://localhost:8000/api/v1` | **Yes** |

---

## Usage

### 1. Register and log in

Create an account via the web UI or the API:

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "securepassword"}'
```

### 2. Upload a document

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer <your-jwt-token>" \
  -F "file=@research_paper.pdf"
```

Supported formats: `.pdf`, `.docx`, `.txt` (max 50 MB).

### 3. Ask questions with citations

```bash
curl -N http://localhost:8000/api/v1/chat/completions \
  -H "Authorization: Bearer <your-jwt-token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "What methodology did the authors use?", "document_ids": ["<doc-id>"]}'
```

The response streams SSE events: `token` (answer text), `citations` (source metadata with page numbers), and `done` (latency metrics).

---

## API Reference

All endpoints are prefixed with `/api/v1`. Interactive Swagger docs are available at `/docs`.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Liveness check (status, version, environment, timestamp) |
| `POST` | `/api/v1/auth/register` | Register a new user account |
| `POST` | `/api/v1/auth/token` | OAuth2 password login → JWT access token |
| `GET` | `/api/v1/auth/me` | Get current authenticated user |
| `POST` | `/api/v1/documents/upload` | Upload and ingest a document (PDF/DOCX/TXT) |
| `GET` | `/api/v1/documents` | List the authenticated user's documents |
| `DELETE` | `/api/v1/documents/{doc_id}` | Delete a document and its vector embeddings |
| `POST` | `/api/v1/chat/completions` | Query documents via SSE-streamed Q&A with citations |

See [docs/api.md](docs/api.md) for detailed request/response schemas.

---

## Project Structure

```text
Cite/
├── backend/
│   ├── app/
│   │   ├── api/v1/endpoints/       # Route handlers (auth, chat, documents, health)
│   │   ├── core/                   # Config, database, security modules
│   │   ├── models/                 # SQLAlchemy ORM models (user, document, chat)
│   │   ├── schemas/                # Pydantic v2 request/response schemas
│   │   ├── services/               # Business logic (see below)
│   │   └── main.py                 # FastAPI app factory & middleware
│   ├── migrations/                 # Alembic database migration scripts
│   ├── tests/
│   │   ├── unit/                   # Chunking, BM25, RRF, reranker, RAG tests
│   │   └── integration/            # API endpoint & user isolation tests
│   ├── Dockerfile                  # Multi-stage Python 3.11-slim image
│   ├── pyproject.toml              # Project metadata, ruff/black/mypy/pytest config
│   └── requirements.txt            # Pinned Python dependencies
├── frontend/
│   ├── src/
│   │   ├── app/                    # Next.js App Router pages & API routes
│   │   ├── components/             # React components (auth, chat, citations, sidebar)
│   │   ├── context/                # Auth context provider
│   │   ├── lib/                    # Type-safe API client
│   │   └── utils/                  # Supabase client utilities
│   ├── Dockerfile                  # Multi-stage Next.js standalone image
│   └── package.json                # Node dependencies & scripts
├── eval/
│   ├── datasets/                   # Ground-truth Q&A pairs & test sets
│   └── benchmarks/                 # Automated RAG evaluation runners
├── docs/
│   ├── architecture.md             # System architecture overview
│   ├── api.md                      # API endpoint reference
│   └── decisions.md                # Engineering decision records
├── docker-compose.yml              # Local dev stack (4 services)
├── Makefile                        # Dev automation (run, test, lint, format)
├── .env.example                    # Environment variable template
└── .github/workflows/ci.yml       # CI: lint, type-check, test, build
```

### Backend Services (`backend/app/services/`)

| Module | Responsibility |
| :--- | :--- |
| `parsers.py` | PDF/DOCX/TXT text extraction with page number preservation |
| `chunker.py` | Recursive token-aware text chunking with configurable overlap |
| `embeddings.py` | Batch sentence-transformer embedding generation |
| `vector_store.py` | ChromaDB client wrapper with multi-tenant filtering |
| `bm25.py` | BM25 Okapi sparse index builder and searcher |
| `hybrid_retriever.py` | RRF-based dense + sparse fusion and retrieval orchestration |
| `reranker.py` | Cross-Encoder neural passage reranking |
| `llm.py` | Multi-provider LLM client (OpenAI/Anthropic/Ollama) |
| `rag.py` | End-to-end RAG pipeline: retrieval → prompt → generation → citation parsing |
| `ingestion.py` | Ingestion coordinator (parse → chunk → embed → index) |
| `intent.py` | Query intent classification |
| `document_store.py` | Document metadata CRUD operations |

---

## Architecture

```mermaid
flowchart TD
    subgraph Client ["Frontend (Next.js + TypeScript)"]
        UI[Web Interface]
        SSE[SSE Stream Consumer]
    end

    subgraph Ingestion ["Ingestion Pipeline"]
        Upload[PDF / DOCX / TXT Upload]
        Upload --> Parser[PyMuPDF / python-docx Parser]
        Parser --> Chunker[Recursive Semantic Chunker]
        Chunker --> Embedder[Sentence-Transformers Embeddings]
        Embedder --> ChromaDB[(ChromaDB Vector Store)]
        Parser --> PG[(PostgreSQL Metadata)]
    end

    subgraph Retrieval ["Hybrid Retrieval Pipeline"]
        Query[User Question]
        Query --> DenseEmbed[Dense Query Embedding]
        Query --> BM25[BM25 Sparse Tokenization]
        DenseEmbed --> VectorSearch[(ChromaDB k-NN Search)]
        BM25 --> SparseSearch[BM25 Index Search]
        VectorSearch & SparseSearch --> RRF[Reciprocal Rank Fusion]
        RRF --> Reranker[Cross-Encoder Reranker]
        Reranker --> TopK[Top-K Context Chunks]
    end

    subgraph Generation ["Grounded Generation"]
        TopK --> Prompt[Citation-Enforced Prompt]
        Prompt --> LLM[LLM Provider]
        LLM --> CitationParser[Citation Extraction & Verification]
        CitationParser -->|SSE Stream| SSE
        SSE --> UI
    end
```

---

## Testing

The test suite uses `pytest` and `pytest-asyncio` with fixtures defined in `backend/tests/conftest.py`.

### Run all tests

```bash
make test
```

Or directly:

```bash
cd backend && pytest tests app/tests -v
```

### Run with coverage

```bash
make test-cov
```

### Linting & formatting

```bash
# Check for issues
make lint

# Auto-fix formatting
make format
```

### CI Pipeline

The [CI workflow](.github/workflows/ci.yml) runs on every push/PR to `main`:
- **Backend:** `ruff check` → `mypy app` → `pytest`
- **Frontend:** `npm ci` → `eslint` → `next build`

---

## Troubleshooting

| Problem | Cause | Solution |
| :--- | :--- | :--- |
| `ConnectionRefusedError` on port 5432 | PostgreSQL is not running | Run `docker compose up -d postgres` or start your local PostgreSQL service |
| `ConnectionRefusedError` on port 8001 | ChromaDB is not running | Run `docker compose up -d chroma` |
| `OPENAI_API_KEY` error on startup | Missing or empty API key | Set a valid key in `.env` or switch `LLM_PROVIDER` to `ollama` or `mock` |
| `ModuleNotFoundError: torchcodec` | Torch codec compatibility issue on Windows | Already handled in `main.py` — ensure you're using the project's entry point |
| Embedding model download hangs | First-run model download from Hugging Face Hub | Wait for the ~90 MB `BAAI/bge-small-en-v1.5` model to download; subsequent starts are instant |
| Upload rejected | File exceeds 50 MB or unsupported extension | Supported: `.pdf`, `.docx`, `.txt`. Max size: 50 MB (configurable via `MAX_UPLOAD_SIZE_MB`) |
| `alembic upgrade head` fails | `DATABASE_URL` not set or database unreachable | Verify `DATABASE_URL` in `.env` and ensure PostgreSQL is running |
| Frontend can't reach backend | Incorrect `NEXT_PUBLIC_API_URL` | Set to `http://localhost:8000/api/v1` in `frontend/.env.local` |

---

## Roadmap & Known Limitations

Based on milestones tracked in [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md):

### Completed ✅
- PDF/DOCX/TXT ingestion with page-number-aware chunking
- ChromaDB vector storage with multi-tenant partitioning
- Citation-enforced prompt engineering and inline citation parsing
- SSE streaming with structured event types
- JWT authentication and row-level user isolation
- Hybrid search (BM25 + Dense + RRF) with Cross-Encoder reranking
- Multi-stage latency profiling and 3-way retrieval mode switching

### In Progress / Planned 🚧
- [ ] RAG evaluation suite — automated Faithfulness, Answer Relevance, and Context Precision metrics (`eval/`)
- [ ] Structured JSON logging and observability
- [ ] Production-hardened Docker Compose configuration (`docker-compose.prod.yml`)
- [ ] CI/CD pipeline expansion — automated Docker builds, deployment blueprints
- [ ] Rate limiting on sensitive endpoints (auth, upload, chat)
- [ ] Sentence-window and semantic chunking strategies (noted in [docs/decisions.md](docs/decisions.md))

---

## Contributing

Contributions are welcome. To get started:

1. Fork and clone the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Set up the development environment (see [Installation](#installation))
4. Follow the coding standards: strict type hints, Google-style docstrings, and files under 250 lines
5. Add tests for any new services or endpoints
6. Run quality checks before committing:
   ```bash
   make lint
   make test
   ```
7. Open a pull request against `main`

<!-- TODO: Add a CONTRIBUTING.md with detailed branching conventions and code review process -->

---

## License

<!-- TODO: Add a LICENSE file to the repository and update this section -->

No license file is currently present in this repository. Please add a `LICENSE` file to clarify usage terms.

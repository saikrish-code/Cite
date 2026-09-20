# CiteRAG 📄🔍

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg?logo=next.js)](https://nextjs.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-quality Retrieval-Augmented Generation (RAG) assistant tailored for academic literature and technical research papers. Built with FastAPI, ChromaDB, Sentence-Transformers, Next.js, and PostgreSQL.

---

## Repository Structure

```text
cite/
├── backend/       # FastAPI async backend (APIs, core config, retrieval, ingestion)
├── frontend/      # Next.js 14+ App Router, TypeScript & Tailwind CSS
├── eval/          # RAG benchmarking and evaluation datasets & metrics
├── docs/          # Architecture design, system diagrams, and API specifications
├── Makefile       # Automation commands for testing, linting, formatting, and running
├── .env.example   # Environment variable configuration template
└── PROJECT_CONTEXT.md # Master architectural reference and feature roadmap
```

---

## Quick Start

### 1. Backend Setup

```bash
# Using uv (recommended)
uv venv .venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r backend/requirements.txt

# Or using standard pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

Run the backend development server:
```bash
make run-backend
# Or directly:
cd backend && uvicorn app.main:app --reload --port 8000
```
Open [http://localhost:8000/health](http://localhost:8000/health) or [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger documentation.

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to view the client app.

---

## Testing & Code Quality

Run tests:
```bash
make test
```

Run linter & code style checks:
```bash
make lint
make format
```

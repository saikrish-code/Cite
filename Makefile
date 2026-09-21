.PHONY: help install run run-backend run-frontend test test-cov lint format clean

PYTHON ?= python
UVICORN ?= uvicorn
PYTEST ?= pytest
RUFF ?= ruff
BLACK ?= black

help:
	@echo "CiteRAG Monorepo Management Commands:"
	@echo "  make install        - Install backend dependencies"
	@echo "  make run            - Run FastAPI backend server"
	@echo "  make run-backend    - Run FastAPI backend server"
	@echo "  make run-frontend   - Run Next.js frontend dev server"
	@echo "  make test           - Run pytest test suite"
	@echo "  make test-cov       - Run pytest with coverage report"
	@echo "  make lint           - Check code formatting & linting with ruff and black"
	@echo "  make format         - Auto-format code with ruff and black"
	@echo "  make clean          - Remove temporary files and test caches"
	@echo "  make eval           - Run the RAG evaluation suite"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r backend/requirements.txt

eval:
	cd eval/benchmarks && $(PYTHON) evaluator.py

run: run-backend

run-backend:
	cd backend && $(UVICORN) app.main:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	cd frontend && npm run dev

test:
	cd backend && $(PYTEST) tests app/tests -v

test-cov:
	cd backend && $(PYTEST) tests app/tests -v --cov=app --cov-report=term-missing

lint:
	cd backend && $(RUFF) check .
	cd backend && $(BLACK) --check .

format:
	cd backend && $(RUFF) check --fix .
	cd backend && $(BLACK) .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

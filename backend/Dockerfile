FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck and runtime system deps (libpq5 for postgres)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder and install
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Add non-root user
RUN addgroup --system citerag && adduser --system --group citerag

# Copy application code
COPY . .

# Create directory for vector store if it falls back to local
RUN mkdir -p /app/chroma_data && chown -R citerag:citerag /app/chroma_data

# Set ownership
RUN chown -R citerag:citerag /app

# Set Hugging Face cache to a writable directory
ENV HF_HOME=/tmp/huggingface

USER citerag

# Expose the API port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

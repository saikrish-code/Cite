# API Reference

CiteRAG exposes RESTful and streaming endpoints adhering to OpenAPI 3.1 standards.

## Endpoints

### 1. System Health
- **`GET /health`** (and **`GET /api/v1/health`**):
  - Returns service status, version, environment, and server timestamp.
  - Response format:
    ```json
    {
      "status": "ok",
      "environment": "development",
      "version": "0.1.0",
      "timestamp": "2026-09-19T22:30:00.000000Z"
    }
    ```

### 2. Authentication (`/api/v1/auth`)
- `POST /api/v1/auth/register`: Register new user account.
- `POST /api/v1/auth/token`: OAuth2 password flow returning JWT access & refresh tokens.
- `GET /api/v1/auth/me`: Fetch current authenticated user.

### 3. Document Management (`/api/v1/documents`)
- `POST /api/v1/documents/upload`: Upload PDF paper for ingestion & indexing.
- `GET /api/v1/documents`: List user's indexed documents.
- `DELETE /api/v1/documents/{doc_id}`: Remove document and clean up vector collections.

### 4. Interactive Q&A (`/api/v1/chat`)
- `POST /api/v1/chat/completions`: Query document corpus with real-time SSE token & citation streaming.

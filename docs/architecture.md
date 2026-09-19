# System Architecture

CiteRAG is composed of four principal subsystems:

1. **Ingestion Engine:**
   - PDF parsing preserving reading order, headers, and page numbers.
   - Recursive semantic chunking with sliding-window token overlap.
   - Batch vector embeddings generation via `sentence-transformers`.

2. **Hybrid Retrieval Layer:**
   - Dense vector similarity search via ChromaDB.
   - Sparse lexical keyword retrieval via BM25.
   - Candidate list aggregation via Reciprocal Rank Fusion (RRF).
   - Re-ranking of top candidate passages via Cross-Encoder neural scoring.

3. **Grounded Generation & Streaming Engine:**
   - Strict citation-enforced system prompts.
   - Token-by-token streaming via Server-Sent Events (SSE).
   - Real-time inline citation extraction and textual entailment validation.

4. **Persistence & Access Control:**
   - PostgreSQL 16 for user profiles, document ownership, and chat histories.
   - Row-level access boundaries ensuring multi-tenant isolation.

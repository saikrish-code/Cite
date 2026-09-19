"""End-to-end demonstration script: PDF Ingestion, Embedding, Storage, and Retrieval.

This script demonstrates the complete CiteRAG pipeline:
1. Ingests a sample academic PDF using IngestionService (PyMuPDF parser + RecursiveTokenChunker)
2. Extracts page-level text and generates retrieval-ready chunks
3. Computes dense embeddings using SentenceTransformerEmbeddingService (BAAI/bge-small-en-v1.5)
4. Indexes chunks and citation metadata (user_id, document_id, filename, page, chunk_index) in ChromaVectorStore
5. Executes a semantic similarity search query and prints ranked results with citation provenance.

Usage:
    python backend/scripts/ingest_and_query.py [--pdf PATH] [--query QUERY] [--k K]
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time

# Ensure backend root is on sys.path
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings  # noqa: E402
from app.services.embeddings import SentenceTransformerEmbeddingService  # noqa: E402
from app.services.ingestion import IngestionService  # noqa: E402
from app.services.vector_store import ChromaVectorStore  # noqa: E402


def setup_logging() -> None:
    """Configure console logging format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def run_pipeline(
    pdf_path: pathlib.Path,
    query: str,
    k: int = 3,
    user_id: str = "researcher_demo",
    document_id: str = "doc_attention_sample",
    persist_dir: str | None = None,
) -> None:
    """Execute end-to-end ingestion, indexing, and retrieval.

    Args:
        pdf_path: Path to the target PDF file.
        query: Natural language query to execute.
        k: Number of matching chunks to retrieve.
        user_id: Owner user ID.
        document_id: Unique document identifier.
        persist_dir: Filesystem directory for Chroma data, or ':memory:'.
    """
    print("=" * 80)
    print(" CiteRAG: End-to-End Ingestion, Embedding & Vector Retrieval Demo")
    print("=" * 80)

    if not pdf_path.is_file():
        print(f"Error: PDF file not found at '{pdf_path}'")
        sys.exit(1)

    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # 1. Initialize Ingestion Service
    print(f"\n[Step 1/4] Reading and Ingesting PDF: {pdf_path.name}")
    start_time = time.perf_counter()
    file_bytes = pdf_path.read_bytes()

    ingestion_service = IngestionService()
    parsed_doc = ingestion_service.parse(file_bytes, filename=pdf_path.name)
    chunks = ingestion_service.chunk(parsed_doc)
    parse_duration = (time.perf_counter() - start_time) * 1000

    print(
        f"  [OK] Document parsed: {parsed_doc.total_pages} pages in {parse_duration:.1f} ms"
    )
    print(
        f"  [OK] Generated {len(chunks)} text chunks (strategy: RecursiveTokenChunker)"
    )
    for i, c in enumerate(chunks[:2]):
        print(f"    - Chunk #{i}: Page {c.metadata.page_number} ({len(c.text)} chars)")
    if len(chunks) > 2:
        print(f"    - ... and {len(chunks) - 2} more chunks")

    # 2. Initialize Embedding Service
    print(
        f"\n[Step 2/4] Initializing Embedding Service: {settings.EMBEDDING_MODEL_NAME}"
    )
    embed_start = time.perf_counter()
    embedder = SentenceTransformerEmbeddingService()
    dim = embedder.dimension
    embed_init_time = (time.perf_counter() - embed_start) * 1000
    print(
        f"  [OK] Embedding model loaded: {embedder.model_name} ({dim}-dimensional vectors) in {embed_init_time:.1f} ms"
    )

    # 3. Index into Chroma Vector Store
    target_dir = persist_dir or ":memory:"
    print(
        f"\n[Step 3/4] Indexing {len(chunks)} chunks into ChromaDB (storage: '{target_dir}')"
    )
    vector_store = ChromaVectorStore(
        embedding_service=embedder,
        persist_directory=target_dir,
        collection_name="demo_academic_papers",
    )

    index_start = time.perf_counter()
    chunk_ids = vector_store.add_chunks(
        chunks=chunks,
        document_id=document_id,
        user_id=user_id,
    )
    index_time = (time.perf_counter() - index_start) * 1000

    print(f"  [OK] Successfully indexed {len(chunk_ids)} chunks in {index_time:.1f} ms")
    print("  [OK] Metadata stored for each chunk:")
    print(f"      - user_id:     {user_id}")
    print(f"      - document_id: {document_id}")
    print(f"      - filename:    {pdf_path.name}")
    print(f"      - page:        [1..{parsed_doc.total_pages}]")
    print(f"      - chunk_index: [0..{len(chunks) - 1}]")

    # 4. Perform Similarity Search
    print("\n[Step 4/4] Executing Semantic Search Query")
    print(f'  Query: "{query}"')
    print(f"  Filter: {{'user_id': '{user_id}', 'document_id': '{document_id}'}}")
    print(f"  Top-K: {k}")

    query_start = time.perf_counter()
    results = vector_store.similarity_search(
        query=query,
        k=k,
        filters={"user_id": user_id, "document_id": document_id},
    )
    query_time = (time.perf_counter() - query_start) * 1000

    print(
        f"\n  Search completed in {query_time:.1f} ms. Retrieved {len(results)} matches:"
    )
    print("-" * 80)

    for rank, result in enumerate(results, start=1):
        meta = result.metadata
        print(
            f"\n  [Rank {rank}] Score: {result.score:.4f} | Citation: [{meta.filename} p.{meta.page} chunk #{meta.chunk_index}]"
        )
        print(f"  Chunk ID: {result.chunk_id}")
        # Display clean preview snippet
        snippet = result.text.strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        print(f'  Snippet: "{snippet}"')

    # 5. Synthesize Grounded RAG Answer
    print("\n[Step 5/5] Synthesizing Grounded Answer with Citations via RAGService")
    from app.services.llm import MockLLMClient, get_llm_client
    from app.services.rag import RAGService

    # Check if a live API key is configured; otherwise use realistic MockLLMClient
    if settings.LLM_PROVIDER.lower() == "openai" and settings.OPENAI_API_KEY:
        llm = get_llm_client()
    elif (
        settings.LLM_PROVIDER.lower() in ("anthropic", "claude")
        and settings.ANTHROPIC_API_KEY
    ):
        llm = get_llm_client()
    elif settings.LLM_PROVIDER.lower() == "ollama":
        llm = get_llm_client()
    else:
        llm = MockLLMClient(
            default_response=(
                "The Transformer model relies entirely on self-attention mechanisms [1]. "
                "Recurrent models typically factor computation along symbol positions [2], "
                "whereas the Transformer dispenses with recurrence to achieve parallelization [1]."
            ),
            model="mock-gpt4-citations",
        )

    rag_service = RAGService(vector_store=vector_store, llm_client=llm)
    rag_start = time.perf_counter()
    rag_resp = rag_service.answer(
        query=query,
        k=k,
        document_id=document_id,
        user_id=user_id,
    )
    rag_duration = (time.perf_counter() - rag_start) * 1000

    print(
        f"  [OK] Answer generated in {rag_duration:.1f} ms using LLM model '{llm.model_name}'"
    )
    print(f"  Context Found: {rag_resp.context_found}")
    print("\n  Answer:")
    print(f"    {rag_resp.answer}")
    print("\n  Resolved Citations:")
    for cit in rag_resp.citations:
        print(
            f"    - [{cit.source_id}] {cit.filename} (Page {cit.page}) | Score: {cit.score:.4f}"
        )
        snip = cit.snippet.replace("\n", " ")
        if len(snip) > 120:
            snip = snip[:120] + "..."
        print(f'      Snippet: "{snip}"')

    print("\n" + "=" * 80)
    print(" Pipeline completed successfully!")
    print("=" * 80 + "\n")


def main() -> None:
    """CLI entrypoint."""
    setup_logging()

    default_pdf = BACKEND_DIR / "tests" / "fixtures" / "sample.pdf"
    default_query = "What is the Transformer model and how does attention work?"

    parser = argparse.ArgumentParser(
        description="CiteRAG: Ingest sample PDF, index into ChromaDB, and run a test query."
    )
    parser.add_argument(
        "--pdf",
        type=pathlib.Path,
        default=default_pdf,
        help=f"Path to PDF file (default: {default_pdf})",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=default_query,
        help=f"Search query (default: '{default_query}')",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of top chunks to retrieve (default: 3)",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=":memory:",
        help="Chroma storage directory or ':memory:' (default: ':memory:')",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="researcher_demo",
        help="User identifier for metadata multi-tenancy (default: 'researcher_demo')",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default="doc_attention_sample",
        help="Document identifier (default: 'doc_attention_sample')",
    )

    args = parser.parse_args()
    run_pipeline(
        pdf_path=args.pdf,
        query=args.query,
        k=args.k,
        user_id=args.user_id,
        document_id=args.doc_id,
        persist_dir=args.persist_dir,
    )


if __name__ == "__main__":
    main()

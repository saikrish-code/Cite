"""Retrieval-Augmented Generation (RAG) question answering service.

Orchestrates:
1. Vector retrieval of top-k context passages from BaseVectorStore
2. Prompt construction enforcing strict context grounding and [1], [2] citations
3. Answer generation via BaseLLMClient (OpenAI, Claude, Ollama, or Mock)
4. Inline citation parsing and structured provenance resolution into Citation schemas
5. "I don't know" handling when context is missing or insufficient
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.schemas.chat import Citation, RAGResponse
from app.schemas.document import VectorSearchResult
from app.services.llm import BaseLLMClient, get_llm_client
from app.services.vector_store import BaseVectorStore, ChromaVectorStore

logger = logging.getLogger(__name__)

STRICT_RAG_SYSTEM_PROMPT = """You are a rigorous, factual academic research assistant.
Your task is to answer the user's question relying EXCLUSIVELY on the provided numbered context passages.

Strict rules:
1. Answer ONLY using facts directly stated in the context passages below. Do NOT use external knowledge, extrapolate, or speculate.
2. For every factual claim made in your response, cite the supporting context passage using its bracketed number, e.g., [1] or [1][2].
3. If the provided context does not contain sufficient information to answer the question, state: "I don't know based on the provided context." Do not invent or guess an answer.
4. Keep the answer direct, objective, and well-structured."""

UNKNOWN_PHRASES = (
    "i don't know",
    "i do not know",
    "insufficient context",
    "not mentioned in the context",
    "not provided in the context",
    "cannot answer based on the provided context",
)


class RAGService:
    """End-to-end RAG answer service with citation enforcement and source grounding.

    Args:
        vector_store: Vector store used for semantic similarity search. Defaults to ChromaVectorStore.
        llm_client: LLM client used for grounded answer synthesis. Defaults to get_llm_client().
        system_prompt: Custom system prompt instructions. Defaults to STRICT_RAG_SYSTEM_PROMPT.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore | None = None,
        llm_client: BaseLLMClient | None = None,
        system_prompt: str = STRICT_RAG_SYSTEM_PROMPT,
    ) -> None:
        self.vector_store = vector_store or ChromaVectorStore()
        self.llm_client = llm_client or get_llm_client()
        self.system_prompt = system_prompt

    def _build_context_and_prompt(
        self, query: str, search_results: list[VectorSearchResult]
    ) -> tuple[str, dict[int, VectorSearchResult]]:
        """Construct prompt context blocks and map 1-indexed source IDs to results.

        Args:
            query: User search query.
            search_results: Retrieved chunks from vector search.

        Returns:
            tuple[str, dict[int, VectorSearchResult]]:
                - Formatted user prompt containing question and numbered context blocks.
                - Mapping of source_id (1, 2, ...) to original VectorSearchResult.
        """
        source_map: dict[int, VectorSearchResult] = {}
        context_blocks: list[str] = []

        for idx, result in enumerate(search_results, start=1):
            source_map[idx] = result
            filename = result.metadata.filename
            page = result.metadata.page
            clean_text = result.text.strip()
            block = f"[{idx}] (Source: {filename}, Page: {page})\n{clean_text}"
            context_blocks.append(block)

        joined_context = "\n\n".join(context_blocks)
        user_prompt = (
            f"Question:\n{query}\n\n"
            f"Context Passages:\n{joined_context}\n\n"
            "Instructions:\n"
            "Answer the question above using ONLY the facts from the Context Passages. "
            "Cite each claim using its passage number (e.g. [1], [2]). "
            "If the context does not contain enough information to answer, state: "
            '"I don\'t know based on the provided context."'
        )
        return user_prompt, source_map

    def _parse_citations(
        self,
        raw_answer: str,
        source_map: dict[int, VectorSearchResult],
        snippet_length: int = 250,
    ) -> list[Citation]:
        """Extract inline citations like [1], [2] from text and map to source metadata.

        Args:
            raw_answer: Model output text.
            source_map: Mapping of passage IDs to VectorSearchResults.
            snippet_length: Maximum characters for the citation snippet excerpt.

        Returns:
            list[Citation]: Resolved structured citations.
        """
        cited_indices = [int(match) for match in re.findall(r"\[(\d+)\]", raw_answer)]
        # Preserve first occurrence order
        unique_indices: list[int] = list(dict.fromkeys(cited_indices))

        citations: list[Citation] = []
        for src_id in unique_indices:
            if src_id in source_map:
                res = source_map[src_id]
                snippet = res.text.strip()
                if len(snippet) > snippet_length:
                    snippet = snippet[:snippet_length] + "..."

                citations.append(
                    Citation(
                        source_id=src_id,
                        filename=res.metadata.filename,
                        page=res.metadata.page,
                        snippet=snippet,
                        score=res.score,
                    )
                )

        return citations

    def _resolve_filters(
        self,
        filters: dict[str, Any] | None,
        document_id: str | None,
        user_id: str | None,
    ) -> dict[str, Any] | None:
        """Merge explicit document_id and user_id into filters mapping."""
        resolved: dict[str, Any] = dict(filters or {})
        if document_id:
            resolved["document_id"] = document_id
        if user_id:
            resolved["user_id"] = user_id
        return resolved or None

    def answer(
        self,
        query: str,
        k: int = 4,
        filters: dict[str, Any] | None = None,
        document_id: str | None = None,
        user_id: str | None = None,
    ) -> RAGResponse:
        """Synchronously retrieve context, prompt LLM, and return structured grounded response.

        Args:
            query: Question to answer.
            k: Number of context passages to retrieve.
            filters: Optional metadata filters.
            document_id: Optional document scope filter.
            user_id: Optional user multi-tenancy filter.

        Returns:
            RAGResponse: Structured answer with resolved citations.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return RAGResponse(
                query=query,
                answer="I don't know based on the provided context.",
                citations=[],
                context_found=False,
            )

        merged_filters = self._resolve_filters(filters, document_id, user_id)
        search_results = self.vector_store.similarity_search(
            query=cleaned_query,
            k=k,
            filters=merged_filters,
        )

        if not search_results:
            return RAGResponse(
                query=cleaned_query,
                answer="I don't know based on the provided context.",
                citations=[],
                context_found=False,
            )

        user_prompt, source_map = self._build_context_and_prompt(
            cleaned_query, search_results
        )

        raw_answer = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=self.system_prompt,
        ).strip()

        is_unknown = any(phrase in raw_answer.lower() for phrase in UNKNOWN_PHRASES)
        if is_unknown:
            return RAGResponse(
                query=cleaned_query,
                answer=raw_answer,
                citations=[],
                context_found=False,
            )

        citations = self._parse_citations(raw_answer, source_map)

        return RAGResponse(
            query=cleaned_query,
            answer=raw_answer,
            citations=citations,
            context_found=True,
        )

    async def aanswer(
        self,
        query: str,
        k: int = 4,
        filters: dict[str, Any] | None = None,
        document_id: str | None = None,
        user_id: str | None = None,
    ) -> RAGResponse:
        """Asynchronously retrieve context, prompt LLM, and return structured grounded response.

        Args:
            query: Question to answer.
            k: Number of context passages to retrieve.
            filters: Optional metadata filters.
            document_id: Optional document scope filter.
            user_id: Optional user multi-tenancy filter.

        Returns:
            RAGResponse: Structured answer with resolved citations.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return RAGResponse(
                query=query,
                answer="I don't know based on the provided context.",
                citations=[],
                context_found=False,
            )

        merged_filters = self._resolve_filters(filters, document_id, user_id)
        search_results = self.vector_store.similarity_search(
            query=cleaned_query,
            k=k,
            filters=merged_filters,
        )

        if not search_results:
            return RAGResponse(
                query=cleaned_query,
                answer="I don't know based on the provided context.",
                citations=[],
                context_found=False,
            )

        user_prompt, source_map = self._build_context_and_prompt(
            cleaned_query, search_results
        )

        raw_answer = (
            await self.llm_client.agenerate(
                prompt=user_prompt,
                system_prompt=self.system_prompt,
            )
        ).strip()

        is_unknown = any(phrase in raw_answer.lower() for phrase in UNKNOWN_PHRASES)
        if is_unknown:
            return RAGResponse(
                query=cleaned_query,
                answer=raw_answer,
                citations=[],
                context_found=False,
            )

        citations = self._parse_citations(raw_answer, source_map)

        return RAGResponse(
            query=cleaned_query,
            answer=raw_answer,
            citations=citations,
            context_found=True,
        )

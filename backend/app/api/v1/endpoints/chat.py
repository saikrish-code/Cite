"""Chat REST API endpoints with Server-Sent Events streaming.

Provides:
- POST /chat — Stream a RAG answer token-by-token via SSE, then send citations
- GET /chat/history — Retrieve paginated chat history
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatHistoryEntry, ChatRequest
from app.services.document_store import chat_store
from app.services.llm import MockLLMClient, get_llm_client
from app.services.rag import RAGService
from app.services.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

router = APIRouter()


async def _sse_event_generator(
    request: ChatRequest,
) -> AsyncIterator[str]:
    """Generate SSE-formatted events from the RAG streaming pipeline.

    Produces events in the standard SSE format:
        event: <event_type>
        data: <json_payload>

    Args:
        request: Validated chat request payload.

    Yields:
        str: SSE-formatted event strings.
    """
    try:
        llm_client = get_llm_client()
        vector_store = ChromaVectorStore()
        rag_service = RAGService(
            vector_store=vector_store,
            llm_client=llm_client,
        )

        user_id = request.user_id or "anonymous"

        async for event_dict in rag_service.astream_answer(
            query=request.query,
            k=request.k,
            document_id=request.document_id,
            user_id=user_id,
        ):
            event_type = event_dict["event"]
            event_data = json.dumps(event_dict["data"])
            yield f"event: {event_type}\ndata: {event_data}\n\n"

            # After done event, save to chat history
            if event_type == "done":
                done_data = event_dict["data"]
                # Collect citations from the previous event (stored temporarily)
                chat_store.add_entry(
                    query=done_data.get("query", request.query),
                    answer=done_data.get("answer", ""),
                )

    except Exception as exc:
        logger.exception("SSE streaming error for query: %s", request.query)
        error_data = json.dumps({"detail": str(exc)})
        yield f"event: error\ndata: {error_data}\n\n"


@router.post(
    "",
    status_code=status.HTTP_200_OK,
    summary="Chat with Documents (SSE Stream)",
    description=(
        "Submit a question and receive a streaming Server-Sent Events response. "
        "Events are streamed in this order: multiple 'token' events (one per generated token), "
        "then a single 'citations' event with resolved source references, "
        "then a final 'done' event with the complete answer."
    ),
    response_description="SSE stream with event types: token, citations, done, error",
)
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream a RAG answer using Server-Sent Events.

    The response streams token events as the LLM generates them, followed by
    citations resolved from the accumulated answer, and finally a done event.

    Args:
        request: Chat request with query and optional filters.

    Returns:
        StreamingResponse: SSE event stream with content-type text/event-stream.
    """
    return StreamingResponse(
        content=_sse_event_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/history",
    response_model=list[ChatHistoryEntry],
    status_code=status.HTTP_200_OK,
    summary="Chat History",
    description="Retrieve paginated chat history entries, ordered by most recent first.",
)
async def get_chat_history(
    limit: int = 20,
    offset: int = 0,
) -> list[ChatHistoryEntry]:
    """Retrieve chat history with pagination.

    Args:
        limit: Maximum entries to return (query param, default 20).
        offset: Number of entries to skip (query param, default 0).

    Returns:
        list[ChatHistoryEntry]: Paginated history entries.
    """
    return chat_store.list_entries(limit=limit, offset=offset)

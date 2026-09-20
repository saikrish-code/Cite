"""Chat REST API endpoints with Server-Sent Events streaming and user isolation.

Provides:
- POST /chat — Stream a user-isolated RAG answer token-by-token via SSE
- GET /chat/history — Retrieve paginated chat history for the authenticated caller
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.core.database import AsyncSessionLocal, get_db
from app.models.chat import ChatMessage
from app.models.document import Document
from app.models.user import User
from app.schemas.chat import ChatHistoryEntry, ChatRequest
from app.services.document_store import chat_store
from app.services.llm import get_llm_client
from app.services.rag import RAGService
from app.services.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

router = APIRouter()


async def _save_chat_message_db(
    user_id: str,
    query: str,
    answer: str,
    citations: list[dict[str, Any]],
    context_found: bool,
    document_id: str | None = None,
) -> None:
    """Persist completed chat Q&A exchange to the database."""
    try:
        async with AsyncSessionLocal() as session:
            msg = ChatMessage(
                user_id=user_id,
                document_id=document_id,
                query=query,
                answer=answer,
                citations=citations,
                context_found=context_found,
            )
            session.add(msg)
            await session.commit()
    except Exception as exc:
        logger.exception("Failed to persist chat message in database: %s", exc)


async def _sse_event_generator(
    query: str,
    user_id: str,
    document_id: str | None = None,
    k: int = 4,
    retrieval_mode: str | None = None,
    history: list[ChatHistoryEntry] | None = None,
) -> AsyncIterator[str]:
    """Generate SSE-formatted events from RAG answer stream.

    Yields raw SSE strings formatted as:
    ``event: {type}\ndata: {json}\n\n``
    """
    try:
        llm_client = get_llm_client()
        vector_store = ChromaVectorStore()
        rag_service = RAGService(
            vector_store=vector_store,
            llm_client=llm_client,
        )
        last_citations: list[dict[str, Any]] = []
        last_context_found = True

        async for event_dict in rag_service.astream_answer(
            query=query,
            k=k,
            document_id=document_id,
            user_id=user_id,
            retrieval_mode=retrieval_mode,
            history=history,
        ):
            event_type = event_dict["event"]
            event_data = json.dumps(event_dict["data"])
            yield f"event: {event_type}\ndata: {event_data}\n\n"

            if event_type == "citations":
                last_citations = event_dict["data"].get("citations", [])
                last_context_found = event_dict["data"].get("context_found", True)

            # After done event, save to DB and in-memory chat store
            if event_type == "done":
                done_data = event_dict["data"]
                final_answer = done_data.get("answer", "")
                final_query = done_data.get("query", query)

                # Persist in DB
                await _save_chat_message_db(
                    user_id=user_id,
                    query=final_query,
                    answer=final_answer,
                    citations=last_citations,
                    context_found=last_context_found,
                    document_id=document_id,
                )

                # Update in-memory store
                chat_store.add_entry(
                    query=final_query,
                    answer=final_answer,
                )

    except Exception as exc:
        logger.exception("SSE streaming error for query: %s", query)
        error_data = json.dumps({"detail": str(exc)})
        yield f"event: error\ndata: {error_data}\n\n"


@router.post(
    "",
    status_code=status.HTTP_200_OK,
    summary="Chat with Documents (User-Isolated SSE Stream)",
    description=(
        "Submit a question and receive a streaming Server-Sent Events response. "
        "Vector retrieval is strictly constrained to the authenticated user's documents."
    ),
    response_description="SSE stream with event types: token, citations, done, error",
)
async def chat_stream(
    request: ChatRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StreamingResponse:
    """Stream a RAG answer isolated strictly to the authenticated caller."""
    # If a specific document is requested, verify it belongs to current_user
    if request.document_id:
        stmt = select(Document).where(
            Document.id == request.document_id,
            Document.user_id == current_user.id,
        )
        res = await db.execute(stmt)
        doc = res.scalar_one_or_none()
        if doc is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{request.document_id}' not found or unauthorized.",
            )

    # Fetch recent history for context resolution
    hist_stmt = (
        select(ChatMessage)
        .where(ChatMessage.user_id == current_user.id)
        .order_by(ChatMessage.created_at.desc())
    )
    if request.document_id:
        hist_stmt = hist_stmt.where(ChatMessage.document_id == request.document_id)
        
    hist_res = await db.execute(hist_stmt.limit(5))
    records = hist_res.scalars().all()
    # Reverse to chronological order
    history = [
        ChatHistoryEntry(
            id=r.id,
            query=r.query,
            answer=r.answer,
            citations=r.citations,
            context_found=r.context_found,
            created_at=r.created_at,
        ) for r in reversed(records)
    ]

    return StreamingResponse(
        content=_sse_event_generator(
            query=request.query,
            user_id=current_user.id,
            document_id=request.document_id,
            k=request.k,
            retrieval_mode=request.retrieval_mode,
            history=history,
        ),
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
    description="Retrieve paginated chat history entries belonging exclusively to the caller.",
)
async def get_chat_history(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = 20,
    offset: int = 0,
) -> list[ChatHistoryEntry]:
    """Retrieve user-scoped chat history with pagination."""
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.user_id == current_user.id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(stmt)
    records = result.scalars().all()

    return [
        ChatHistoryEntry(
            id=r.id,
            query=r.query,
            answer=r.answer,
            citations=r.citations,
            context_found=r.context_found,
            created_at=r.created_at,
        )
        for r in records
    ]

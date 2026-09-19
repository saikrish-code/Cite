"""Central router registration for API v1."""

from fastapi import APIRouter

from app.api.v1.endpoints import auth, chat, documents, health

api_router = APIRouter()

# Register endpoint sub-routers
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(
    documents.router, prefix="/documents", tags=["Documents"]
)
api_router.include_router(chat.router, prefix="/chat", tags=["Chat"])

"""Central router registration for API v1."""

from fastapi import APIRouter

from app.api.v1.endpoints import health

api_router = APIRouter()

# Register endpoint sub-routers
api_router.include_router(health.router, tags=["Health"])

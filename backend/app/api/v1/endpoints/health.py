"""Health check endpoint implementation."""

from datetime import UTC, datetime

from fastapi import APIRouter, status

from app.core.config import settings
from app.schemas.health import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Returns the operational status, environment, and metadata of the CiteRAG backend.",
)
async def check_health() -> HealthResponse:
    """Evaluate and return service health status.

    Returns:
        HealthResponse: Status payload containing service health, environment, and timestamp.
    """
    return HealthResponse(
        status="ok",
        project_name=settings.PROJECT_NAME,
        environment=settings.ENVIRONMENT,
        version=settings.VERSION,
        timestamp=datetime.now(UTC),
    )

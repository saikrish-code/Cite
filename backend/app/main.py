"""Main FastAPI application factory and ASGI entrypoint for CiteRAG."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints.health import check_health
from app.api.v1.router import api_router
from app.core.config import settings
from app.schemas.health import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycles.

    Args:
        app: FastAPI instance.

    Yields:
        None: Continues execution while the application is active.
    """
    # Startup actions (e.g. database connection pool verification, model warmup)
    yield
    # Shutdown actions (e.g. closing connection pools, flushing telemetry)


def create_application() -> FastAPI:
    """Instantiate and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application instance.
    """
    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Set up CORS middleware
    if settings.cors_origins_list:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins_list,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Mount root health check endpoint directly at /health
    application.add_api_route(
        "/health",
        check_health,
        methods=["GET"],
        response_model=HealthResponse,
        tags=["Health"],
        summary="Root Health Check",
    )

    # Mount API v1 router
    application.include_router(api_router, prefix=settings.API_V1_STR)

    return application


app = create_application()

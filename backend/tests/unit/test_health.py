"""Smoke tests for the health check endpoints."""

import pytest
from httpx import AsyncClient

from app.core.config import settings


@pytest.mark.asyncio
async def test_root_health_check_returns_200(async_client: AsyncClient) -> None:
    """Verify that GET /health returns 200 OK and valid status payload."""
    response = await async_client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["project_name"] == settings.PROJECT_NAME
    assert data["environment"] == settings.ENVIRONMENT
    assert data["version"] == settings.VERSION
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_api_v1_health_check_returns_200(async_client: AsyncClient) -> None:
    """Verify that GET /api/v1/health returns 200 OK and identical health payload."""
    response = await async_client.get(f"{settings.API_V1_STR}/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == settings.VERSION

"""Health check response schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Schema representing service health status and deployment metadata."""

    model_config = ConfigDict(from_attributes=True)

    status: str = Field(default="ok", description="Overall health state of the service")
    project_name: str = Field(..., description="Name of the running project")
    environment: str = Field(
        ..., description="Current environment (e.g. development, production)"
    )
    version: str = Field(..., description="Semantic application version")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of the health check evaluation",
    )
